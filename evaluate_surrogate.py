"""Evaluate surrogate model: predicted vs real PPL, ranking accuracy, correlation."""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from scipy import stats

from config import (
    MODEL_NAME, NUM_UAVS, SEQ_LENGTH, GAMMA_0, K_AVERAGING,
    DATA_SAVE_PATH, SURROGATE_SAVE_PATH, SURROGATE_HIDDEN_DIMS, SURROGATE_DROPOUT,
)
from utils import (
    set_seed, generate_deployment_matrix, generate_channel_state,
    compute_pdp_per_layer,
)
from train_surrogate import SurrogateMLP


def evaluate():
    set_seed(999)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load surrogate model ──
    ckpt = torch.load(SURROGATE_SAVE_PATH, weights_only=False)
    model = SurrogateMLP(
        input_dim=ckpt["input_dim"],
        hidden_dims=ckpt.get("hidden_dims", SURROGATE_HIDDEN_DIMS),
        dropout=ckpt.get("dropout", SURROGATE_DROPOUT),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    feat_mean = np.array(ckpt["feat_mean"])
    feat_std = np.array(ckpt["feat_std"])
    target_mean = ckpt.get("target_mean", 0.0)
    target_std = ckpt.get("target_std", 1.0)

    print(f"Surrogate model loaded (input_dim={ckpt['input_dim']})")

    # ── Generate fresh test samples ──
    N_TEST = 200
    print(f"\nGenerating {N_TEST} fresh test samples with REAL LLM inference...")

    # Load LLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16,
                                                device_map="auto")
    llm.eval()
    num_layers = llm.config.num_hidden_layers

    # Load a text chunk
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokens = tokenizer.encode("\n".join(ds["text"]), return_tensors="np")[0]
    start = np.random.randint(0, len(tokens) - SEQ_LENGTH)
    input_ids = torch.tensor(tokens[start:start + SEQ_LENGTH], dtype=torch.long).unsqueeze(0).to(device)

    # Forward hook setup (same as generate_data.py)
    from generate_data import HookContext, make_corruption_hook

    ctx = HookContext(num_layers)
    hooks = []
    for l in range(num_layers):
        hooks.append(llm.model.layers[l].register_forward_hook(make_corruption_hook(l, ctx)))

    real_ppls = []
    pred_ppls = []

    for i in range(N_TEST):
        X = generate_deployment_matrix(NUM_UAVS, num_layers)
        h = generate_channel_state(NUM_UAVS)
        ctx.update(X, h)

        # Real PPL (K=5 averaged, same as training)
        ppls_k = []
        with torch.no_grad():
            for _ in range(K_AVERAGING):
                out = llm(input_ids=input_ids, labels=input_ids)
                ppls_k.append(torch.exp(out.loss).item())
        real_ppl = np.mean(ppls_k)
        real_ppls.append(real_ppl)

        # Surrogate prediction — must match train_surrogate.build_features
        pdp = compute_pdp_per_layer(X, h, num_layers, GAMMA_0)
        log_att = np.where(pdp > 1e-8, np.log(1.0 - pdp), 0.0)
        n_trans = float((pdp[:-1] > 1e-8).sum())
        feat = np.concatenate([
            X.flatten(), h.flatten(), pdp, log_att,
            [log_att.sum()], [n_trans], [pdp.mean()], [pdp.max()],
        ]).reshape(1, -1)
        feat = (feat - feat_mean) / feat_std
        with torch.no_grad():
            pred_norm = model(torch.tensor(feat, dtype=torch.float32).to(device)).item()
        log_ppl_pred = pred_norm * target_std + target_mean
        pred_ppls.append(np.exp(log_ppl_pred))

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{N_TEST}] done")

    for hook in hooks:
        hook.remove()

    real = np.array(real_ppls)
    pred = np.array(pred_ppls)
    log_real = np.log(real)
    log_pred = np.log(pred)

    # ── Metrics ──
    mae = np.abs(pred - real).mean()
    mape = (np.abs(pred - real) / real).mean() * 100
    log_mae = np.abs(log_pred - log_real).mean()

    # Pearson & Spearman correlation
    pearson_r, pearson_p = stats.pearsonr(real, pred)
    spearman_r, spearman_p = stats.spearmanr(real, pred)

    # Ranking accuracy: for random pairs, does surrogate correctly rank them?
    n_pairs = 5000
    idx_a = np.random.randint(0, N_TEST, n_pairs)
    idx_b = np.random.randint(0, N_TEST, n_pairs)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]
    real_order = (real[idx_a] > real[idx_b]).astype(int)
    pred_order = (pred[idx_a] > pred[idx_b]).astype(int)
    rank_acc = (real_order == pred_order).mean()

    # Within-factor accuracy: is prediction within 2x of real?
    within_2x = (np.abs(log_pred - log_real) < np.log(2)).mean()
    within_1_5x = (np.abs(log_pred - log_real) < np.log(1.5)).mean()

    print(f"\n{'='*55}")
    print(f"  Surrogate Model Evaluation ({N_TEST} fresh samples)")
    print(f"{'='*55}")
    print(f"  PPL MAE:              {mae:>15,.0f}")
    print(f"  PPL MAPE:             {mape:>14.1f}%")
    print(f"  log(PPL) MAE:         {log_mae:>14.4f}")
    print(f"  Pearson r:            {pearson_r:>14.4f}  (p={pearson_p:.2e})")
    print(f"  Spearman ρ:           {spearman_r:>14.4f}  (p={spearman_p:.2e})")
    print(f"  Ranking Accuracy:     {rank_acc:>13.1%}  (random pairs)")
    print(f"  Within 1.5x:          {within_1_5x:>13.1%}")
    print(f"  Within 2x:            {within_2x:>13.1%}")
    print(f"{'='*55}")

    # Show some examples
    print(f"\n  Sample predictions (sorted by real PPL):")
    order = np.argsort(real)
    print(f"  {'Real PPL':>14s}  {'Pred PPL':>14s}  {'Ratio':>8s}  {'log err':>8s}")
    for idx in [order[0], order[N_TEST//4], order[N_TEST//2], order[3*N_TEST//4], order[-1]]:
        ratio = pred[idx] / real[idx]
        log_err = abs(log_pred[idx] - log_real[idx])
        print(f"  {real[idx]:>14,.0f}  {pred[idx]:>14,.0f}  {ratio:>7.2f}x  {log_err:>7.3f}")


if __name__ == "__main__":
    evaluate()
