"""Compare Spearman at K=1 vs K=5 vs K=10 for the same test samples."""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from scipy import stats

from config import (
    MODEL_NAME, NUM_UAVS, SEQ_LENGTH, GAMMA_0,
    SURROGATE_SAVE_PATH, SURROGATE_HIDDEN_DIMS, SURROGATE_DROPOUT,
)
from utils import (
    set_seed, generate_deployment_matrix, generate_channel_state,
    compute_pdp_per_layer,
)
from train_surrogate import SurrogateMLP
from generate_data import HookContext, make_corruption_hook


def main():
    set_seed(999)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
    )
    llm.eval()
    num_layers = llm.config.num_hidden_layers

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokens = tokenizer.encode("\n".join(ds["text"]), return_tensors="np")[0]
    start = np.random.randint(0, len(tokens) - SEQ_LENGTH)
    input_ids = torch.tensor(
        tokens[start:start + SEQ_LENGTH], dtype=torch.long,
    ).unsqueeze(0).to(device)

    ctx = HookContext(num_layers)
    hooks = [
        llm.model.layers[l].register_forward_hook(make_corruption_hook(l, ctx))
        for l in range(num_layers)
    ]

    N = 200
    real_k1, real_k5, real_k10, pred_ppls = [], [], [], []

    for i in range(N):
        X = generate_deployment_matrix(NUM_UAVS, num_layers)
        h = generate_channel_state(NUM_UAVS)
        ctx.update(X, h)

        ppls = []
        with torch.no_grad():
            for _ in range(10):
                out = llm(input_ids=input_ids, labels=input_ids)
                ppls.append(torch.exp(out.loss).item())

        real_k1.append(ppls[0])
        real_k5.append(np.mean(ppls[:5]))
        real_k10.append(np.mean(ppls))

        pdp = compute_pdp_per_layer(X, h, num_layers, GAMMA_0)
        log_att = np.where(pdp > 1e-8, np.log(1.0 - pdp), 0.0)
        n_trans = float((pdp[:-1] > 1e-8).sum())
        feat = np.concatenate([
            X.flatten(), h.flatten(), pdp, log_att,
            [log_att.sum()], [n_trans], [pdp.mean()], [pdp.max()],
        ]).reshape(1, -1)
        feat = (feat - feat_mean) / feat_std
        with torch.no_grad():
            pred_norm = model(
                torch.tensor(feat, dtype=torch.float32).to(device),
            ).item()
        pred_ppls.append(np.exp(pred_norm * target_std + target_mean))

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{N}]")

    for hook in hooks:
        hook.remove()

    real_k1 = np.array(real_k1)
    real_k5 = np.array(real_k5)
    real_k10 = np.array(real_k10)
    pred = np.array(pred_ppls)

    print(f"\n{'='*60}")
    print(f"  Surrogate vs Real PPL at different K (N={N})")
    print(f"{'='*60}")
    for k_name, real in [("K=1 ", real_k1), ("K=5 ", real_k5), ("K=10", real_k10)]:
        sp, _ = stats.spearmanr(real, pred)
        pr, _ = stats.pearsonr(real, pred)
        mape = (np.abs(real - pred) / real).mean() * 100
        within11 = (np.abs(np.log(pred) - np.log(real)) < np.log(1.1)).mean()
        within15 = (np.abs(np.log(pred) - np.log(real)) < np.log(1.5)).mean()
        within20 = (np.abs(np.log(pred) - np.log(real)) < np.log(2)).mean()
        print(f"  {k_name}: Spearman={sp:.4f} Pearson={pr:.4f} "
              f"MAPE={mape:.1f}% "
              f"W1.1x={within11:.1%} W1.5x={within15:.1%} W2x={within20:.1%}")


if __name__ == "__main__":
    main()
