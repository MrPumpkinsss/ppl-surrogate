"""Phase 1: Offline data collection via error injection.

Generates (X, h, PPL) samples by simulating distributed LLM deployment
across UAVs and injecting channel-induced errors via forward hooks.
"""

import os
import sys
import time
import signal

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    MODEL_NAME, NUM_UAVS, SEQ_LENGTH, NUM_SAMPLES,
    GAMMA_0, NUM_CHUNKS, K_AVERAGING, DATA_DIR, DATA_SAVE_PATH,
)
from utils import (
    set_seed, generate_deployment_matrix, generate_channel_state,
    compute_pdp_per_layer,
)


# ---------------------------------------------------------------------------
# Hook mechanism
# ---------------------------------------------------------------------------

class HookContext:
    """Mutable context shared by all forward hooks during one forward pass.

    Updated before each sample to reflect the current deployment X and
    channel state h. All pre-computed PDPs are stored here so each hook
    simply reads its layer's PDP.
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.pdp_per_layer = np.zeros(num_layers, dtype=np.float32)

    def update(self, X: np.ndarray, h: np.ndarray):
        self.pdp_per_layer = compute_pdp_per_layer(X, h, self.num_layers, GAMMA_0)


def make_corruption_hook(layer_idx: int, ctx: HookContext):
    """Create a forward hook for layer `layer_idx`.

    If this layer's output must traverse a UAV boundary (layer l → l+1 on
    different UAVs), applies dropout with the pre-computed PDP to simulate
    physical-layer packet loss.
    """
    def hook(module, inputs, output):
        pdp = ctx.pdp_per_layer[layer_idx]
        if pdp < 1e-8:
            return output
        corrupted = F.dropout(output.float(), p=float(pdp), training=True) * (1.0 - float(pdp))
        return corrupted.to(output.dtype)
    return hook


# ---------------------------------------------------------------------------
# Model & data loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(device: torch.device):
    """Load Qwen3-0.6B in bfloat16 and its tokenizer."""
    print(f"Loading model {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to(device)
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model, tokenizer


def prepare_wikitext_chunks(tokenizer, seq_length: int, num_chunks: int):
    """Load WikiText-2, tokenize, and extract fixed-length token chunks."""
    print("Loading WikiText-2 dataset ...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n".join(ds["text"])
    tokens = tokenizer.encode(text, return_tensors="np")[0]
    print(f"Total tokens in WikiText-2 test: {len(tokens)}")

    max_start = len(tokens) - seq_length
    if max_start <= 0:
        raise RuntimeError("WikiText-2 test split is too short for the requested seq_length")

    chunk_starts = np.random.choice(max_start, size=min(num_chunks, max_start), replace=False)
    chunks = [torch.tensor(tokens[s : s + seq_length], dtype=torch.long) for s in chunk_starts]
    print(f"Prepared {len(chunks)} chunks of length {seq_length}")
    return chunks


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_data():
    set_seed(42)
    os.makedirs(DATA_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model, tokenizer = load_model_and_tokenizer(device)
    num_layers = model.config.num_hidden_layers
    print(f"Number of hidden layers: {num_layers}")

    chunks = prepare_wikitext_chunks(tokenizer, SEQ_LENGTH, NUM_CHUNKS)

    # Register hooks once on all decoder layers
    ctx = HookContext(num_layers)
    hooks = []
    for l in range(num_layers):
        layer = model.model.layers[l]
        hook_fn = make_corruption_hook(l, ctx)
        handle = layer.register_forward_hook(hook_fn)
        hooks.append(handle)
    print(f"Registered {len(hooks)} forward hooks")

    # Pre-allocate storage
    all_X = np.zeros((NUM_SAMPLES, NUM_UAVS * num_layers), dtype=np.float32)
    all_h = np.zeros((NUM_SAMPLES, NUM_UAVS * NUM_UAVS), dtype=np.float32)
    all_ppl = np.zeros(NUM_SAMPLES, dtype=np.float32)

    collected = 0
    start_time = time.time()

    # Graceful interrupt handler
    interrupted = False
    def _signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
        print(f"\n[Interrupt] Signal received. Saving {collected} samples collected so far ...")
    original_handler = signal.signal(signal.SIGINT, _signal_handler)

    try:
        for i in range(NUM_SAMPLES):
            if interrupted:
                break

            X = generate_deployment_matrix(NUM_UAVS, num_layers)
            h = generate_channel_state(NUM_UAVS)
            ctx.update(X, h)

            input_ids = chunks[i % len(chunks)].unsqueeze(0).to(device)

            try:
                ppls = []
                with torch.no_grad():
                    for _ in range(K_AVERAGING):
                        outputs = model(input_ids=input_ids, labels=input_ids)
                        loss = outputs.loss
                        if loss is not None and torch.isfinite(loss):
                            ppls.append(torch.exp(loss).item())
                ppl = np.mean(ppls) if ppls else float("nan")

                all_X[i] = X.flatten()
                all_h[i] = h.flatten()
                all_ppl[i] = ppl
                collected = i + 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  [OOM] Skipping sample {i}")
                    torch.cuda.empty_cache()
                    continue
                raise

            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (NUM_SAMPLES - i - 1)
                gpu_mem = (f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"
                           if torch.cuda.is_available() else "N/A")
                print(f"[{i+1}/{NUM_SAMPLES}] PPL={ppl:.2f} | "
                      f"GPU={gpu_mem} | Elapsed={elapsed:.0f}s | ETA={eta:.0f}s")
                torch.cuda.empty_cache()

            if (i + 1) % 1000 == 0:
                np.savez(DATA_SAVE_PATH,
                         X=all_X[:collected], h=all_h[:collected], ppl=all_ppl[:collected])
                print(f"  Checkpoint saved ({collected} samples)")

    finally:
        signal.signal(signal.SIGINT, original_handler)
        for hook in hooks:
            hook.remove()
        print("Hooks removed.")

    # Trim to actual collected count
    all_X = all_X[:collected]
    all_h = all_h[:collected]
    all_ppl = all_ppl[:collected]

    # Filter out NaN PPL values
    valid = np.isfinite(all_ppl)
    all_X = all_X[valid]
    all_h = all_h[valid]
    all_ppl = all_ppl[valid]
    print(f"Valid samples: {len(all_ppl)} / {collected}")

    np.savez(DATA_SAVE_PATH, X=all_X, h=all_h, ppl=all_ppl)
    elapsed = time.time() - start_time
    print(f"Done. {len(all_ppl)} samples saved to {DATA_SAVE_PATH} in {elapsed:.0f}s")
    print(f"PPL stats: min={all_ppl.min():.2f}, max={all_ppl.max():.2f}, "
          f"mean={all_ppl.mean():.2f}, median={np.median(all_ppl):.2f}")


if __name__ == "__main__":
    collect_data()
