"""Phase 2: Surrogate MLP model training.

Trains a lightweight MLP regressor to predict log(PPL) from the deployment
strategy matrix X, channel state matrix h, and pre-computed PDP features.
Using log(PPL) compresses the target range, and pre-computed PDPs give the
model direct access to the most informative features.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from config import (
    DATA_SAVE_PATH, MODEL_DIR, SURROGATE_SAVE_PATH,
    SURROGATE_HIDDEN_DIMS, SURROGATE_DROPOUT,
    TRAIN_TEST_SPLIT, LEARNING_RATE, BATCH_SIZE,
    MAX_EPOCHS, MAX_TRAINING_MINUTES, TARGET_VAL_MAE, PATIENCE,
    LR_SCHEDULER_PATIENCE, LR_FACTOR, LR_MIN,
    NUM_UAVS, GAMMA_0,
)
from utils import set_seed, compute_pdp_per_layer


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PPLDataset(Dataset):
    """PyTorch Dataset with (X, h, PDP) features and log(PPL) targets."""

    def __init__(self, features: np.ndarray, log_ppl: np.ndarray):
        self.features = torch.from_numpy(features.copy()).float()
        self.targets = torch.from_numpy(log_ppl.copy()).float()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# ---------------------------------------------------------------------------
# Surrogate MLP
# ---------------------------------------------------------------------------

class SurrogateMLP(nn.Module):
    """MLP regressor: (X, h, PDP) → log(PPL).

    Architecture:
        Input → [Linear → BN → ReLU → Dropout] × N → Linear(1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        dropout: float = SURROGATE_DROPOUT,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = SURROGATE_HIDDEN_DIMS

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Early stopping with three termination criteria:

    1. Target reached: val_mae < TARGET_VAL_MAE
    2. Time limit: wall-clock > MAX_TRAINING_MINUTES
    3. Patience: no improvement for PATIENCE epochs

    Always restores best weights on stop.
    """

    def __init__(
        self,
        patience: int = PATIENCE,
        target_mae: float = TARGET_VAL_MAE,
        max_minutes: float = MAX_TRAINING_MINUTES,
    ):
        self.patience = patience
        self.target_mae = target_mae
        self.max_minutes = max_minutes
        self.best_mae = float("inf")
        self.best_weights = None
        self.counter = 0
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def should_stop(self, current_mae: float, model: nn.Module, epoch: int) -> bool:
        elapsed = time.time() - self.start_time

        if current_mae < self.best_mae:
            self.best_mae = current_mae
            self.best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1

        stop_reason = None
        if current_mae < self.target_mae:
            stop_reason = f"Target MAE reached: {current_mae:.4f} < {self.target_mae}"
        elif elapsed > self.max_minutes * 60:
            stop_reason = f"Time limit: {elapsed / 60:.1f} min > {self.max_minutes} min"
        elif self.counter >= self.patience:
            stop_reason = f"Patience exhausted: no improvement for {self.patience} epochs"

        if stop_reason:
            print(f"[EarlyStop] {stop_reason}")

        return stop_reason is not None

    def restore_best_weights(self, model: nn.Module) -> None:
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            print(f"[EarlyStop] Restored best weights (val_mae={self.best_mae:.4f})")


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(X: np.ndarray, h: np.ndarray):
    """Build enhanced features from X, h with PDP and cumulative attenuation."""
    n_samples = X.shape[0]
    X_reshaped = X.reshape(n_samples, NUM_UAVS, -1)
    num_layers = X_reshaped.shape[2]

    # Pre-compute PDP per layer boundary
    pdp_features = np.zeros((n_samples, num_layers), dtype=np.float32)
    for i in range(n_samples):
        pdp_features[i] = compute_pdp_per_layer(
            X_reshaped[i], h[i].reshape(NUM_UAVS, NUM_UAVS), num_layers, GAMMA_0
        )

    # Cumulative attenuation: Σ log(1-PDP) — key predictor for log(PPL)
    log_attenuation = np.zeros((n_samples, num_layers), dtype=np.float32)
    for i in range(n_samples):
        for l in range(num_layers):
            if pdp_features[i, l] > 1e-8:
                log_attenuation[i, l] = np.log(1.0 - pdp_features[i, l])

    # Summary features
    n_transitions = (pdp_features[:, :-1] > 1e-8).sum(axis=1, keepdims=True).astype(np.float32)
    total_log_att = log_attenuation.sum(axis=1, keepdims=True)
    pdp_mean = pdp_features.mean(axis=1, keepdims=True)
    pdp_max = pdp_features.max(axis=1, keepdims=True)

    features = np.concatenate([
        X,                    # deployment matrix (flattened)
        h,                    # channel state (flattened)
        pdp_features,         # PDP per layer boundary
        log_attenuation,      # log(1-PDP) per layer
        total_log_att,        # Σ log(1-PDP) — cumulative attenuation
        n_transitions,        # number of UAV transitions
        pdp_mean,             # mean PDP
        pdp_max,              # max PDP
    ], axis=1)

    return features, num_layers


def load_and_preprocess_data():
    """Load dataset, build features, apply log(PPL), standardize, split."""
    print(f"Loading data from {DATA_SAVE_PATH} ...")
    data = np.load(DATA_SAVE_PATH)
    X, h, ppl = data["X"], data["h"], data["ppl"]
    print(f"Loaded {len(ppl)} samples | PPL range: [{ppl.min():.2f}, {ppl.max():.2f}], "
          f"mean={ppl.mean():.2f}")

    # Build enhanced features
    features, num_layers = build_features(X, h)
    input_dim = features.shape[1]
    print(f"Feature dim: {input_dim} (X={X.shape[1]} + h={h.shape[1]} + "
          f"PDP={num_layers} + 3 stats)")

    # Convert to log(PPL)
    log_ppl = np.log(ppl)
    print(f"log(PPL) range: [{log_ppl.min():.4f}, {log_ppl.max():.4f}], "
          f"mean={log_ppl.mean():.4f}")

    # Shuffle
    indices = np.random.permutation(len(log_ppl))
    features, log_ppl = features[indices], log_ppl[indices]

    # Split
    split_idx = int(len(log_ppl) * TRAIN_TEST_SPLIT)
    feat_train, feat_test = features[:split_idx], features[split_idx:]
    log_ppl_train, log_ppl_test = log_ppl[:split_idx], log_ppl[split_idx:]

    # Standardize features (fit on train only)
    feat_mean = feat_train.mean(axis=0)
    feat_std = feat_train.std(axis=0) + 1e-8

    feat_train = (feat_train - feat_mean) / feat_std
    feat_test = (feat_test - feat_mean) / feat_std

    # Standardize targets (log(PPL)) for stable training
    target_mean = log_ppl_train.mean()
    target_std = log_ppl_train.std() + 1e-8

    log_ppl_train_norm = (log_ppl_train - target_mean) / target_std
    log_ppl_test_norm = (log_ppl_test - target_mean) / target_std

    print(f"Train: {len(log_ppl_train)} | Val: {len(log_ppl_test)}")
    print(f"Target (log PPL) normalized: mean={target_mean:.4f}, std={target_std:.4f}")

    norm_params = {
        "feat_mean": feat_mean, "feat_std": feat_std,
        "target_mean": target_mean, "target_std": target_std,
        "input_dim": input_dim,
    }
    return (feat_train, log_ppl_train_norm), (feat_test, log_ppl_test_norm), norm_params


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_surrogate():
    set_seed(42)
    os.makedirs(MODEL_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    (feat_train, log_ppl_train), (feat_test, log_ppl_test), norm_params = \
        load_and_preprocess_data()

    train_dataset = PPLDataset(feat_train, log_ppl_train)
    test_dataset = PPLDataset(feat_test, log_ppl_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = norm_params["input_dim"]

    # Model
    model = SurrogateMLP(input_dim=input_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"SurrogateMLP parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=LR_MIN,
    )
    criterion = nn.MSELoss()

    target_mean = norm_params["target_mean"]
    target_std = norm_params["target_std"]

    # Early stopping on de-normalized log(PPL) MAE
    early_stopper = EarlyStopping(target_mae=TARGET_VAL_MAE)
    early_stopper.start()

    best_val_mae = float("inf")

    for epoch in range(MAX_EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0.0
        train_samples = 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            preds = model(features)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(targets)
            train_samples += len(targets)

        train_loss_avg = train_loss / train_samples

        # --- Validate ---
        model.eval()
        val_mae_sum = 0.0
        val_mse_sum = 0.0
        val_samples = 0
        with torch.no_grad():
            for features, targets in test_loader:
                features, targets = features.to(device), targets.to(device)
                preds = model(features)
                # De-normalize to log(PPL) scale for reporting
                preds_real = preds * target_std + target_mean
                targets_real = targets * target_std + target_mean
                val_mae_sum += (preds_real - targets_real).abs().sum().item()
                val_mse_sum += ((preds_real - targets_real) ** 2).sum().item()
                val_samples += len(targets)

        val_mae = val_mae_sum / val_samples
        val_mse = val_mse_sum / val_samples
        elapsed = time.time() - early_stopper.start_time

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            marker = " *"
        else:
            marker = ""

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1:4d}/{MAX_EPOCHS} | "
                  f"train_loss={train_loss_avg:.4f} | "
                  f"val_mse={val_mse:.4f} | val_mae={val_mae:.4f}{marker} | "
                  f"lr={lr:.1e} | time={elapsed/60:.1f}min")

        scheduler.step()

        if early_stopper.should_stop(val_mae, model, epoch):
            break

    # Restore best weights
    early_stopper.restore_best_weights(model)

    # Final evaluation
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            preds = model(features)
            all_preds.append(preds.cpu())
            all_targets.append(targets)

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    # De-normalize
    all_preds_log = all_preds * target_std + target_mean
    all_targets_log = all_targets * target_std + target_mean

    log_mae = np.abs(all_preds_log - all_targets_log).mean()
    log_mse = ((all_preds_log - all_targets_log) ** 2).mean()
    ppl_preds = np.exp(all_preds_log)
    ppl_targets = np.exp(all_targets_log)
    ppl_mae = np.abs(ppl_preds - ppl_targets).mean()
    ppl_mape = (np.abs(ppl_preds - ppl_targets) / ppl_targets).mean() * 100

    print(f"\n=== Final Evaluation (best weights) ===")
    print(f"log(PPL) MAE:  {log_mae:.4f}")
    print(f"log(PPL) MSE:  {log_mse:.4f}")
    print(f"PPL MAE:       {ppl_mae:.0f}")
    print(f"PPL MAPE:      {ppl_mape:.1f}%")

    # Save checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dims": SURROGATE_HIDDEN_DIMS,
        "dropout": SURROGATE_DROPOUT,
        "use_log_target": True,
        "num_uavs": NUM_UAVS,
        **{k: (v.tolist() if isinstance(v, np.ndarray) else v)
           for k, v in norm_params.items()},
    }
    torch.save(checkpoint, SURROGATE_SAVE_PATH)
    print(f"Model saved to {SURROGATE_SAVE_PATH}")


if __name__ == "__main__":
    train_surrogate()
