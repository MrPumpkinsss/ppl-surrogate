"""Shared configuration constants for PPL surrogate model project."""

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen3-0.6B"

# ---------------------------------------------------------------------------
# UAV deployment
# ---------------------------------------------------------------------------
NUM_UAVS = 5

# ---------------------------------------------------------------------------
# Phase 1 – Data collection
# ---------------------------------------------------------------------------
SEQ_LENGTH = 512          # Fixed token length per sample
NUM_SAMPLES = 20000       # Number of (X, h, PPL) samples to collect
GAMMA_0 = 1.0             # Reference SNR parameter for PDP computation
SNR_SCALE = 10.0          # Scale factor: SNR = channel_gain * SNR_SCALE
NUM_CHUNKS = 50           # Number of pre-extracted WikiText-2 token chunks
K_AVERAGING = 10          # Forward passes per (X, h) to average out dropout noise
DATA_DIR = "data"
DATA_SAVE_PATH = f"{DATA_DIR}/dataset.npz"

# ---------------------------------------------------------------------------
# Phase 2 – Surrogate model training
# ---------------------------------------------------------------------------
SURROGATE_INPUT_DIM = 165  # 5*28 (X) + 5*5 (h) = 140 + 25
SURROGATE_HIDDEN_DIMS = [512, 256, 128, 64]
SURROGATE_DROPOUT = 0.05
TRAIN_TEST_SPLIT = 0.8
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 2000          # Upper bound; early stopping will terminate sooner
MAX_TRAINING_MINUTES = 30
TARGET_VAL_MAE = 0.05
PATIENCE = 50
LR_SCHEDULER_PATIENCE = 15
LR_FACTOR = 0.5
LR_MIN = 1e-6
MODEL_DIR = "models"
SURROGATE_SAVE_PATH = f"{MODEL_DIR}/surrogate_mlp.pth"
