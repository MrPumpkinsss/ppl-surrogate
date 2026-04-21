"""Utility functions for PPL surrogate model project."""

import random
from typing import Tuple

import numpy as np
import torch

from config import GAMMA_0, SNR_SCALE


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_deployment_matrix(num_uavs: int, num_layers: int) -> np.ndarray:
    """Generate a random deployment matrix X of shape (num_uavs, num_layers).

    Each column is a one-hot vector: each layer is assigned to exactly one UAV.
    """
    assignments = np.random.randint(0, num_uavs, size=num_layers)
    X = np.zeros((num_uavs, num_layers), dtype=np.float32)
    X[assignments, np.arange(num_layers)] = 1.0
    return X


def generate_channel_state(num_uavs: int) -> np.ndarray:
    """Generate a random channel state matrix h of shape (num_uavs, num_uavs).

    Diagonal = 1.0 (perfect self-communication).
    Off-diagonal = random in [0.1, 1.0] (UAV-to-UAV channel gains).
    """
    h = np.random.uniform(0.1, 1.0, size=(num_uavs, num_uavs)).astype(np.float32)
    np.fill_diagonal(h, 1.0)
    return h


def compute_pdp(gamma_0: float, channel_gain: float) -> float:
    """Compute Packet Drop Probability: PDP = 1 - exp(-gamma_0 / SNR).

    SNR = channel_gain * SNR_SCALE to keep PDP in a reasonable range.
    """
    snr = channel_gain * SNR_SCALE
    pdp = 1.0 - np.exp(-gamma_0 / snr)
    return float(np.clip(pdp, 0.0, 0.95))


def compute_pdp_per_layer(
    X: np.ndarray, h: np.ndarray, num_layers: int, gamma_0: float = GAMMA_0
) -> np.ndarray:
    """Pre-compute PDP for each layer boundary.

    Returns shape (num_layers,) where pdp[l] is the packet drop probability
    between layer l and layer l+1. If they are on the same UAV, pdp[l] = 0.
    """
    layer_to_uav = np.argmax(X, axis=0)  # shape (num_layers,)
    pdp = np.zeros(num_layers, dtype=np.float32)
    for l in range(num_layers - 1):
        uav_curr = layer_to_uav[l]
        uav_next = layer_to_uav[l + 1]
        if uav_curr != uav_next:
            pdp[l] = compute_pdp(gamma_0, h[uav_curr, uav_next])
    return pdp
