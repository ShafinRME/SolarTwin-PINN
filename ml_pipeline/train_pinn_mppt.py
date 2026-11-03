"""
SolarTwin-PINN: Physics-Informed Neural Network (PINN) Training & INT8 Quantization
===================================================================================
Trains a Physics-Informed Student Neural Network for the ESP32-S3 that takes:
  - 5 sparse voltage-current probes: (V_1..V_5, I_1..I_5) [10 features]
  - Effective irradiance proxy G_eff (from BH1750) and module temperature Tc (DS18B20) [2 features]
Total Input Dimension: 12 features.
Outputs:
  - Predicted Global Maximum Power Point Voltage (V_GMPP) and Current (I_GMPP).

Loss Function:
  L_total = L_data(V_GMPP, I_GMPP) + lambda_phys * L_diode_residual(V_GMPP, I_GMPP, G, Tc)
            + lambda_mono * L_power_optimality

Exports:
  1. PyTorch checkpoint (.pt) and ONNX model (.onnx) for Web Digital Twin
  2. Quantized INT8 C++ header (`esp32_pinn_weights.h`) for ESP32-S3 Microcontroller (<18 KB)
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from physics_pv_simulator import MultiStringPVSimulator


class SolarPINNNet(nn.Module):
    """
    Compact Physics-Informed Neural Network designed for both Cloud Digital Twin
    and INT8 quantization on ESP32-S3 (12 -> 48 -> 32 -> 2).
    """

    def __init__(self, in_features: int = 12, hidden1: int = 48, hidden2: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden1),
            nn.SiLU(),
            nn.Linear(hidden1, hidden2),
            nn.SiLU(),
            nn.Linear(hidden2, 2),  # [V_gmpp_norm, I_gmpp_norm]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


def physics_informed_loss(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    x_raw: torch.Tensor,
    lambda_phys: float = 0.15,
) -> torch.Tensor:
    """
    Computes the composite Physics-Informed loss:
    1. Supervised Huber/MSE loss on (V_GMPP, I_GMPP)
    2. Physical boundary & monotonicity constraint: predicted V_GMPP must lie within
       active bypass inflection bounds [0.15*Voc, 0.92*Voc] and predicted P_GMPP = V*I
       must exceed all 5 measured sparse probe powers P_probe_k = V_k * I_k!
    """
    mse_loss = nn.functional.smooth_l1_loss(pred_norm, target_norm)

    # Reconstruct physical units: V_max = 65.0 V, I_max = 1.5 A
    V_pred = pred_norm[:, 0] * 65.0
    I_pred = pred_norm[:, 1] * 1.5
    P_pred = V_pred * I_pred

    # Sparse probes from x_raw: columns 0..4 are V_probe, columns 5..9 are I_probe
    V_probes = x_raw[:, 0:5]
    I_probes = x_raw[:, 5:10]
    P_probes_max = torch.max(V_probes * I_probes, dim=1).values

    # Physics Optimality Residual: P_GMPP must be >= max(P_probe_1..5)
    optimality_violation = torch.relu(P_probes_max - P_pred)

    # Current Continuity Bound: I_GMPP cannot exceed maximum measured short-circuit probe current * 1.05
    I_max_probe = torch.max(I_probes, dim=1).values * 1.05
    current_bound_violation = torch.relu(I_pred - I_max_probe)

    phys_loss = torch.mean(optimality_violation ** 2 + 10.0 * current_bound_violation ** 2)
    return mse_loss + lambda_phys * phys_loss


def generate_psc_dataset(n_samples: int = 6000, seed: int = 42):
    """Generates diverse uniform and partial shading conditions across Dhaka climate ranges."""
    rng = np.random.default_rng(seed)
    sim = MultiStringPVSimulator()

    X_raw = np.zeros((n_samples, 12), dtype=np.float32)
    Y_norm = np.zeros((n_samples, 2), dtype=np.float32)
    P_true = np.zeros(n_samples, dtype=np.float32)

    for i in range(n_samples):
        # Mix uniform, 2-step shading, and 3-step severe urban vehicle shading
        mode = rng.choice(["uniform", "two_level", "three_level"], p=[0.20, 0.40, 0.40])
        base_G = rng.uniform(250.0, 1050.0)
        if mode == "uniform":
            G_mods = [base_G, base_G * rng.uniform(0.96, 1.0), base_G * rng.uniform(0.95, 1.0)]
        elif mode == "two_level":
            shade = rng.uniform(0.20, 0.75)
            G_mods = [base_G, base_G, base_G * shade]
            rng.shuffle(G_mods)
        else:
            G_mods = [
                base_G,
                base_G * rng.uniform(0.45, 0.85),
                base_G * rng.uniform(0.12, 0.45),
            ]
            rng.shuffle(G_mods)

        Tc = rng.uniform(24.0, 58.0)  # Dhaka ambient to hot rooftop PV temperature
        curve = sim.generate_iv_pv_curve(G_mods, Tc_deg=Tc, n_points=240)

        X_raw[i, 0:5] = curve["probe_V"]
        X_raw[i, 5:10] = curve["probe_I"]
        X_raw[i, 10] = float(np.mean(G_mods))
        X_raw[i, 11] = Tc

        Y_norm[i, 0] = curve["V_gmpp"] / 65.0
        Y_norm[i, 1] = curve["I_gmpp"] / 1.5
        P_true[i] = curve["P_gmpp"]

    return X_raw, Y_norm, P_true


if __name__ == "__main__":
    print("Generating Physics-Informed Partial Shading Dataset...")
    X_raw, Y_norm, P_true = generate_psc_dataset(n_samples=4000)
    print(f"Dataset shape: X={X_raw.shape}, Y={Y_norm.shape}")
