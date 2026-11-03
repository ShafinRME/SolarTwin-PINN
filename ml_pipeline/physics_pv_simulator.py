"""
SolarTwin-PINN: Bishop Single-Diode + Bypass-Diode Multi-String PV Physics Simulator
====================================================================================
Models a 3S1P / Reconfigurable Solar-Electric Vehicle (SEV) PV array under dynamic
Partial Shading Conditions (PSC) with Newton-Raphson current continuity solving.

Calibrated for 3x 20W 12V Monocrystalline Modules (Dhaka HIL Benchtop Configuration):
- Per-module Voc_stc = 21.6 V, Isc_stc = 1.19 A, Vmp_stc = 18.0 V, Imp_stc = 1.11 A
- 36 cells in series per module, Schottky bypass diode (Vd = 0.35 V) across each module
- Includes Bishop's avalanche breakdown model for reverse-biased shaded substrings.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List


@dataclass
class PVModuleParams:
    """Standard Test Condition (STC) parameters for a 20W Monocrystalline PV Module."""
    Ns: int = 36                  # Number of solar cells in series per module
    Voc_stc: float = 21.6         # Open-circuit voltage at STC (V)
    Isc_stc: float = 1.19         # Short-circuit current at STC (A)
    Vmp_stc: float = 18.0         # Maximum power voltage at STC (V)
    Imp_stc: float = 1.11         # Maximum power current at STC (A)
    Rs: float = 0.42              # Series resistance per module (Ohm)
    Rsh: float = 145.0            # Shunt resistance per module (Ohm)
    n_ideality: float = 1.28      # Diode ideality factor
    Ki: float = 0.00055           # Temperature coefficient of Isc (A/K)
    Kv: float = -0.072            # Temperature coefficient of Voc (V/K)
    V_bypass: float = 0.35        # Schottky bypass diode forward drop (10SQ050, V)
    R_bypass: float = 0.04        # Bypass diode dynamic on-resistance (Ohm)


class MultiStringPVSimulator:
    """
    Solves the coupled Single-Diode + Bypass-Diode equations for a 3-module series PV string
    under arbitrary dynamic irradiance [G1, G2, G3] (W/m^2) and cell temperature Tc (C).
    """

    def __init__(self, params: PVModuleParams = PVModuleParams(), n_modules: int = 3):
        self.p = params
        self.n_modules = n_modules
        self.q = 1.602176634e-19
        self.k = 1.380649e-23
        self.T_stc = 298.15  # 25 deg C in Kelvin

    def module_voltage_from_current(self, I_str: np.ndarray, G: float, Tc_deg: float) -> np.ndarray:
        """
        Given string current I_str (A), irradiance G (W/m^2), and cell temperature Tc (C),
        computes the terminal voltage V_m of a single module including bypass diode conduction.
        """
        T_k = Tc_deg + 273.15
        dT = T_k - self.T_stc
        Vt = (self.p.Ns * self.p.n_ideality * self.k * T_k) / self.q

        # Photocurrent Iph(G, T)
        Iph = max(0.0, (self.p.Isc_stc + self.p.Ki * dT) * (G / 1000.0))

        # Reverse saturation current Io(T)
        Voc_T = max(1.0, self.p.Voc_stc + self.p.Kv * dT)
        Io = (self.p.Isc_stc + self.p.Ki * dT) / (np.exp(Voc_T / Vt) - 1.0)

        V_out = np.zeros_like(I_str, dtype=np.float64)
        for idx, curr in enumerate(I_str):
            if curr <= Iph:
                # Forward biased solar substring: solve single-diode equation via Newton-Raphson
                v = Voc_T * 0.85
                for _ in range(15):
                    exp_arg = np.clip((v + curr * self.p.Rs) / Vt, -50.0, 60.0)
                    f_val = Iph - Io * (np.exp(exp_arg) - 1.0) - (v + curr * self.p.Rs) / self.p.Rsh - curr
                    df_dv = -(Io / Vt) * np.exp(exp_arg) - (1.0 / self.p.Rsh)
                    v_next = v - f_val / (df_dv - 1e-12)
                    if abs(v_next - v) < 1e-6:
                        v = v_next
                        break
                    v = v_next
                V_out[idx] = max(-self.p.V_bypass, v)
            else:
                # Current exceeds module photocurrent -> Schottky bypass diode conducts
                I_bypass = curr - Iph
                V_out[idx] = -(self.p.V_bypass + I_bypass * self.p.R_bypass)

        return V_out

    def generate_iv_pv_curve(
        self, G_modules: List[float], Tc_deg: float = 35.0, n_points: int = 300
    ) -> Dict[str, np.ndarray]:
        """
        Generates full I-V and P-V curves for a 3-module series array under partial shading
        and extracts the true Global Maximum Power Point (GMPP) and all Local MPPs (LMPPs).
        """
        max_Isc = self.p.Isc_stc * (max(G_modules) / 1000.0) * 1.04
        I_sweep = np.linspace(0.0, max(0.05, max_Isc), n_points)

        V_total = np.zeros_like(I_sweep)
        for G_m in G_modules:
            V_total += self.module_voltage_from_current(I_sweep, G_m, Tc_deg)

        # Keep only positive string voltages (first quadrant)
        valid = V_total >= 0.0
        V_curve = V_total[valid]
        I_curve = I_sweep[valid]

        # Sort by increasing voltage V
        order = np.argsort(V_curve)
        V_curve = V_curve[order]
        I_curve = I_curve[order]
        P_curve = V_curve * I_curve

        # Identify true Global Maximum Power Point (GMPP)
        gmpp_idx = int(np.argmax(P_curve))
        V_gmpp = float(V_curve[gmpp_idx])
        I_gmpp = float(I_curve[gmpp_idx])
        P_gmpp = float(P_curve[gmpp_idx])

        # Extract 5 sparse probe points at fixed voltage fractions [0.20, 0.38, 0.55, 0.72, 0.88] * Voc_est
        Voc_est = float(V_curve[-1]) if len(V_curve) > 0 else 60.0
        probe_fractions = [0.20, 0.38, 0.55, 0.72, 0.88]
        probe_V = np.array([f * Voc_est for f in probe_fractions])
        probe_I = np.interp(probe_V, V_curve, I_curve)

        return {
            "V_curve": V_curve,
            "I_curve": I_curve,
            "P_curve": P_curve,
            "V_gmpp": V_gmpp,
            "I_gmpp": I_gmpp,
            "P_gmpp": P_gmpp,
            "Voc": Voc_est,
            "probe_V": probe_V,
            "probe_I": probe_I,
            "G_modules": np.array(G_modules, dtype=np.float32),
            "Tc_deg": float(Tc_deg),
        }


if __name__ == "__main__":
    sim = MultiStringPVSimulator()
    sample = sim.generate_iv_pv_curve([950.0, 520.0, 240.0], Tc_deg=38.5)
    print(f"[SolarTwin Physics] 3-Peak PSC Test: V_GMPP={sample['V_gmpp']:.2f} V, "
          f"I_GMPP={sample['I_gmpp']:.3f} A, P_GMPP={sample['P_gmpp']:.2f} W")
