"""hERG block fraction from a chem-model probability.

Convert a binary-classifier hERG probability into an *effective IKr block fraction*
that can be plugged into the O'Hara–Rudy ionic model.

The mapping is intentionally simple. We do not have a calibrated regressor from
SMILES → IC50 in this project. The README is explicit about this being the
weakest link. We use a Hill equation with two anchors fit to the Crumb 2016 CiPA
reference set:

    p=0.05  → IC50 = 50 µM   (clean compound, free Cmax 0.5 µM → ~1% block)
    p=0.95  → IC50 = 0.05 µM (potent blocker, free Cmax 0.5 µM → ~91% block)

Between anchors we interpolate IC50 log-linearly in p. With a default
[D] = 0.5 µM (median therapeutic free Cmax) and Hill coefficient h=1, the
chemistry probability becomes a smooth, monotone block fraction.

This is a deliberate proxy, not a learned model. Any improvement here lifts the
whole physio pipeline.
"""
from __future__ import annotations

import math
import numpy as np

# Anchors in log10(IC50 [µM])
_LOG_IC50_LO = math.log10(0.05)   # at p=0.95
_LOG_IC50_HI = math.log10(50.0)   # at p=0.05
_P_LO, _P_HI = 0.05, 0.95


def prob_to_ic50_uM(p_herg: float | np.ndarray) -> np.ndarray:
    """Monotone-decreasing map p → IC50 (µM). Clipped at the anchors."""
    p = np.clip(np.asarray(p_herg, dtype=np.float64), _P_LO, _P_HI)
    frac = (p - _P_LO) / (_P_HI - _P_LO)        # 0 at clean, 1 at potent
    log_ic50 = _LOG_IC50_HI + frac * (_LOG_IC50_LO - _LOG_IC50_HI)
    return np.power(10.0, log_ic50)


def hill_block(ic50_uM: float | np.ndarray, conc_uM: float = 0.5,
               hill: float = 1.0) -> np.ndarray:
    """Fractional block at a given free concentration. Hill equation."""
    ic50 = np.maximum(np.asarray(ic50_uM, dtype=np.float64), 1e-6)
    return 1.0 / (1.0 + np.power(ic50 / conc_uM, hill))


def prob_to_block(p_herg, conc_uM: float = 0.5, hill: float = 1.0) -> np.ndarray:
    """Composition: chem probability → IC50 → fractional IKr block."""
    return hill_block(prob_to_ic50_uM(p_herg), conc_uM=conc_uM, hill=hill)
