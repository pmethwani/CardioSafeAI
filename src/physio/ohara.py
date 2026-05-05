"""O'Hara–Rudy 2011 simulator + biomarker extraction.

Wrapper around myokit. One simulator is reused across compounds — only the IKr
conductance scaling and reset state change between drugs.

Biomarkers per drug (8-d vector, in this order):
    0  APD90        ms
    1  APD50        ms
    2  triangulation = APD90 - APD50    ms
    3  dV/dt_max    mV/ms
    4  RMP          mV   (resting membrane potential, last sample of beat)
    5  V_peak       mV
    6  EAD_flag     {0,1}   (any local minimum during repolarisation)
    7  qNet         C/F     (∫ of net repolarising currents over the last beat)

A disk cache keyed by (smiles, conc_uM, n_beats) skips repeat simulations.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


_DEFAULT_MMT = Path(__file__).resolve().parent / "ohara-2011.mmt"

BIOMARKER_NAMES = [
    "APD90", "APD50", "triangulation", "dVdt_max",
    "RMP", "V_peak", "EAD_flag", "qNet",
]
BIOMARKER_DIM = len(BIOMARKER_NAMES)


@dataclass
class SimResult:
    t: np.ndarray
    V: np.ndarray
    biomarkers: np.ndarray  # shape (BIOMARKER_DIM,)


# ---------- biomarker extraction (pure numpy) ----------

def _apd_at(t: np.ndarray, V: np.ndarray, frac: float) -> float:
    """AP duration at <frac> of repolarisation. APD<int(100*frac)>."""
    Vmax_i = int(np.argmax(V))
    Vmax, Vmin = V.max(), V.min()
    target = Vmax - frac * (Vmax - Vmin)
    repol = V[Vmax_i:]
    repol_t = t[Vmax_i:]
    below = np.where(repol <= target)[0]
    if below.size == 0:
        return float("nan")
    end_i = below[0]
    # take stimulus start as the first sample (we record from beat start)
    return float(repol_t[end_i] - t[0])


def _ead_flag(V: np.ndarray) -> int:
    """1 if any local minimum exists in the repolarisation phase before RMP recovery."""
    if V.size < 5:
        return 0
    Vmax_i = int(np.argmax(V))
    repol = V[Vmax_i:]
    if repol.size < 5:
        return 0
    # local minimum: V[i-1] > V[i] < V[i+1] AND V[i] still depolarised relative to RMP
    rmp = float(V[-1])
    for i in range(2, repol.size - 2):
        if repol[i] < repol[i - 1] and repol[i] < repol[i + 1] and (repol[i] - rmp) > 5.0:
            return 1
    return 0


def _qnet(t: np.ndarray, currents: dict[str, np.ndarray]) -> float:
    """qNet = ∫ (IKr + IKs + IK1 + Ito + INaL + ICaL) dt over the recorded window.
    Currents are in A/F, time in ms → qNet in C/F (numerically: A/F * ms / 1000 = C/F)."""
    keys = ("IKr", "IKs", "IK1", "Ito", "INaL", "ICaL")
    total = np.zeros_like(t, dtype=np.float64)
    for k in keys:
        if k in currents:
            total = total + currents[k]
    # trapezoidal integration; ms → s
    return float(np.trapz(total, t) / 1000.0)


def extract_biomarkers(t: np.ndarray, V: np.ndarray,
                       currents: Optional[dict[str, np.ndarray]] = None) -> np.ndarray:
    apd90 = _apd_at(t, V, 0.90)
    apd50 = _apd_at(t, V, 0.50)
    tri = apd90 - apd50 if not (np.isnan(apd90) or np.isnan(apd50)) else float("nan")
    dt = np.diff(t)
    dV = np.diff(V)
    dVdt_max = float(np.max(dV / np.maximum(dt, 1e-9))) if dt.size else float("nan")
    rmp = float(V[-1])
    vpk = float(V.max())
    ead = _ead_flag(V)
    qn = _qnet(t, currents) if currents else 0.0
    return np.array([apd90, apd50, tri, dVdt_max, rmp, vpk, ead, qn], dtype=np.float32)


# ---------- simulator ----------

class OharaSimulator:
    """Reusable myokit simulator. Pace at cycle_length ms for n_beats. Record
    last-beat V and currents. Modify IKr block per call by scaling ikr.gKr."""

    def __init__(self, mmt_path: str | Path = _DEFAULT_MMT, *, cell_mode: int = 0,
                 cycle_length_ms: float = 1000.0):
        import myokit
        self._myokit = myokit
        self.mmt_path = Path(mmt_path)
        if not self.mmt_path.exists():
            raise FileNotFoundError(f"{self.mmt_path} not found — check the vendored .mmt path")
        self.model, _proto, _ = myokit.load(str(self.mmt_path))
        # Endo / Epi / Mid
        try:
            self.model.set_value("cell.mode", cell_mode)
        except Exception:
            pass
        self.cycle = float(cycle_length_ms)
        self._gKr_default = float(self.model.get("ikr.gKr").rhs().eval())
        self._proto = myokit.pacing.blocktrain(self.cycle, duration=0.5, offset=20)
        self._sim = myokit.Simulation(self.model, self._proto)
        self._reset_state = list(self._sim.state())

    def reset(self) -> None:
        self._sim.reset()
        self._sim.set_state(self._reset_state)
        self._sim.set_constant("ikr.gKr", self._gKr_default)

    def simulate(self, *, ikr_block: float = 0.0, n_beats: int = 1000,
                 record_last: int = 1, log_currents: bool = True) -> SimResult:
        """Run n_beats then record `record_last` more beats. Returns last-beat
        traces + biomarkers."""
        self.reset()
        scale = max(0.0, 1.0 - float(ikr_block))
        self._sim.set_constant("ikr.gKr", self._gKr_default * scale)

        if n_beats > record_last:
            self._sim.run(self.cycle * (n_beats - record_last), log=[])

        log_vars = ["engine.time", "membrane.V"]
        if log_currents:
            log_vars += ["ikr.IKr", "iks.IKs", "ik1.IK1", "ito.Ito",
                         "inal.INaL", "ical.ICaL"]
        log = self._sim.run(self.cycle * record_last, log=log_vars)

        t = np.asarray(log["engine.time"], dtype=np.float64)
        V = np.asarray(log["membrane.V"], dtype=np.float64)
        # Anchor t to the start of the recorded window so APD extraction is offset-free.
        t = t - t[0]
        currents = None
        if log_currents:
            currents = {
                "IKr":  np.asarray(log["ikr.IKr"]),
                "IKs":  np.asarray(log["iks.IKs"]),
                "IK1":  np.asarray(log["ik1.IK1"]),
                "Ito":  np.asarray(log["ito.Ito"]),
                "INaL": np.asarray(log["inal.INaL"]),
                "ICaL": np.asarray(log["ical.ICaL"]),
            }
        bm = extract_biomarkers(t, V, currents)
        return SimResult(t=t, V=V, biomarkers=bm)


# ---------- batch + cache ----------

def _cache_key(smiles: str, conc_uM: float, n_beats: int, cell_mode: int) -> str:
    h = hashlib.sha1(f"{smiles}|{conc_uM:.6g}|{n_beats}|{cell_mode}".encode()).hexdigest()
    return h[:16]


def biomarkers_for(
    smiles_list: Iterable[str],
    p_herg: Iterable[float],
    *,
    conc_uM: float = 0.5,
    hill: float = 1.0,
    n_beats: int = 600,
    cell_mode: int = 0,
    cache_path: str | Path | None = None,
    progress: bool = False,
) -> np.ndarray:
    """Run the ORd simulator for a list of (smiles, p_herg) pairs.
    Returns array of shape (N, BIOMARKER_DIM)."""
    from src.physio.herg_block import prob_to_block

    smiles_list = list(smiles_list)
    p_herg = list(p_herg)
    out = np.full((len(smiles_list), BIOMARKER_DIM), np.nan, dtype=np.float32)

    cache: dict[str, list[float]] = {}
    cache_file = Path(cache_path) if cache_path else None
    if cache_file and cache_file.exists():
        with open(cache_file) as fp:
            cache = json.load(fp)

    sim: Optional[OharaSimulator] = None

    for i, (s, p) in enumerate(zip(smiles_list, p_herg)):
        key = _cache_key(s, conc_uM, n_beats, cell_mode)
        if key in cache:
            out[i] = np.asarray(cache[key], dtype=np.float32)
            continue
        if sim is None:
            sim = OharaSimulator(cell_mode=cell_mode)
        block = float(prob_to_block(p, conc_uM=conc_uM, hill=hill))
        try:
            r = sim.simulate(ikr_block=block, n_beats=n_beats, record_last=1)
            out[i] = r.biomarkers
            cache[key] = out[i].tolist()
        except Exception as e:
            if progress:
                print(f"  [{i}] sim failed for {s[:40]}: {e}")
            out[i] = 0.0
        if progress and (i + 1) % 25 == 0:
            print(f"  simulated {i+1}/{len(smiles_list)}")

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as fp:
            json.dump(cache, fp)

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out
