"""Plot helpers. Matplotlib only — no seaborn dep, keeps the install lean."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def roc_panel(
    curves: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    aurocs: Mapping[str, float],
    out_path: str | Path,
) -> Path:
    """curves: {model_name: (y_true, y_score)}, aurocs: {model_name: auc}."""
    from sklearn.metrics import roc_curve
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, (y_true, y_score) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={aurocs[name]:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("hERG TDC test set — ROC")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    out = Path(out_path)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def reliability_diagram(
    y_true: Sequence[int],
    series: Mapping[str, Sequence[float]],
    out_path: str | Path,
    n_bins: int = 15,
) -> Path:
    """Multi-line reliability diagram. series: {label: y_prob}."""
    plt = _mpl()
    y_true = np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(5, 5))
    bins = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])
    for name, prob in series.items():
        prob = np.asarray(prob)
        idx = np.digitize(prob, bins) - 1
        idx = np.clip(idx, 0, n_bins - 1)
        emp = np.full(n_bins, np.nan)
        for b in range(n_bins):
            sel = idx == b
            if sel.sum() >= 5:
                emp[b] = y_true[sel].mean()
        ax.plot(centers, emp, marker="o", label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title(f"Reliability ({n_bins} bins)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    out = Path(out_path)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def coverage_bars(
    rows: Iterable[tuple[str, float, float]],  # (label, alpha, coverage)
    out_path: str | Path,
) -> Path:
    plt = _mpl()
    rows = list(rows)
    labels = [f"{r[0]}\n α={r[1]:.2f}" for r in rows]
    cov = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(rows)), 4))
    ax.bar(labels, cov)
    ax.axhline(0.9, color="k", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0, 1.05)
    ax.set_title("Conformal coverage — ID vs OOD")
    out = Path(out_path)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out


def ap_traces(
    traces: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    out_path: str | Path,
) -> Path:
    """traces: {label: (t_ms, V_mV)}."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (t, v) in traces.items():
        ax.plot(t, v, label=name, lw=1.2)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("V (mV)")
    ax.set_title("O'Hara–Rudy 2011 endo APs (last paced beat)")
    ax.legend(fontsize=8)
    out = Path(out_path)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    return out
