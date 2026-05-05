"""Split-conformal calibration for binary classifiers.

We use MAPIE's `MapieClassifier(method='lac', cv='prefit')` against a fitted
`FusionClassifier`. The interface here is:

    cal = SplitConformal(method='lac').fit(estimator, X_cal, y_cal)
    sets, p, score = cal.predict_sets(X_test, alpha=[0.05, 0.10, 0.20])
    coverage = cal.coverage(y_test, sets, alpha=...)
    ece = expected_calibration_error(y_test, p, n_bins=15)

We also ship a from-scratch LAC fallback for environments where MAPIE isn't
available — useful for the local-friendly mode.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    n = y_prob.size
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        conf = y_prob[sel].mean()
        acc = (y_true[sel] == 1).mean()
        ece += (sel.sum() / n) * abs(conf - acc)
    return float(ece)


class SplitConformal:
    """Thin wrapper that uses MAPIE if importable, otherwise a self-contained LAC."""

    def __init__(self, method: str = "lac"):
        self.method = method
        self._mapie = None
        self._scores_per_class: dict[int, np.ndarray] | None = None
        self._estimator = None

    def fit(self, estimator, X_cal, y_cal):
        self._estimator = estimator
        try:
            from mapie.classification import MapieClassifier
            mp = MapieClassifier(estimator=estimator, method=self.method, cv="prefit")
            mp.fit(X_cal, y_cal)
            self._mapie = mp
            return self
        except Exception:
            # fall back to LAC: nonconformity = 1 - p_y
            self._mapie = None
            proba = estimator.predict_proba(X_cal)
            y = np.asarray(y_cal).astype(int)
            scores = 1.0 - proba[np.arange(len(y)), y]
            self._scores_per_class = {
                0: scores[y == 0],
                1: scores[y == 1],
            }
            return self

    def predict_sets(self, X, alpha: Sequence[float] = (0.05, 0.1, 0.2)):
        alpha = list(alpha)
        if self._mapie is not None:
            p, sets = self._mapie.predict(X, alpha=alpha)
            # MAPIE returns shape (N, 2, len(alpha)); collapse to bool sets
            return sets.astype(bool), self._mapie.estimator_.predict_proba(X), alpha
        if self._estimator is None:
            raise RuntimeError("fit() first")
        proba = self._estimator.predict_proba(X)
        n = proba.shape[0]
        sets = np.zeros((n, 2, len(alpha)), dtype=bool)
        # Mondrian: per-class quantiles
        for ai, a in enumerate(alpha):
            for c in (0, 1):
                cs = self._scores_per_class.get(c, np.array([1.0]))
                # split-conformal quantile with finite-sample correction
                k = int(np.ceil((cs.size + 1) * (1 - a)))
                k = min(max(k, 1), cs.size)
                qhat = np.sort(cs)[k - 1]
                included = (1.0 - proba[:, c]) <= qhat
                sets[:, c, ai] = included
        return sets, proba, alpha

    @staticmethod
    def coverage(y_true, sets, alpha) -> dict[float, float]:
        y_true = np.asarray(y_true).astype(int)
        out = {}
        for ai, a in enumerate(alpha):
            included = sets[np.arange(len(y_true)), y_true, ai]
            out[float(a)] = float(included.mean())
        return out

    @staticmethod
    def mean_set_size(sets, alpha) -> dict[float, float]:
        out = {}
        for ai, a in enumerate(alpha):
            out[float(a)] = float(sets[:, :, ai].sum(axis=1).mean())
        return out
