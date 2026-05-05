"""Late-fusion classifier: chem probability ⊕ physio biomarkers → MLP head.

Input vector layout (1 + BIOMARKER_DIM = 9 by default):
    [chem_prob, APD90, APD50, triangulation, dVdt_max, RMP, V_peak, EAD, qNet]

Two ways to use it:
  - sklearn-style wrapper `FusionClassifier` (predict_proba), so MAPIE can
    consume it for split-conformal calibration with method='lac' / cv='prefit'.
  - direct `train_fusion()` for the notebook flow.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin


def build_inputs(chem_prob: np.ndarray, biomarkers: np.ndarray) -> np.ndarray:
    """chem_prob: (N,), biomarkers: (N, K). Returns (N, 1+K)."""
    chem_prob = np.asarray(chem_prob, dtype=np.float32).reshape(-1, 1)
    biomarkers = np.asarray(biomarkers, dtype=np.float32)
    if chem_prob.shape[0] != biomarkers.shape[0]:
        raise ValueError(f"length mismatch: {chem_prob.shape} vs {biomarkers.shape}")
    return np.concatenate([chem_prob, biomarkers], axis=1)


class FusionMLP:
    """Tiny torch MLP with an sklearn-shaped `fit / predict_proba` API.

    Kept torch-only (no skorch) to minimise deps. Two hidden layers, dropout 0.3,
    BCE loss with class-imbalance reweighting, AdamW, early stop on val AUROC.
    """

    def __init__(self, in_dim: int, hidden: int = 64, *,
                 epochs: int = 200, lr: float = 1e-3, weight_decay: float = 1e-4,
                 device: str = "cpu", seed: int = 1):
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.device = device
        self.seed = int(seed)
        self._net = None
        self.classes_ = np.array([0, 1])

    def _build(self):
        import torch
        from torch import nn
        torch.manual_seed(self.seed)
        return nn.Sequential(
            nn.Linear(self.in_dim, self.hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(self.hidden, 1),
        )

    def fit(self, X: np.ndarray, y: np.ndarray, *, X_val=None, y_val=None):
        import torch
        from torch import nn
        net = self._build().to(self.device)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=self.device)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        yt = torch.tensor(y, dtype=torch.float32, device=self.device)

        best_state, best_auc = None, -1.0
        for ep in range(self.epochs):
            net.train()
            opt.zero_grad()
            logits = net(Xt).squeeze(-1)
            loss = crit(logits, yt)
            loss.backward(); opt.step()
            if X_val is not None and y_val is not None and (ep + 1) % 10 == 0:
                net.eval()
                with torch.no_grad():
                    Xv = torch.tensor(X_val, dtype=torch.float32, device=self.device)
                    p = torch.sigmoid(net(Xv).squeeze(-1)).cpu().numpy()
                from sklearn.metrics import roc_auc_score
                try:
                    auc = roc_auc_score(y_val, p)
                except ValueError:
                    auc = 0.0
                if auc > best_auc:
                    best_auc = auc
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if best_state is not None:
            net.load_state_dict(best_state)
        self._net = net
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("fit() before predict_proba()")
        import torch
        self._net.eval()
        with torch.no_grad():
            p = torch.sigmoid(self._net(torch.tensor(X, dtype=torch.float32, device=self.device)).squeeze(-1)).cpu().numpy()
        return np.stack([1 - p, p], axis=1)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class FusionClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-shaped pipeline: standardise → FusionMLP. Designed for MAPIE."""

    def __init__(self, in_dim: int, hidden: int = 64, epochs: int = 200,
                 lr: float = 1e-3, weight_decay: float = 1e-4, device: str = "cpu",
                 seed: int = 1):
        self.in_dim = in_dim
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.seed = seed
        self.scaler_ = None
        self.mlp_ = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, X_val=None, y_val=None):
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.mlp_ = FusionMLP(
            in_dim=self.in_dim, hidden=self.hidden, epochs=self.epochs,
            lr=self.lr, weight_decay=self.weight_decay, device=self.device, seed=self.seed,
        )
        Xv_s = self.scaler_.transform(X_val) if X_val is not None else None
        self.mlp_.fit(Xs, np.asarray(y), X_val=Xv_s, y_val=y_val)
        return self

    def predict_proba(self, X):
        Xs = self.scaler_.transform(X)
        return self.mlp_.predict_proba(Xs)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
