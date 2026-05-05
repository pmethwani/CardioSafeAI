"""GIN models. GINEmb exposes a separate `embed()` so the fusion model can reuse the
graph representation without a second forward pass."""
from __future__ import annotations


def _torch_modules():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GINConv, global_add_pool
    return torch, nn, F, GINConv, global_add_pool


def build_gin_emb(in_dim: int = 19, hid: int = 128):
    torch, nn, F, GINConv, global_add_pool = _torch_modules()

    class GINEmb(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = GINConv(nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, hid)))
            self.c2 = GINConv(nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, hid)))
            self.c3 = GINConv(nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, hid)))
            self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hid, 1))

        def embed(self, d):
            h = F.relu(self.c1(d.x, d.edge_index))
            h = F.relu(self.c2(h, d.edge_index))
            h = F.relu(self.c3(h, d.edge_index))
            return global_add_pool(h, d.batch)

        def forward(self, d):
            return self.head(self.embed(d)).squeeze(-1)

    return GINEmb()


def build_gin_desc(in_dim: int = 19, hid: int = 128, desc_dim: int = 200):
    torch, nn, F, GINConv, global_add_pool = _torch_modules()

    class GINDesc(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = GINConv(nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, hid)))
            self.c2 = GINConv(nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, hid)))
            self.c3 = GINConv(nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, hid)))
            self.desc_mlp = nn.Sequential(
                nn.Linear(desc_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64)
            )
            self.head = nn.Sequential(
                nn.Linear(hid + 64, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1)
            )

        def forward(self, d):
            h = F.relu(self.c1(d.x, d.edge_index))
            h = F.relu(self.c2(h, d.edge_index))
            h = F.relu(self.c3(h, d.edge_index))
            h_graph = global_add_pool(h, d.batch)
            h_desc = self.desc_mlp(d.desc)
            return self.head(torch.cat([h_graph, h_desc], dim=1)).squeeze(-1)

    return GINDesc()


def train_gin(model, train_loader, *, epochs: int = 50, lr: float = 1e-3,
              weight_decay: float = 0.0, device: str = "cpu"):
    torch, nn, _, _, _ = _torch_modules()
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = crit(model(batch), batch.y)
            loss.backward()
            opt.step()
    return model


def predict_proba(model, loader, device: str = "cpu"):
    torch, *_ = _torch_modules()
    import numpy as np
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds.append(torch.sigmoid(model(batch)).cpu().numpy())
            ys.append(batch.y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(ys)


def embed_dataset(model, loader, device: str = "cpu"):
    """Run model.embed() over a dataloader; returns (N, hid) embedding matrix and labels."""
    torch, *_ = _torch_modules()
    import numpy as np
    model.eval()
    embs, ys = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            embs.append(model.embed(batch).cpu().numpy())
            ys.append(batch.y.cpu().numpy())
    return np.concatenate(embs), np.concatenate(ys)
