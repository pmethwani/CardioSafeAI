"""Chemistry featurisers — Morgan, Avalon, RDKit-200d, MapLight stack, GIN graph builder.

Lifted from the original notebook (Cells 5, 13/14, 16, 17) with two changes:
- Avalon is optional (some RDKit conda builds omit it; fall back to ECFP+RDKit only)
- Single source of truth for atom features used by every GIN variant
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# --- RDKit ---
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

try:
    from rdkit.Avalon import pyAvalonTools as _avalon
    _HAS_AVALON = True
except ImportError:
    _avalon = None
    _HAS_AVALON = False


# --- Morgan / ECFP ---
def morgan(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.uint8)


# --- RDKit 200-d descriptors ---
DESC_NAMES: list[str] = [d[0] for d in Descriptors._descList]
_calc = MoleculeDescriptors.MolecularDescriptorCalculator(DESC_NAMES)


def rdkit_descriptors(mol) -> np.ndarray:
    try:
        d = np.array(_calc.CalcDescriptors(mol), dtype=np.float32)
        return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return np.zeros(len(DESC_NAMES), dtype=np.float32)


# --- MapLight feature stack: ECFP4 ⊕ Avalon ⊕ RDKit-200d ---
MAPLIGHT_DIM_WITH_AVALON = 2048 + 512 + len(DESC_NAMES)
MAPLIGHT_DIM_NO_AVALON = 2048 + len(DESC_NAMES)


def maplight_features(smiles: str) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    ecfp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), dtype=np.uint8)
    parts = [ecfp]
    if _HAS_AVALON:
        avalon = np.array(_avalon.GetAvalonFP(mol, nBits=512), dtype=np.uint8)
        parts.append(avalon)
    parts.append(rdkit_descriptors(mol))
    return np.concatenate([p.astype(np.float32) for p in parts])


def featurize_dataframe(smiles_list: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, mask) so callers can drop failing molecules consistently."""
    feats = []
    mask = []
    for s in smiles_list:
        f = maplight_features(s)
        if f is None:
            mask.append(False)
            feats.append(np.zeros(MAPLIGHT_DIM_WITH_AVALON if _HAS_AVALON else MAPLIGHT_DIM_NO_AVALON, dtype=np.float32))
        else:
            mask.append(True)
            feats.append(f)
    return np.stack(feats), np.array(mask)


# --- GIN graph builder ---
ATOM_FEATURES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se", "H", "*"]
ATOM_FEATURE_DIM = len(ATOM_FEATURES) + 5  # one-hot + degree, charge, Hs, aromatic, in-ring


def atom_features(atom) -> list[int]:
    f = [0] * len(ATOM_FEATURES)
    sym = atom.GetSymbol()
    f[ATOM_FEATURES.index(sym) if sym in ATOM_FEATURES else -1] = 1
    f += [
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        int(atom.GetIsAromatic()),
        int(atom.IsInRing()),
    ]
    return f


def smiles_to_pyg(smiles: str, y: float, desc: np.ndarray | None = None):
    """Build a torch_geometric Data object. Lazy import torch so this module
    can be inspected without the heavy deps installed."""
    import torch
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    edges: list[list[int]] = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges += [[i, j], [j, i]]
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.zeros((2, 0), dtype=torch.long)
    )
    kw = {"x": x, "edge_index": edge_index, "y": torch.tensor([y], dtype=torch.float)}
    if desc is not None:
        kw["desc"] = torch.tensor(desc, dtype=torch.float).unsqueeze(0)
    return Data(**kw)
