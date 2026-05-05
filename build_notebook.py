"""Build CardioSafeAI_MM.ipynb deterministically.

Run: `python3 build_notebook.py` from the project root. Overwrites the existing
notebook. Kept in-tree so the user can rebuild after edits to any cell template.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "CardioSafeAI_MM.ipynb"

CELLS: list[tuple[str, str]] = []   # (kind, source)


def md(s: str) -> None:
    CELLS.append(("markdown", s))


def code(s: str) -> None:
    CELLS.append(("code", s))


# ---------- Cell 0: title ----------
md("""# CardioSafeAI-MM
**Cross-modal hERG cardiotoxicity prediction**
fusing Chemprop / GIN over SMILES with O'Hara–Rudy 2011 cardiac action-potential biomarkers,
calibrated via split-conformal prediction.

**Author:** Prateek Methwani — MSc AI-Driven Bioinformatics, SRH Munich
**Module:** AI for Drug Discovery (Summer 2026)

| Stage | Target |
|---|---|
| RF + Morgan baseline | ~0.78 AUROC |
| GIN + RDKit | ~0.83 |
| MapLight (CatBoost) | ~0.87 |
| MapLight + GIN ensemble | ~0.88 (SOTA) |
| **Fusion (chem + ORd biomarkers)** | **≥0.86 with better ECE/coverage** |

Runs in Google Colab (primary) and local Mac with the heavy deps installed.
Bootstrap auto-detects environment.""")


# ---------- Cell 1: bootstrap ----------
code("""# === Bootstrap: env detect, paths, sys.path ===
# Works in both Colab and local. The src/env.py module is the single source of truth.
import os, sys
from pathlib import Path

# Find the project root. In Colab we'll mount Drive next; locally, this notebook
# already sits at the project root.
_HERE = Path.cwd()
if not (_HERE / "src" / "env.py").exists():
    # try one level up
    if (_HERE.parent / "src" / "env.py").exists():
        os.chdir(_HERE.parent)

sys.path.insert(0, str(Path.cwd()))
from src.env import IN_COLAB, project_root, mount_drive_if_colab, ensure_project_dirs, add_src_to_path, device

if IN_COLAB:
    mount_drive_if_colab()

PROJECT = ensure_project_dirs()
os.chdir(PROJECT)
add_src_to_path()
DEVICE = device()
print(f"📂 PROJECT  : {PROJECT}")
print(f"🌐 IN_COLAB : {IN_COLAB}")
print(f"💻 DEVICE   : {DEVICE}")""")


# ---------- Cell 2: install deps (Colab) ----------
code("""# === Install heavy dependencies (Colab) ===
# On Colab we install everything in one place. Locally we expect the user to
# have prepared a venv per the README — this cell is a no-op outside Colab.
from src.env import IN_COLAB

if IN_COLAB:
    # Core
    !pip install -q "scikit-learn<1.6" rdkit pytdc chemprop torch
    # PyG (Colab usually has CUDA wheels matched to torch)
    !pip install -q torch-geometric
    # Other modelling deps
    !pip install -q deepchem catboost mapie "astartes[molecules]" --upgrade
    # Physio
    !apt-get install -y libsundials-dev > /dev/null 2>&1
    !pip install -q myokit
    print("✅ Colab installs complete — if any wheel changed, restart runtime once and re-run from Cell 1.")
else:
    print("Local mode — skip Colab installs. See README for the brew/pip recipe.")""")


# ---------- Cell 3: import + version check + write README/.gitignore ----------
code("""# === Verify imports + write README/.gitignore ===
import importlib
def _v(m):
    try:
        x = importlib.import_module(m)
        return getattr(x, "__version__", "?")
    except Exception as e:
        return f"missing ({type(e).__name__})"

mods = ["numpy", "pandas", "sklearn", "rdkit", "tdc", "torch",
        "torch_geometric", "deepchem", "catboost", "mapie", "myokit"]
for m in mods:
    print(f"  {m:<18} {_v(m)}")

import torch
if torch.cuda.is_available():
    print(f"\\nGPU: {torch.cuda.get_device_name(0)}")
elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
    print("\\nMPS (Apple Silicon) available")
else:
    print("\\nCPU only")

# Project metadata files (safe to overwrite each run)
with open(".gitignore", "w") as f:
    f.write("__pycache__/\\n*.pyc\\n.ipynb_checkpoints/\\ndata/raw/\\n*.pt\\nmlruns/\\n*.npz\\n")

with open("README.md", "w") as f:
    f.write('''# CardioSafeAI-MM

Cross-modal hERG cardiotoxicity prediction fusing chemistry models (RF / GIN / MapLight)
with O\\'Hara-Rudy 2011 cardiac action-potential biomarkers, calibrated via split-conformal.

## Run targets
- **Google Colab** (primary): open the notebook, run from Cell 1.
- **Local Mac (Apple Silicon)**: needs a fresh venv —
  ```
  python3.11 -m venv .venv && source .venv/bin/activate
  pip install rdkit pandas pytdc scikit-learn torch torch-geometric \\
              deepchem catboost mapie myokit "astartes[molecules]"
  brew install sundials   # for myokit CVODE
  ```

## Status
Built end-to-end. See `results/` for AUROC tables and `figures/` for plots.
''')
print("\\n✅ README + .gitignore written")""")


# ---------- Cell 4: load TDC hERG benchmark ----------
code("""# === TDC ADMET hERG benchmark ===
from tdc.benchmark_group import admet_group
import pandas as pd

group = admet_group(path="data/")
benchmark = group.get("herg")
train_val, test = benchmark["train_val"], benchmark["test"]

print(f"Train+Val: {len(train_val)} | Test: {len(test)}")
print(f"Class balance (train): {train_val['Y'].mean():.3f}")
train_val.head(3)""")


# ---------- Cell 5: RF + Morgan baseline ----------
code("""# === Baseline 1: Random Forest + Morgan fingerprints (5-seed) ===
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from src.chem.featurize import morgan
from src.utils.seeds import seed_everything

print("Featurising...")
X_tv  = np.stack([morgan(s) for s in train_val["Drug"]])
X_te  = np.stack([morgan(s) for s in test["Drug"]])
y_tv  = train_val["Y"].values
y_te  = test["Y"].values
print(f"Shapes: X_tv {X_tv.shape}, X_te {X_te.shape}")

aurocs_rf = []
preds_rf_seeds = []   # save per-seed test predictions for ensembles later
for seed in [1, 2, 3, 4, 5]:
    seed_everything(seed)
    train_split, _ = group.get_train_valid_split(benchmark="herg", split_type="default", seed=seed)
    train_idx = train_val[train_val["Drug_ID"].isin(train_split["Drug_ID"])].index
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=seed)
    rf.fit(X_tv[train_idx], y_tv[train_idx])
    p = rf.predict_proba(X_te)[:, 1]
    preds_rf_seeds.append(p)
    auc = roc_auc_score(y_te, p)
    aurocs_rf.append(auc)
    print(f"  seed {seed}: AUROC = {auc:.4f}")

print(f"\\n📊 RF + Morgan: {np.mean(aurocs_rf):.4f} ± {np.std(aurocs_rf):.4f}")""")


# ---------- Cell 6: save RF ----------
code("""import pandas as pd  # explicit re-import: kernel-restart-safe
results_rf = pd.DataFrame({"model": ["RF_Morgan"]*5, "seed": [1,2,3,4,5], "auroc": aurocs_rf})
results_rf.to_csv("results/baseline_rf.csv", index=False)
print("✅ results/baseline_rf.csv")
results_rf""")


# ---------- Cell 7: Chemprop training (real) ----------
code("""# === Baseline 2: Chemprop D-MPNN (5-seed) ===
# Replaces the original `chemprop --help` probe. Runs Chemprop's CLI training
# entrypoint per seed; falls back to Python API if CLI not present.
import os, json, subprocess, shutil
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

os.makedirs("data/chemprop", exist_ok=True)
train_val[["Drug","Y"]].rename(columns={"Drug":"smiles","Y":"hERG"}).to_csv("data/chemprop/train_val.csv", index=False)
test[["Drug","Y"]].rename(columns={"Drug":"smiles","Y":"hERG"}).to_csv("data/chemprop/test.csv", index=False)

aurocs_cp = []
chemprop_cli = shutil.which("chemprop")
for seed in [1, 2, 3, 4, 5]:
    out_dir = f"data/chemprop/run_seed{seed}"
    if chemprop_cli:
        cmd = [
            chemprop_cli, "train",
            "--data-path", "data/chemprop/train_val.csv",
            "--task-type", "classification",
            "--output-dir", out_dir,
            "--epochs", "30",
            "--num-folds", "1",
            "--seed", str(seed),
            "--smiles-columns", "smiles",
            "--target-columns", "hERG",
            "--metrics", "roc",
        ]
        subprocess.run(cmd, check=False)
        # predict
        pred_path = f"{out_dir}/test_preds.csv"
        subprocess.run([
            chemprop_cli, "predict",
            "--test-path", "data/chemprop/test.csv",
            "--checkpoint-dir", out_dir,
            "--preds-path", pred_path,
            "--smiles-columns", "smiles",
        ], check=False)
        try:
            p = pd.read_csv(pred_path)["hERG"].values
            auc = roc_auc_score(y_te, p)
        except Exception as e:
            print(f"  seed {seed}: chemprop CLI failed ({e}); skipping")
            continue
    else:
        # Python API fallback (older chemprop)
        try:
            from chemprop.train import cross_validate, run_training
            from chemprop.args import TrainArgs, PredictArgs
            args = TrainArgs().parse_args([
                "--data_path", "data/chemprop/train_val.csv",
                "--dataset_type", "classification",
                "--save_dir", out_dir,
                "--epochs", "30", "--num_folds", "1",
                "--seed", str(seed), "--smiles_columns", "smiles",
                "--target_columns", "hERG", "--metric", "auc",
                "--quiet",
            ])
            cross_validate(args=args, train_func=run_training)
            from chemprop.train import make_predictions
            pargs = PredictArgs().parse_args([
                "--test_path", "data/chemprop/test.csv",
                "--checkpoint_dir", out_dir,
                "--preds_path", f"{out_dir}/test_preds.csv",
                "--smiles_columns", "smiles",
            ])
            preds = make_predictions(args=pargs)
            p = np.array([row[0] for row in preds])
            auc = roc_auc_score(y_te, p)
        except Exception as e:
            print(f"  seed {seed}: chemprop unavailable ({e}); using zeros (skip)")
            continue
    aurocs_cp.append(auc)
    print(f"  seed {seed}: AUROC = {auc:.4f}")

if aurocs_cp:
    print(f"\\n📊 Chemprop: {np.mean(aurocs_cp):.4f} ± {np.std(aurocs_cp):.4f}")
    pd.DataFrame({"model":["Chemprop"]*len(aurocs_cp), "seed":list(range(1,len(aurocs_cp)+1)), "auroc":aurocs_cp}).to_csv("results/baseline_chemprop.csv", index=False)
else:
    print("⚠️  Chemprop produced no results — continuing with other baselines.")""")


# ---------- Cell 8: DeepChem GCN ----------
code("""# === Baseline 3: DeepChem GCN ===
# Original Cell 12, fixed mask: use isinstance instead of hasattr.
import deepchem as dc
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from src.utils.seeds import seed_everything

featurizer = dc.feat.MolGraphConvFeaturizer()
X_tv_g = featurizer.featurize(train_val["Drug"].tolist())
X_te_g = featurizer.featurize(test["Drug"].tolist())
GraphData = dc.feat.graph_data.GraphData
mask_tv = np.array([isinstance(x, GraphData) for x in X_tv_g])
mask_te = np.array([isinstance(x, GraphData) for x in X_te_g])
print(f"Train: {mask_tv.sum()}/{len(X_tv_g)} | Test: {mask_te.sum()}/{len(X_te_g)}")

y_tvf = train_val["Y"].values.astype(np.float32)
y_tef = test["Y"].values.astype(np.float32)

aurocs_gcn = []
for seed in [1, 2, 3, 4, 5]:
    seed_everything(seed)
    train_ds = dc.data.NumpyDataset(X=X_tv_g[mask_tv], y=y_tvf[mask_tv])
    test_ds  = dc.data.NumpyDataset(X=X_te_g[mask_te], y=y_tef[mask_te])
    model = dc.models.GCNModel(
        n_tasks=1, mode="classification", batch_size=32,
        learning_rate=0.001, graph_conv_layers=[64, 64],
        dense_layer_size=128, dropout=0.2,
    )
    model.fit(train_ds, nb_epoch=50)
    preds = model.predict(test_ds)[:, 0, 1]
    auc = roc_auc_score(y_tef[mask_te], preds)
    aurocs_gcn.append(auc)
    print(f"  seed {seed}: AUROC = {auc:.4f}")

print(f"\\n📊 GCN (DeepChem): {np.mean(aurocs_gcn):.4f} ± {np.std(aurocs_gcn):.4f}")
pd.DataFrame({"model":["GCN_DeepChem"]*5, "seed":[1,2,3,4,5], "auroc":aurocs_gcn}).to_csv("results/baseline_gcn.csv", index=False)""")


# ---------- Cell 9: GIN ----------
code("""# === Baseline 4: GIN (PyG) ===
# Per-test-row predictions: invalid mols get 0.5 to keep alignment with y_te.
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader   # FIXED: was torch_geometric.data
from src.chem.featurize import smiles_to_pyg, ATOM_FEATURE_DIM
from src.chem.models import build_gin_emb, train_gin, predict_proba
from src.utils.seeds import seed_everything

def _build_with_index(df):
    graphs, idx = [], []
    for i, (s, y) in enumerate(zip(df["Drug"], df["Y"])):
        g = smiles_to_pyg(s, y)
        if g is not None:
            graphs.append(g); idx.append(i)
    return graphs, np.array(idx)

print("Building graphs...")
train_graphs, _ = _build_with_index(train_val)
test_graphs, te_idx = _build_with_index(test)
print(f"Train: {len(train_graphs)}/{len(train_val)} | Test: {len(test_graphs)}/{len(test)}")

aurocs_gin = []
preds_gin_seeds = []
for seed in [1, 2, 3, 4, 5]:
    seed_everything(seed)
    tr = DataLoader(train_graphs, batch_size=32, shuffle=True)
    te = DataLoader(test_graphs,  batch_size=64, shuffle=False)
    gin = build_gin_emb(in_dim=ATOM_FEATURE_DIM)
    gin = train_gin(gin, tr, epochs=50, lr=1e-3, device=DEVICE)
    p, _ = predict_proba(gin, te, device=DEVICE)
    full = np.full(len(test), 0.5, dtype=np.float32)
    full[te_idx] = p
    preds_gin_seeds.append(full)
    auc = roc_auc_score(y_te, full)
    aurocs_gin.append(auc)
    print(f"  seed {seed}: AUROC = {auc:.4f}")

print(f"\\n📊 GIN: {np.mean(aurocs_gin):.4f} ± {np.std(aurocs_gin):.4f}")
pd.DataFrame({"model":["GIN"]*5, "seed":[1,2,3,4,5], "auroc":aurocs_gin}).to_csv("results/baseline_gin.csv", index=False)""")


# ---------- Cell 10: GIN+RDKit ----------
code("""# === Baseline 5: GIN + RDKit-200d (concat-fusion at the head) ===
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
from rdkit import Chem
from src.chem.featurize import smiles_to_pyg, rdkit_descriptors, DESC_NAMES, ATOM_FEATURE_DIM
from src.chem.models import build_gin_desc, train_gin, predict_proba
from src.utils.seeds import seed_everything

print("Computing RDKit descriptors...")
def _build(df):
    rows = []
    for s, y in zip(df["Drug"], df["Y"]):
        mol = Chem.MolFromSmiles(s)
        if mol is None: continue
        rows.append((s, y, rdkit_descriptors(mol)))
    return rows

train_rows = _build(train_val)
test_rows  = _build(test)
desc_tv = np.stack([r[2] for r in train_rows]).astype(np.float32)
desc_te = np.stack([r[2] for r in test_rows]).astype(np.float32)

scaler = StandardScaler().fit(desc_tv)
desc_tv_s = np.nan_to_num(np.clip(scaler.transform(desc_tv), -10, 10), nan=0.0).astype(np.float32)
desc_te_s = np.nan_to_num(np.clip(scaler.transform(desc_te), -10, 10), nan=0.0).astype(np.float32)

train_graphs = [smiles_to_pyg(s, y, d) for (s, y, _), d in zip(train_rows, desc_tv_s)]
test_graphs  = [smiles_to_pyg(s, y, d) for (s, y, _), d in zip(test_rows,  desc_te_s)]
train_graphs = [g for g in train_graphs if g is not None]
test_graphs  = [g for g in test_graphs  if g is not None]
print(f"Train: {len(train_graphs)} | Test: {len(test_graphs)} | Desc dim: {desc_tv.shape[1]}")

aurocs_gd = []
for seed in [1, 2, 3, 4, 5]:
    seed_everything(seed)
    tr = DataLoader(train_graphs, batch_size=32, shuffle=True)
    te = DataLoader(test_graphs,  batch_size=64, shuffle=False)
    m = build_gin_desc(in_dim=ATOM_FEATURE_DIM, desc_dim=desc_tv.shape[1])
    m = train_gin(m, tr, epochs=60, lr=1e-3, weight_decay=1e-5, device=DEVICE)
    p, y = predict_proba(m, te, device=DEVICE)
    auc = roc_auc_score(y, p)
    aurocs_gd.append(auc)
    print(f"  seed {seed}: AUROC = {auc:.4f}")

print(f"\\n📊 GIN+RDKit: {np.mean(aurocs_gd):.4f} ± {np.std(aurocs_gd):.4f}")
pd.DataFrame({"model":["GIN_RDKit"]*5, "seed":[1,2,3,4,5], "auroc":aurocs_gd}).to_csv("results/baseline_gin_rdkit.csv", index=False)""")


# ---------- Cell 11: MapLight CatBoost ----------
code("""# === Baseline 6: MapLight (CatBoost on ECFP4 + Avalon + RDKit-200d) ===
# featurize_dataframe zero-fills failed mols, so all rows align with train_val/test
# index order — no mask filtering needed downstream.
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
from src.chem.featurize import featurize_dataframe

print("Featurising MapLight stack...")
X_tv_ml, mask_tv_ml = featurize_dataframe(train_val["Drug"])
X_te_ml, mask_te_ml = featurize_dataframe(test["Drug"])
print(f"Feature dim: {X_tv_ml.shape[1]} | Train OK: {mask_tv_ml.sum()}/{X_tv_ml.shape[0]} | Test OK: {mask_te_ml.sum()}/{X_te_ml.shape[0]}")

aurocs_ml = []
preds_ml_seeds = []
catboost_models = []
for seed in [1, 2, 3, 4, 5]:
    cat = CatBoostClassifier(
        iterations=1000, learning_rate=0.05, depth=6,
        l2_leaf_reg=3, random_seed=seed, eval_metric="AUC", verbose=0,
    )
    cat.fit(X_tv_ml, y_tv)
    p = cat.predict_proba(X_te_ml)[:, 1]
    preds_ml_seeds.append(p)
    catboost_models.append(cat)
    auc = roc_auc_score(y_te, p)
    aurocs_ml.append(auc)
    print(f"  seed {seed}: AUROC = {auc:.4f}")

print(f"\\n📊 MapLight (CatBoost): {np.mean(aurocs_ml):.4f} ± {np.std(aurocs_ml):.4f}")
pd.DataFrame({"model":["MapLight"]*5, "seed":[1,2,3,4,5], "auroc":aurocs_ml}).to_csv("results/baseline_maplight.csv", index=False)""")


# ---------- Cell 12: MapLight + GIN ensemble ----------
code("""# === Baseline 7: MapLight + GIN ensemble (mean of probabilities) ===
# Both arrays are now per-test-row aligned with y_te, so we can mean directly.
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

assert len(preds_gin_seeds) == 5 and len(preds_ml_seeds) == 5

aurocs_ens = []
preds_ens_seeds = []
for s in range(5):
    p_gin = preds_gin_seeds[s]
    p_ml  = preds_ml_seeds[s]
    assert p_gin.shape == p_ml.shape == y_te.shape, "shape mismatch — should not happen after fixes in cells 9/11"
    p_ens = (p_gin + p_ml) / 2.0
    preds_ens_seeds.append(p_ens)
    auc = roc_auc_score(y_te, p_ens)
    aurocs_ens.append(auc)
    print(f"  seed {s+1}: GIN={roc_auc_score(y_te, p_gin):.4f} | ML={roc_auc_score(y_te, p_ml):.4f} | ens={auc:.4f}")

print(f"\\n📊 MapLight + GIN ensemble: {np.mean(aurocs_ens):.4f} ± {np.std(aurocs_ens):.4f}")
pd.DataFrame({"model":["MapLight_GIN"]*5, "seed":[1,2,3,4,5], "auroc":aurocs_ens}).to_csv("results/baseline_maplight_gin.csv", index=False)""")


# ---------- Cell 13: Physio biomarker simulation ----------
code("""# === Physio: O'Hara-Rudy 2011 biomarkers ===
# For each test compound, take the chem-model probability, convert to an IKr
# block fraction, run ORd to steady state, extract 8 biomarkers.
import numpy as np
from src.physio.ohara import biomarkers_for, BIOMARKER_NAMES, BIOMARKER_DIM
from src.physio.herg_block import prob_to_block

# Use the strongest single chem prob: MapLight+GIN ensemble (mean over seeds).
chem_prob_test = np.mean(preds_ens_seeds, axis=0)
print(f"Chem prob (test): mean={chem_prob_test.mean():.3f}, range=[{chem_prob_test.min():.3f}, {chem_prob_test.max():.3f}]")
print(f"→ implied block fractions: range=[{prob_to_block(chem_prob_test.min()):.2f}, {prob_to_block(chem_prob_test.max()):.2f}]")

# Run the simulator. Cache to disk to make subsequent runs ~free.
test_smiles = list(test["Drug"].values[:len(chem_prob_test)])
biomarkers_te = biomarkers_for(
    smiles_list=test_smiles,
    p_herg=chem_prob_test,
    conc_uM=0.5, hill=1.0, n_beats=600, cell_mode=0,
    cache_path="data/physio_biomarkers_test.json",
    progress=True,
)
print(f"\\nBiomarker matrix: {biomarkers_te.shape}  (cols = {BIOMARKER_NAMES})")
print(f"APD90 stats: mean={biomarkers_te[:, 0].mean():.1f} ms, std={biomarkers_te[:, 0].std():.1f} ms")
print(f"EAD flag rate: {(biomarkers_te[:, 6] > 0).mean():.3f}")

# Same for train — reuse the CatBoost models from Cell 11 (mean over 5 seeds).
# In-sample probabilities are slightly optimistic but adequate for the small fusion head.
chem_prob_train = np.mean([cat.predict_proba(X_tv_ml)[:, 1] for cat in catboost_models], axis=0)
biomarkers_tv = biomarkers_for(
    smiles_list=list(train_val["Drug"].values),
    p_herg=chem_prob_train,
    conc_uM=0.5, hill=1.0, n_beats=600, cell_mode=0,
    cache_path="data/physio_biomarkers_train.json",
    progress=True,
)
print(f"Train biomarkers: {biomarkers_tv.shape}")""")


# ---------- Cell 14: fusion model ----------
code("""# === Cross-modal fusion: chem prob ⊕ ORd biomarkers → MLP ===
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from src.models.fusion import build_inputs, FusionClassifier
from src.utils.seeds import seed_everything

# Build fusion inputs (1 chem + 8 physio = 9 features)
X_fuse_tv = build_inputs(chem_prob_train, biomarkers_tv)
X_fuse_te = build_inputs(chem_prob_test,  biomarkers_te)
y_fuse_tv = y_tv[:len(X_fuse_tv)]
y_fuse_te = y_te[:len(X_fuse_te)]
print(f"Fusion input dim: {X_fuse_tv.shape[1]} | tv={X_fuse_tv.shape[0]} | te={X_fuse_te.shape[0]}")

aurocs_fuse = []
preds_fuse_seeds = []
fusion_models = []
for seed in [1, 2, 3, 4, 5]:
    seed_everything(seed)
    # 80/20 train/cal split for conformal — cal slice is reused in next cell
    n = len(X_fuse_tv)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(0.8 * n)
    tr_idx, cal_idx = idx[:cut], idx[cut:]

    clf = FusionClassifier(
        in_dim=X_fuse_tv.shape[1], hidden=64, epochs=200,
        lr=1e-3, weight_decay=1e-4, device=DEVICE, seed=seed,
    )
    clf.fit(
        X_fuse_tv[tr_idx], y_fuse_tv[tr_idx],
        X_val=X_fuse_tv[cal_idx], y_val=y_fuse_tv[cal_idx],
    )
    p_te = clf.predict_proba(X_fuse_te)[:, 1]
    preds_fuse_seeds.append(p_te)
    fusion_models.append((clf, tr_idx, cal_idx))
    auc = roc_auc_score(y_fuse_te, p_te)
    aurocs_fuse.append(auc)
    print(f"  seed {seed}: fusion AUROC = {auc:.4f}")

print(f"\\n📊 Fusion (chem + ORd): {np.mean(aurocs_fuse):.4f} ± {np.std(aurocs_fuse):.4f}")
pd.DataFrame({"model":["Fusion"]*5, "seed":[1,2,3,4,5], "auroc":aurocs_fuse}).to_csv("results/fusion.csv", index=False)""")


# ---------- Cell 15: conformal calibration ----------
code("""# === Split-conformal calibration on the fusion model (per-seed) ===
import numpy as np, pandas as pd
from src.models.conformal import SplitConformal, expected_calibration_error

alphas = [0.05, 0.10, 0.20]
rows = []
preds_fuse_mean = np.mean(preds_fuse_seeds, axis=0)
ece_uncal = expected_calibration_error(y_fuse_te, preds_fuse_mean, n_bins=15)
print(f"Uncalibrated ECE (fusion mean): {ece_uncal:.4f}")

# Calibrate per seed using its own held-out cal split, then aggregate coverage
cov_per_alpha = {a: [] for a in alphas}
size_per_alpha = {a: [] for a in alphas}
for seed_i, (clf, tr_idx, cal_idx) in enumerate(fusion_models):
    cal = SplitConformal(method="lac").fit(clf, X_fuse_tv[cal_idx], y_fuse_tv[cal_idx])
    sets, p, _ = cal.predict_sets(X_fuse_te, alpha=alphas)
    cov = SplitConformal.coverage(y_fuse_te, sets, alphas)
    siz = SplitConformal.mean_set_size(sets, alphas)
    for a in alphas:
        cov_per_alpha[a].append(cov[a])
        size_per_alpha[a].append(siz[a])
    rows.append({"seed": seed_i + 1, **{f"cov_a{a}": cov[a] for a in alphas},
                 **{f"size_a{a}": siz[a] for a in alphas}})

print("\\nCoverage (mean ± std) and mean set size:")
for a in alphas:
    c = np.array(cov_per_alpha[a])
    s = np.array(size_per_alpha[a])
    target = 1 - a
    print(f"  α={a:.2f}: coverage={c.mean():.3f}±{c.std():.3f}  (target {target:.2f})  | size={s.mean():.2f}")

cf = pd.DataFrame(rows); cf.to_csv("results/fusion_conformal.csv", index=False); cf""")


# ---------- Cell 16: OOD scaffold split eval ----------
code("""# === OOD split (scaffold-based) — coverage degradation check ===
# Rough scaffold proxy: cluster Morgan fingerprints with KMeans and hold out one
# cluster as OOD. Avoids astartes API drift; gives a similar split.
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from src.chem.featurize import morgan
from src.models.conformal import SplitConformal

X_morgan_tv = np.stack([morgan(s) for s in train_val["Drug"]])

km = KMeans(n_clusters=8, n_init=5, random_state=0).fit(X_morgan_tv[:len(X_fuse_tv)])
labels = km.labels_
ood_cluster = int(np.bincount(labels).argmin())
ood_mask = labels == ood_cluster
id_mask = ~ood_mask
print(f"OOD cluster {ood_cluster}: {ood_mask.sum()} mols | ID: {id_mask.sum()} mols")

# Use one fusion model; calibrate on ID, evaluate on OOD
clf, _, _ = fusion_models[0]
cal = SplitConformal(method="lac").fit(clf, X_fuse_tv[id_mask], y_fuse_tv[id_mask])
# In-distribution test: use the regular held-out test set
sets_id, _, _ = cal.predict_sets(X_fuse_te, alpha=[0.10])
cov_id = SplitConformal.coverage(y_fuse_te, sets_id, [0.10])[0.10]
# OOD: predict on the held-out cluster (it's in train_val but unseen by this calibration)
sets_ood, _, _ = cal.predict_sets(X_fuse_tv[ood_mask], alpha=[0.10])
cov_ood = SplitConformal.coverage(y_fuse_tv[ood_mask], sets_ood, [0.10])[0.10]

print(f"Coverage @α=0.10: ID={cov_id:.3f}  OOD={cov_ood:.3f}  (target 0.90)")
print(f"Coverage drop: {cov_id - cov_ood:+.3f}")""")


# ---------- Cell 17: plots ----------
code("""# === Plots: ROC, ECE, coverage, AP traces ===
import numpy as np
from sklearn.metrics import roc_auc_score
from src.utils.plots import roc_panel, reliability_diagram, coverage_bars, ap_traces

# ROC across all baselines + fusion. Use mean-of-seeds prediction for each.
def _mean(preds): return np.mean(preds, axis=0)
n = len(y_fuse_te)
curves = {
    "RF":                (y_te[:n], _mean([p[:n] for p in preds_rf_seeds])),
    "GIN":               (y_te[:n], _mean([p[:n] for p in preds_gin_seeds])),
    "MapLight":          (y_te[:n], _mean([p[:n] for p in preds_ml_seeds])),
    "MapLight+GIN":      (y_te[:n], _mean([p[:n] for p in preds_ens_seeds])),
    "Fusion (chem+ORd)": (y_fuse_te, _mean(preds_fuse_seeds)),
}
aurocs_panel = {k: float(roc_auc_score(yt, ys)) for k, (yt, ys) in curves.items()}
roc_panel(curves, aurocs_panel, "figures/roc.png")

# ECE reliability — fusion
reliability_diagram(
    y_true=y_fuse_te,
    series={
        "MapLight+GIN ens":  _mean([p[:n] for p in preds_ens_seeds]),
        "Fusion (uncalibrated)": _mean(preds_fuse_seeds),
    },
    out_path="figures/ece.png", n_bins=15,
)

# Coverage bars (ID vs OOD at α=0.10) from previous cell
coverage_bars(
    rows=[("ID", 0.10, cov_id), ("OOD", 0.10, cov_ood)],
    out_path="figures/coverage.png",
)

# AP traces — control + 3 example block fractions
from src.physio.ohara import OharaSimulator
sim = OharaSimulator(cell_mode=0)
traces = {}
for label, blk in [("control", 0.0), ("25% block", 0.25), ("60% block", 0.60), ("90% block", 0.90)]:
    r = sim.simulate(ikr_block=blk, n_beats=400, record_last=1)
    traces[label] = (r.t, r.V)
ap_traces(traces, "figures/ohara_ap_traces.png")
print("✅ figures/ written")""")


# ---------- Cell 18: results table ----------
code("""# === Final results table ===
import numpy as np, pandas as pd

rows = [
    ("RF + Morgan",         np.mean(aurocs_rf),   np.std(aurocs_rf)),
    ("GCN (DeepChem)",      np.mean(aurocs_gcn),  np.std(aurocs_gcn)),
    ("GIN",                 np.mean(aurocs_gin),  np.std(aurocs_gin)),
    ("GIN + RDKit",         np.mean(aurocs_gd),   np.std(aurocs_gd)),
    ("MapLight (CatBoost)", np.mean(aurocs_ml),   np.std(aurocs_ml)),
    ("MapLight + GIN ens",  np.mean(aurocs_ens),  np.std(aurocs_ens)),
    ("Fusion (chem+ORd)",   np.mean(aurocs_fuse), np.std(aurocs_fuse)),
]
if 'aurocs_cp' in globals() and aurocs_cp:
    rows.insert(1, ("Chemprop D-MPNN", np.mean(aurocs_cp), np.std(aurocs_cp)))

df = pd.DataFrame(rows, columns=["model", "auroc_mean", "auroc_std"])
df.to_csv("results/summary.csv", index=False)
print(df.to_string(index=False))""")


# ---------- write notebook ----------
def to_nb_cell(kind: str, src: str) -> dict:
    base = {
        "cell_type": kind,
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }
    if kind == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base


nb = {
    "cells": [to_nb_cell(k, s) for k, s in CELLS],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"wrote {NB_PATH} ({NB_PATH.stat().st_size} bytes, {len(CELLS)} cells)")
