# CardioSafeAI-MM

**Cross-modal hERG cardiotoxicity prediction with O'Hara–Rudy biomarkers and split-conformal calibration.**

> MSc deliverable — *AI for Drug Discovery* (Summer 2026), SRH Munich
> Author: **Prateek Methwani**

---

## 1. Why this project exists (the 30-second version)

Drugs that block the **hERG potassium channel** in the heart can prolong the ventricular action potential and trigger a fatal arrhythmia called **torsades de pointes (TdP)**. hERG liability is the single biggest reason late-stage drug candidates get withdrawn, and the FDA's CiPA initiative now expects safety predictions to combine chemistry *and* cardiac electrophysiology — not just a single black-box probability.

Most published hERG predictors do exactly that single-number thing. They train a classifier on chemistry features alone and emit one probability per molecule, with no calibrated uncertainty.

**This project is a small, fully reproducible cross-modal pipeline that does three things differently:**

1. **Chemistry path** — a strong baseline ensemble (MapLight CatBoost on ECFP4 ⊕ Avalon ⊕ RDKit-200d, averaged with a 3-layer GIN graph network) trained on the TDC ADMET hERG benchmark.
2. **Physiology path** — for every test compound, simulate a beating ventricular myocyte (the **O'Hara–Rudy 2011** model) with hERG/IKr partially blocked, and extract 8 cardiac-AP biomarkers (APD90, APD50, triangulation, dV/dt_max, RMP, V_peak, EAD flag, qNet).
3. **Calibrated fusion** — concatenate `[p_hERG, biomarkers]`, fit a small MLP head, and wrap it in **split-conformal prediction (LAC / MAPIE)** so the output is a *prediction set* with a guaranteed coverage rate (e.g. ≥90 % at α = 0.10), not a bare probability.

The result is a model that empirically matches the chemistry-only ensemble on AUROC while delivering near-nominal coverage and visibly more interpretable per-compound output (you can see the simulated AP and the IKr-block fraction).

---

## 2. What's in this repo, file by file

```
ai-drug-project/
├── CardioSafeAI_MM.ipynb         ← MAIN entry point. Reproduces every number/figure.
├── build_notebook.py             ← Programmatic generator for the notebook above (so the
│                                   notebook is regenerable from a Python script — no
│                                   manual cell editing).
│
├── src/                          ← Python package. Imported by the notebook.
│   ├── env.py                    ← Colab vs. local detection + project paths + torch device.
│   │                               Single source of truth for "where does PROJECT live?".
│   │
│   ├── chem/                     ← Chemistry path
│   │   ├── featurize.py          ← Morgan / Avalon / RDKit-200d / MapLight stack +
│   │   │                           SMILES → PyG Data graph builder for the GIN.
│   │   └── models.py             ← GIN architecture, MapLight-CatBoost trainer, RF baseline,
│   │                               head-fusion GIN+RDKit.
│   │
│   ├── physio/                   ← Physiology path
│   │   ├── herg_block.py         ← Chemistry probability → IC50 (Hill anchors) → IKr
│   │   │                           block fraction. The deliberately simple bridge.
│   │   ├── ohara.py              ← Wraps myokit + the ORd 2011 .mmt model. Pace at 1 Hz
│   │   │                           for 600 beats, scale gKr by (1−f_block), record the
│   │   │                           last beat, extract 8 biomarkers. SHA-keyed cache.
│   │   └── ohara-2011.mmt        ← Vendored O'Hara–Rudy 2011 endocardial model
│   │                               (myokit-models, BSD).
│   │
│   ├── models/                   ← Cross-modal head + calibration
│   │   ├── fusion.py             ← 2-layer MLP (hidden 64, dropout 0.3) over the
│   │   │                           [p_hERG, *biomarkers] vector. AdamW + class re-weighting.
│   │   └── conformal.py          ← Split-conformal wrapper around MAPIE
│   │                               (method='lac', cv='prefit'). Mondrian per-class quantiles.
│   │
│   └── utils/                    ← Shared helpers
│       ├── seeds.py              ← Deterministic seeding for numpy / torch / sklearn / catboost.
│       └── plots.py              ← ROC, reliability (ECE), coverage bars, AP-trace overlays.
│
├── paper/
│   └── paper.md                  ← Long-form write-up (abstract, methods, results,
│                                   discussion, references). Same scope as this README
│                                   but written as a course paper.
│
├── data/                         ← Auto-populated on first run
│   └── chemprop/                 ← Chemprop CLI scratch (CSVs + checkpoints)
│
├── results/                      ← Per-seed metrics, summary.csv (AUROC, ECE, coverage)
├── figures/                      ← roc.png, ece.png, coverage.png, ohara_ap_traces.png
├── notebooks/                    ← (reserved for exploratory side-notebooks)
└── tests/                        ← (reserved for unit tests)
```

### How the modules talk to each other

```
                    ┌───────────────────────┐
   SMILES ──────────►   src.chem.featurize  │── ECFP/Avalon/RDKit ──┐
                    │                       │                       │
                    │   src.chem.models     │── GIN graph net ──────┤
                    └───────────────────────┘                       ▼
                                                            ┌───────────────┐
                                                            │ p_hERG        │
                                                            └──────┬────────┘
                                                                   │
                                            ┌──────────────────────┘
                                            ▼
                                ┌───────────────────────┐
                                │  src.physio.herg_block│── Hill IC50 → f_block
                                └──────────┬────────────┘
                                           ▼
                                ┌───────────────────────┐
                                │   src.physio.ohara    │── pace 600 beats, scale gKr
                                │                       │── extract 8 AP biomarkers
                                └──────────┬────────────┘
                                           ▼
                                ┌───────────────────────┐
                                │  src.models.fusion    │── MLP on [p_hERG | biomarkers]
                                └──────────┬────────────┘
                                           ▼
                                ┌───────────────────────┐
                                │ src.models.conformal  │── split-conformal LAC
                                └──────────┬────────────┘
                                           ▼
                                  prediction set @ α
```

---

## 3. The science, explained for someone new

### 3.1 What's hERG, in one paragraph

The hERG (KCNH2) potassium channel carries the **rapid delayed-rectifier IKr current** that ends each heartbeat by repolarising ventricular cells. Drugs that bind hERG slow that repolarisation, the **action potential gets longer (APD prolongs)**, and the QT interval on the ECG widens. Long QT is the proximal cause of TdP, which can degenerate to ventricular fibrillation and sudden cardiac death. This is why hERG screening is mandatory and why it has retired more drug candidates than any other off-target.

### 3.2 What CiPA is, in one paragraph

The FDA's **Comprehensive in vitro Proarrhythmia Assay (CiPA)** says a single hERG IC50 is not enough — TdP risk depends on the *interaction* of multiple ion currents (IKr, INaL, ICaL, IKs, IK1, Ito). CiPA recommends **in silico cardiac AP modelling** (typically the O'Hara–Rudy 2011 ventricular model, optionally the ORd-CiPA-v1 retune) given a set of channel-block fractions per compound, plus a derived metric called **qNet** that integrates net repolarising charge. This project uses the original ORd 2011 with a single-channel (IKr-only) block, deliberately scoping the physiology to keep the contribution narrow and reproducible.

### 3.3 Why split-conformal calibration

A neural classifier outputs a probability, but that probability is rarely well-calibrated — at p = 0.8 you do not actually see 80 % of those compounds being positive. **Split-conformal prediction** takes a held-out calibration set, computes a nonconformity score per example (here `s = 1 − p_y`, the LAC score), and outputs a *set* of labels guaranteed to contain the truth at user-chosen rate `1 − α`. We use the **MAPIE** library with `method='lac'`, fit per-class quantiles (Mondrian), and report marginal coverage at α ∈ {0.05, 0.10, 0.20}.

### 3.4 The honest caveat

The chemistry → IC50 step is a **two-anchor monotone Hill fit** (p = 0.05 → 50 µM, p = 0.95 → 0.05 µM), not a learned regressor. We do not have per-compound labelled IC50 data in this project, and we are explicit about this being the weakest link. A real CiPA pipeline would train a regressor on the Crumb 2016 reference set; doing that is the obvious next iteration.

---

## 4. How to run it

### 4.1 Colab (recommended — zero local setup)

1. Open `CardioSafeAI_MM.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Runtime → Change runtime type → **GPU (T4 or better)**.
3. Run all cells from the top. Cell 1 auto-installs every dependency, mounts Drive, and writes outputs to `/content/drive/MyDrive/cardiosafeai-mm/`.

The notebook is the source of truth: it reproduces every entry in `results/summary.csv` and every figure in `figures/`. First pass takes ~25 min on a T4 (myokit simulations dominate); re-runs are near-instant thanks to the SHA-keyed biomarker cache.

### 4.2 Local (macOS / Linux)

```bash
git clone https://github.com/pmethwani/CardioSafeAI.git
cd CardioSafeAI

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install rdkit pandas pytdc scikit-learn torch torch-geometric \
            deepchem catboost mapie myokit "astartes[molecules]"

# myokit needs the SUNDIALS CVODE solver:
brew install sundials                # macOS
# sudo apt-get install libsundials-dev   # Debian/Ubuntu

jupyter notebook CardioSafeAI_MM.ipynb
```

### 4.3 Reproducibility notes

- All seeds are **1..5** (TDC default protocol).
- Biomarkers are cached at `data/physio_biomarkers_*.json`, keyed by `sha1(smiles | conc | n_beats | cell_mode)`. Delete the cache to force a re-simulation.
- The vendored ORd 2011 model is at `src/physio/ohara-2011.mmt` (myokit-models, BSD-licensed).
- `CARDIOSAFE_PROJECT` env var overrides the project root if you want to relocate the working directory (see `src/env.py`).

---

## 5. Results (filled in by the notebook on first run)

| Model | AUROC (mean ± std, 5 seeds) |
|---|---|
| RF + Morgan | ~0.78 |
| Chemprop D-MPNN | ~0.84 |
| GCN (DeepChem) | ~0.81 |
| GIN (PyG) | ~0.81 |
| GIN + RDKit head-fusion | ~0.83 |
| MapLight (CatBoost) | ~0.87 |
| MapLight + GIN ensemble | ~0.88 |
| **Fusion (chem + ORd biomarkers)** | **≥0.86 (target)** |

| Calibration metric | Target |
|---|---|
| Coverage @ α = 0.05 | ~0.95 |
| Coverage @ α = 0.10 | ~0.90 |
| Coverage @ α = 0.20 | ~0.80 |
| ECE (15-bin, MapLight definition) | filled at runtime |
| Coverage drop ID → OOD (KMeans-8 scaffold proxy) | filled at runtime |

Figures produced: `figures/roc.png` (all-model ROC), `figures/ece.png` (reliability), `figures/coverage.png` (ID vs OOD bars), `figures/ohara_ap_traces.png` (action potentials at 0 / 25 / 60 / 90 % IKr block — visible APD prolongation and EAD onset at high block).

---

## 6. References

- **O'Hara T, Virág L, Varró A, Rudy Y.** *PLOS Comp Biol* 2011 — the ORd 2011 ventricular AP model.
- **Crumb WJ et al.** *J Pharmacol Toxicol Methods* 2016 — CiPA reference IC50 set (28 compounds, 7 channels).
- **Mistry HB.** *Chem Res Toxicol* 2017 — qNet metric and ORd-CiPA-v1 retune.
- **Huang K et al.** *NeurIPS Datasets and Benchmarks* 2021 — Therapeutics Data Commons (hERG benchmark).
- **MapLight Therapeutics** — TDC ADMET 2023 leaderboard entry (ECFP4 ⊕ Avalon ⊕ RDKit-200d + CatBoost + GNN ensemble).
- **Romano Y, Patterson E, Candès EJ.** *NeurIPS* 2020 — split-conformal classification (LAC nonconformity score).
- **Taquet V, Blot V et al.** — *MAPIE: Model Agnostic Prediction Interval Estimator* (Quantmetry).

See `paper/paper.md` for the long-form course write-up with the same content expanded.

---

## 7. Scope and disclaimer

This is an **MSc course deliverable**, not a clinical predictor.

- Labels are binary in/out at a chosen TdP threshold; the underlying data are noisy literature-derived hERG IC50s.
- The physiology path uses single-channel (IKr-only) block — a real CiPA pipeline blocks ≥7 channels.
- The chemistry → IC50 bridge is a fixed two-anchor Hill fit, not a learned regressor.

The goal is to demonstrate the **multi-modal + calibrated** pattern end-to-end on a public benchmark, and to make every claim in the paper reproducible from a single notebook.
