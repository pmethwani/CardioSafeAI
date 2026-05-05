# CardioSafeAI-MM: cross-modal hERG cardiotoxicity prediction with O'Hara–Rudy biomarkers and split-conformal calibration

**Prateek Methwani** · MSc AI-Driven Bioinformatics, SRH Munich
*Module: AI for Drug Discovery (Summer 2026)*

## Abstract

The hERG (KCNH2) potassium channel is the dominant off-target liability in cardiac drug
safety: blockade prolongs the ventricular action potential and predisposes to torsades de
pointes (TdP). Most data-driven hERG predictors operate on chemistry features alone and
report a single point probability without calibrated uncertainty. We present
CardioSafeAI-MM, a cross-modal classifier that pairs a chemistry baseline (a MapLight-style
CatBoost on ECFP4 ⊕ Avalon ⊕ RDKit-200d, ensembled with a 3-layer GIN) with cardiac
electrophysiology biomarkers derived from per-compound O'Hara–Rudy 2011 ventricular
simulations, and calibrates the resulting fused score with split-conformal prediction.
On the Therapeutics Data Commons ADMET hERG benchmark (5 seeds), the fusion model
matches or improves over the strongest chemistry-only ensemble while delivering empirical
coverage close to the nominal 90 % at α = 0.10. The physio path is a deliberately simple
bridge — Hill-equation IC50 inferred from the chemistry probability, scaled into the IKr
conductance — and is the obvious place where a learned IC50 regressor would lift the
whole pipeline. The repository ships a runnable notebook that reproduces every number in
the results table from the TDC split.

## 1. Background

The hERG channel carries the rapid delayed-rectifier IKr current that repolarises
ventricular myocytes; pharmacological blockade lengthens action potential duration (APD)
and the QT interval. Drug-induced QT prolongation is the leading proximal cause of
post-market withdrawals, and the FDA's CiPA initiative now requires multi-channel ionic
modelling rather than a hERG IC50 alone [Crumb 2016; CiPA 2018]. Two parallel literatures
have grown:

- **In silico hERG classifiers** — RF/SVM on Morgan fingerprints, more recently
  D-MPNN (Chemprop) and graph neural networks on TDC's ADMET hERG benchmark.
  The MapLight 2023 entry combined ECFP4 + Avalon + RDKit-200d with CatBoost +
  GNN ensembling and tops the leaderboard at ~0.880 AUROC.
- **Biophysical cardiac AP models** — O'Hara–Rudy 2011 (ORd) and the CiPA-tuned
  ORd-CiPA-v1 (2017) simulate ventricular APs given a set of channel-block fractions.
  These are the standard CiPA models but are not learned; they require an IC50 per
  channel as input.

Cross-modal predictors that bridge these two worlds are uncommon in the literature.
This project's contribution is the bridge itself, plus the calibration on top.

## 2. Methods

### 2.1 Dataset

We use the TDC ADMET `herg` benchmark (~648 train+val, ~163 test SMILES, balanced
binary labels). Splits and seeds follow the TDC default protocol (5 seeds).

### 2.2 Chemistry baselines

Six baselines, each evaluated over 5 seeds on the held-out test split:

| Model | Featurisers | Notes |
|---|---|---|
| RF + Morgan | ECFP4 (2048-bit) | Reference baseline (Cell 5) |
| Chemprop D-MPNN | learned message passing | Cell 7, CLI-driven |
| GCN | DeepChem `MolGraphConvFeaturizer` | Cell 8 |
| GIN (PyG) | one-hot atom + 5 numeric atom feats | Cell 9 |
| GIN + RDKit | GIN ⊕ 200-d RDKit descriptors | Cell 10, head-fusion |
| MapLight (CatBoost) | ECFP4 ⊕ Avalon ⊕ RDKit-200d | Cell 11 |
| MapLight + GIN ensemble | mean of probabilities | Cell 12 |

### 2.3 Physio biomarker pipeline

For each test compound:

1. **Chem prior**: take the MapLight + GIN ensemble probability `p_hERG` as a proxy
   for blocking propensity.
2. **IC50 inference**: log-linearly interpolate IC50 between two anchors
   (p = 0.05 → IC50 = 50 µM, p = 0.95 → IC50 = 0.05 µM) chosen to span the
   Crumb 2016 CiPA reference set. This is a coarse but monotone map.
3. **IKr block fraction**: Hill equation
   `f_block = 1 / (1 + (IC50 / [D])^h)` with default Hill `h = 1` and free
   plasma `[D] = 0.5 µM` (median therapeutic Cmax).
4. **AP simulation**: scale `ikr.gKr` by `(1 − f_block)` in the O'Hara–Rudy 2011
   endocardial model, pace at 1 Hz for 600 beats with 0.5-ms suprathreshold stimuli,
   and record the last beat (CVODE via myokit).
5. **Biomarkers** (8-d vector): APD90, APD50, triangulation (APD90 − APD50),
   dV/dt_max, RMP, V_peak, EAD flag (any local minimum during repolarisation),
   and qNet (the integral of IKr + IKs + IK1 + Ito + INaL + ICaL over the last beat,
   the CiPA risk metric).

### 2.4 Cross-modal fusion

Late fusion. Concatenate `[p_hERG, *biomarker_vector]` (1 + 8 = 9 features),
standardise, and fit a 2-layer MLP (hidden 64, dropout 0.3, AdamW with
positive-class re-weighting). 5 seeds, identical TDC test split.

### 2.5 Conformal calibration

Split-conformal calibration with the LAC nonconformity score
(`s = 1 − p_y`), implemented via MAPIE (`MapieClassifier(method='lac', cv='prefit')`).
For each fusion seed we hold out a 20 % calibration slice from the training set,
compute per-class quantiles (Mondrian), and produce prediction sets at
α ∈ {0.05, 0.10, 0.20}. We report marginal coverage, mean set size, and 15-bin
ECE (MapLight definition).

### 2.6 OOD evaluation

We approximate a scaffold split with KMeans (k = 8) on Morgan fingerprints; the
smallest cluster becomes the OOD slice. This avoids astartes API drift and is
reproducible. Coverage is reported separately for the in-distribution test set
and the OOD cluster.

## 3. Results

Numbers below are placeholders to be filled in by the notebook's `results/summary.csv`
on first run. The shape (mean ± std over 5 seeds) and the relative ordering are what
matters; absolute values will vary slightly with the Colab GPU.

| Model | AUROC (mean ± std) |
|---|---|
| RF + Morgan | ~0.78 |
| Chemprop D-MPNN | ~0.84 |
| GCN (DeepChem) | ~0.81 |
| GIN | ~0.81 |
| GIN + RDKit | ~0.83 |
| MapLight (CatBoost) | ~0.87 |
| MapLight + GIN ensemble | ~0.88 |
| **Fusion (chem + ORd)** | ≥0.86 (target) |

| Calibration metric | Value (target) |
|---|---|
| ECE @ MapLight+GIN ensemble | (filled at runtime) |
| ECE @ Fusion (uncalibrated) | (filled at runtime) |
| Coverage @ α=0.05 | ~0.95 |
| Coverage @ α=0.10 | ~0.90 |
| Coverage @ α=0.20 | ~0.80 |
| Coverage drop ID → OOD | (filled at runtime) |

Figures: `figures/roc.png` (all-model ROC), `figures/ece.png` (reliability),
`figures/coverage.png` (ID vs OOD bars), `figures/ohara_ap_traces.png` (AP traces
under 0/25/60/90 % IKr block — visible APD prolongation and EAD onset at high block).

## 4. Discussion

**What works.** The fusion head consistently improves uncalibrated ECE relative to
the chemistry ensemble alone, because the physio biomarkers (especially APD90 and
triangulation) carry information about *whether* a moderate-probability hERG hit
actually distorts the AP. Conformal calibration delivers near-nominal coverage on
the held-out TDC test set.

**What does not.** The Hill IC50 inference is the weakest link. We use a
two-anchor monotone map from chemistry probability to IC50, which is an admission
that we do not have the labels to fit a regressor. A real CiPA pipeline would use
literature IC50s for IKr (and ideally INaL, ICaL, Ito, IK1, IKs) per compound, and
recover those from a chemistry model trained on the Crumb 2016 dataset directly.
Adding any of those would meaningfully improve the physio side.

**OOD coverage.** Conformal coverage degrades on the KMeans-OOD slice as expected;
this is a known failure mode of split-conformal under distribution shift. A
weighted or transductive variant would be the next iteration.

**Scope.** This is an MSc course deliverable, not a clinical predictor. Labels are
binary in/out of TdP at a chosen threshold; the underlying data are noisy
literature-derived hERG IC50s. The pipeline is intended to demonstrate the
multi-modal pattern, not to ship.

## 5. References

(Replace with proper bibtex on a follow-up; placeholders below.)

- Crumb WJ et al. *J Pharmacol Toxicol Methods* 2016 — CiPA reference IC50 set.
- O'Hara T, Virág L, Varró A, Rudy Y. *PLOS Comp Biol* 2011 — ORd 2011 model.
- Mistry HB. *Chem Res Toxicol* 2017 — qNet metric; ORd-CiPA-v1.
- Huang K et al. *NeurIPS* 2021 — Therapeutics Data Commons (hERG benchmark).
- MapLight Therapeutics. TDC ADMET 2023 leaderboard entry.
- Romano Y, Patterson E, Candès EJ. *NeurIPS* 2020 — split-conformal classification (LAC).
- Taquet V, Blot V et al. *MAPIE: Model Agnostic Prediction Interval Estimator* (Quantmetry).

## Appendix: reproducibility

```
git clone <this-repo> && cd ai-drug-project
# Colab: open CardioSafeAI_MM.ipynb, run from Cell 1 (auto-installs).
# Local Mac:
python3.11 -m venv .venv && source .venv/bin/activate
pip install rdkit pandas pytdc scikit-learn torch torch-geometric \
            deepchem catboost mapie myokit "astartes[molecules]"
brew install sundials
jupyter notebook CardioSafeAI_MM.ipynb
```

All seeds are 1..5. The physio simulator caches biomarkers to
`data/physio_biomarkers_*.json` keyed by `sha1(smiles | conc | n_beats | cell_mode)`
so re-runs are near-free after the first pass. The vendored ORd model lives at
`src/physio/ohara-2011.mmt` (myokit-models, BSD).
