# ROGII Wellbore Geology Prediction

Kaggle competition: predict TVT (True Vertical Thickness) along the unlogged
eval zone of horizontal wells — pure forward extrapolation from the logged
build section, the offset-well field, and the typewell.

- Metric: RMSE on TVT (lower is better)
- Current best LB: **7.535** (sp45-proj ⊕ fleongg blend)
- This repo: LightGBM residual model, OOF **11.08**, built as an
  uncorrelated third blend member

## Physics

The whole problem reduces to one exact identity (verified on all 773 train
wells, per-well intercept std ~0.01 ft):

```
TVT = -Z + F(x,y) + b_well            b_well constant per well
TVT - last_tvt = -(Z - Z_anchor) + dF
```

`F(x,y)` is a smooth geological surface sampled exactly by every training
well at every row. The model learns only `dF` (drift of F along the lateral);
the `-(Z - Z_anchor)` ramp is added back analytically at reconstruction.
Trees cannot reproduce a smooth linear ramp — detrending it out is worth more
than any feature.

## Architecture

```
773 train wells ──> knn_context.npz (1M-pt cloud of F = TVT + Z, per-well dips)
                          │
horizontal well ──> features.py: 73 features per eval-zone row
typewell        ──         │
                          ▼
        LightGBM (255 leaves, lr 0.02, 2 seeds × 5 GroupKFold folds)
        CatBoost (depth 8, same folds — 0 blend weight, kept as artifact)
                          │
                          ▼
        TVT = last_tvt - (Z - Z_anchor) + dF_pred
```

Feature families (all computable identically for train and test):

| family | examples | note |
|---|---|---|
| Local F-surface plane fit | `plane_dF_path`, `plane_gx/gy`, `plane_fit_rms` | strongest signal; batched weighted LSQ over 32 spatial neighbors, gradient path-integrated along the lateral |
| TVT-domain GR alignment | `align2_tw_est`, `align2_own_*` | geosteering-style: structure first, then GR(TVT) curve matching ±15 ft |
| Own known-zone extrapolators | tail dips/slopes (30/100/500), velocity model `dTVT = β·dZ + c` | only signal for spatially isolated wells |
| Spatial KNN | `knn_dF`, `egfdu_drift`, neighbor dips | leave-own-well-out KD-trees for train wells |
| GR / trajectory | rolling GR, `dz_per_ft`, trig dip, missingness | gating features |

## Results

| # | experiment | score |
|---|---|---|
| 1 | Ridge on tabular features (absolute TVT) | LB 16.09 |
| 2 | Ridge on residuals (TVT − last_tvt) | LB 14.12 |
| 3 | Ridge + sliding GR correction | LB 14.20 |
| 4 | Own particle filter (500 particles, 1 seed) | LB ~16 |
| 5 | sp45-proj fork (128-seed PF + beam) | LB 7.893 |
| 6 | **sp45 ⊕ fleongg blend (0.62/0.38)** | **LB 7.535** |
| 7 | Z-cumsum continuous GR search | LB 60–120 |
| 8 | Z-discrete classifier (7 candidates) | LB 51 |
| 9 | sklearn MLP | OOF 22 |
| 10 | This repo, v1 features (LGB) | OOF 12.04 |
| 11 | This repo, v2 + plane fit + GR alignment (LGB) | OOF 11.26 |
| 12 | **This repo, final (255 leaves, 2-seed avg)** | **OOF 11.08** |

Flat-physics baseline (dF = 0): OOF 107.49. Full run-by-run details in
[experiments.md](experiments.md); domain insights in
[docs/competition_notes.md](docs/competition_notes.md).

## Pipeline

```
python src/build_context.py    # spatial F cloud -> models/knn_context.npz (~10 s)
python src/extract_train.py    # 3.78M eval-zone rows x 73 features (~7 min)
python src/train_model.py      # GroupKFold(5) by well, 2-seed LGB + CB (~40 min)
python src/predict_test.py     # -> submission.csv
```

All artifacts are **pickle-free** (native `.txt`/`.cbm` model formats, `.npz`
arrays, JSON scaler) — no library version pinning needed at inference.

## Kaggle inference

Trained weights: Kaggle dataset
[`homeshwarrao/rogii-lgb-weights-v1`](https://www.kaggle.com/datasets/homeshwarrao/rogii-lgb-weights-v1)
(see [models/README.md](models/README.md)). Attach it plus the competition
data to a notebook and run [kaggle_inference.py](kaggle_inference.py) —
inference only, ~5 min for ~200 wells, no retraining.

## Validation

GroupKFold(5) by well ID — never split within a well. OOF RMSE is computed on
absolute TVT after reconstruction. The local `test/` folder is a 3-well
sample whose wells are truncated train wells; trust the OOF, not local test.
