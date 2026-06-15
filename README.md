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

---

# Research log — exhausting the OOF headroom

After the 11.08 LGB, a second research pass tried to break past it from
scratch and then by meta-combination. **Every path converged on the same
wall: 11.08 is a data ceiling, not a method failure.** All numbers below are
GroupKFold(5) pooled OOF TVT RMSE (the competition metric), leave-own-well-out.

## Why the problem is hard (the reframe)

`b_well` is a true per-well constant (std ~0.01 ft), but `eff_depth = TVT + Z`
is **not** constant along a well — `F(x,y)` swings a **median of 142 ft**
(max 416) within a single eval zone, because the lateral travels thousands of
ft in (x,y). So the entire task is predicting the **per-row** F surface along
the toe. The naive "F = last known eff" constant scores **107.5**. The worst
~77 wells (10%) carry **~90% of all squared error** — they are spatially
isolated / faulted, and dominate the pooled metric.

## Standalone from-scratch models

| model | OOF pooled | why it caps out |
|---|---|---|
| GP on 773 per-well rep points | **177.8** | worse than naive — pairs whole-well-median (x,y) with eval-median eff → misregistered surface; 773 points can't resolve the 142 ft within-well swing |
| Dense-cloud kriging (IDW/plane + anchor) | **116** | median 13.6 (great typical well) but tail explodes to 826; 89% of error in worst 10% |
| Own-dip "Tucker" cumsum (+ confidence routing) | **44.8** | constant-dip extrapolation; oracle pick(cloud,tucker) only 24.8 |
| Offset-drift LightGBM classifier | **37.8** | even a *perfect* offset caps at 15.06 (within-eval F curvature); drift only ~19% predictable from the known zone |
| From-scratch GR alignment (self / typewell) | **104–188** | GR is a *fine-tuner of a spatial prior*, not a standalone locator — fails when centered on a bad prior |
| **existing LGB (already fuses all of the above)** | **11.08** | — |

Key sub-findings: self-correlation (pre-PS horizontal GR) slightly beats
typewell GR (own wins on 430/773 wells) — confirms the task PPTX slide 9. The
existing `align2` feature reaches **30.7** standalone *only* because it centers
its ±15 ft GR search on `plane_dF_path` (a local-plane path integral), which
is a far better spatial prior than cloud IDW on the tail.

## Meta-combination of the existing components

| task | method | OOF pooled |
|---|---|---|
| A | blend `w₁·lgb + w₂·align2 + w₃·tucker` (grid + LS) | 11.06 (≈ no gain; align2/tucker are already LGB inputs) |
| B | hard gate lgb→tucker on `plane_fit_rms`>P75 ∧ `|anchor_b|`>P75 | 14.27 (worse) |
| B | **oracle** gate (gate the wells where tucker actually beats lgb) | **10.05** (proves ~1 ft headroom exists) |
| C | Level-2 soft router (shallow LGB, per-row `model_disagreement`) | 11.08–11.32 (no gain) |

The oracle's 1 ft is **not realizable**: the confidence features flag where
LGB is *uncertain*, but "uncertain" ≠ "tucker *wins* here" — which depends on
dip-steadiness the features don't encode.

## Things that didn't work on the leaderboard

- Blending the 11.08 LGB into the public `sp45+fleongg` blend **hurt** the LB
  (exact-match recovery 7.596, 0.85/0.15 3-way 7.664, both > 7.535). The LGB
  is too weak at 11.08 to help; it would need < 9.
- Improving `plane_dF_path` for the tail via neighbor gating
  (`dist<2000 ∧ |Δdip|<0.01`) is a **null result** — gated ≡ ungated to the
  decimal, because **60.5% of tail eval rows have *zero* neighbors passing the
  gate**. The tail is data-poverty-limited, not selection-limited.

## One genuinely open lead

`F = TVT + Z = formation_marker + const_well`, **exact to 0.01 ft** for every
marker column (`ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA`), `const_well`
identical in known vs eval zone. The markers are **absent from the local test
wells** (train-only, redundant with F there). *If* the hidden Kaggle test
carries them in the eval zone, `TVT = marker − Z + const_well` is a sub-1-ft
solution — verify with [src/kaggle_marker_check.py](src/kaggle_marker_check.py)
(one Kaggle commit). Almost certainly a leak that won't be in the real test,
but the upside is the whole competition.

## numpy 2.x fix

`CtxView.query_plane` used `np.linalg.solve(M, v)` with `M=(n,3,3)`,
`v=(n,3)`. numpy ≥ 2.0 changed batched-solve semantics and raises
`ValueError` (not `LinAlgError`, so the old `except` missed it) → on Kaggle
`extract_well` crashed and LGB silently fell back to a constant. Fixed to
`solve(M, v[..., None])[..., 0]` (unambiguous on numpy 1.x/2.x) with a per-row
`pinv` fallback. Verified numerically identical on numpy 1.26.

## Source-file guide (research scripts)

| file | what it does |
|---|---|
| `src/gp_kriging.py`, `src/run_gp_kriging.py` | GP on rep points (177.8) |
| `src/tucker_routed.py` | own-dip cumsum + confidence routing (44.8) |
| `src/tucker_classifier.py` | LightGBM offset-drift classifier (37.8) |
| `src/gr_alignment.py` | from-scratch per-row GR alignment + fusion |
| `src/blend_refit.py` | Task A — OOF blend re-fit (regenerates LGB OOF) |
| `src/tail_rescue.py` | Task B — tail gating + learnable meta-gate |
| `src/meta_router.py` | Task C — Level-2 soft router |
| `src/kaggle_marker_check.py` | Kaggle-only diagnostic + exact marker solver |
| `src/kaggle_lgb_final_cell.py` | Kaggle notebook cell for the 3-way blend |

## Bottom line

11.08 is the data ceiling for this feature set and these methods. The ~77
tail wells that dominate the metric are genuinely unpredictable from (x,y),
own-dip, or GR. Beating 7.535 LB requires a **genuinely new signal** (e.g.
hidden-test markers, or a stronger public base model to blend) — not another
variation on the existing model.
