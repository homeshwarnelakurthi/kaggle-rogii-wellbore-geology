# Experiment log

Chronological record of everything tried on ROGII Wellbore Geology
Prediction. Metric: RMSE on absolute TVT. "OOF" = out-of-fold,
GroupKFold(5) by well ID over 773 training wells (3.78M eval-zone rows).

## Phase 0 — prior art (before this repo)

| experiment | result | takeaway |
|---|---|---|
| Ridge on tabular features, absolute TVT target | LB 16.09 | absolute target unlearnable; must use residuals |
| Ridge on residuals (TVT − last_tvt) | LB 14.12 | residual framing works |
| Ridge + sliding GR correction | LB 14.20 | naive GR correction hurts |
| Own particle filter (500 particles, 1 seed) | LB ~16 | PF needs many seeds + beam search |
| sp45-proj fork (128-seed likelihood-PF + beam + poly projection) | LB 7.893 | strong public baseline |
| sp45 ⊕ fleongg blend, weights 0.62/0.38 | **LB 7.535** | current best; engines must be diverse |
| Z-cumsum continuous GR search | LB 60–120 | catastrophic — unconstrained GR search snaps to wrong cycles |
| Z-discrete classifier (7 offset candidates) | LB 51 | discrete-only too coarse without PF smoothing |
| sklearn MLP | OOF 22, timed out | wrong tool for this feature scale |

## Phase 1 — LightGBM residual pipeline, v1 features (this repo)

**Setup.** Target = detrended residual `dF = TVT − last_tvt + (Z − Z_anchor)`
(the `-(Z−Z_anchor)` ramp is exact physics; trees should not waste splits
approximating it). 49 features: trajectory geometry, GR rolling stats,
own-known-zone dip/slope/velocity extrapolators, MD-window self-correlation,
IDW spatial KNN of F = TVT + Z (509k-pt cloud, leave-own-well-out),
EGFDU formation KNN. LGB 127 leaves / lr 0.05 / min_child 50;
CatBoost depth 8.

**Result: OOF 12.04 (LGB), 13.50 (CB), best blend 0.95·LGB.**
Flat baseline 107.49. Per-well median 8.23, p90 16.86.

Diagnostics that drove v2:

- `nb_plane_dF` (per-well plane dip × displacement) corr 0.92 with target —
  the regional dip is the signal; a *local* version should be stronger.
- `knn_dF` (IDW) MAE 14.1 ft even with neighbors <500 ft — IDW oversmooths.
- MD-window self-correlation ~useless (MAE 55 ft at conf >0.85): a 51-row MD
  window spans ~1 TVT-ft in the lateral but ~40 TVT-ft in the build section —
  incommensurate stratigraphic scales. Unconstrained argmax also locks onto
  the wrong GR cycle.

## Phase 2 — v2 features

Changes:

1. **Local plane fit of F surface** at every eval row (batched weighted LSQ,
   32 neighbors, gaussian kernel 400 ft, ridge-guarded): `plane_dF`,
   gradient `plane_gx/gy`, `plane_fit_rms`, and `plane_dF_path` = path
   integral of the gradient along the lateral. Per-well MAE ~4–5 ft on
   neighbor-dense wells.
2. **TVT-domain GR alignment** (`seq_align`): smooth lateral GR ≈ GR at a
   single TVT; pick shift δ minimizing |GR_ev − ref(tvt_flat + δ)| over a
   201-row window, vs typewell and own pre-PS GR(TVT) curves.
   Flat-centered ±50 ft version: weak/inconsistent (wrong-cycle locking).
   **Two-stage version** centered on `tvt_flat + plane_dF_path` with ±15 ft
   band (`align2_*`): MAE 6–8 ft standalone — structure first, GR fine-tune
   second, exactly like a geosteering workflow.
3. Banded multi-scale self-correlation (kept, still weak), tail-curvature
   features (`slope_tail_500`, `dip_trend`, `dip_change`), denser cloud
   (1M pts, subsample 5), stronger LGB regularization (63 leaves, lr 0.03,
   min_child 200).

**Result: OOF 11.26 (LGB), 12.80 (CB), best blend 1.00·LGB.**
`plane_dF_path` #2 by importance, `align2_tw_est` #7.

Post-hoc findings:

- `plane_dF_path` *pooled* RMSE as a standalone prediction is 70.6 despite
  per-well median 8.8 — gradient path integration diverges on
  sparse-neighbor wells (worst well 927 ft). Re-centering the target on it
  would poison those wells; let the trees gate it via `knn_dist_mean` /
  `plane_fit_rms` instead.
- Per-well smoothing + anchor-zeroing of OOF predictions: only −0.04 RMSE.
  Predictions already smooth; not the lever.
- Worst well decile carries 51% of squared error (matches the known
  competition stat): long eval zones + distant neighbors + missing GR.

## Phase 3 — final model

Fold-0 config shootout: 63/0.03/mc200 → 10.27; 255/0.02/mc400 → 10.20;
127/0.02/mc300 → 10.22. Chose 255 leaves / lr 0.02 / min_child 400 /
colsample 0.6 / L1 1.0 / L2 10.0, **2 seeds per fold averaged**.

**Result: OOF 11.08 (LGB). CatBoost 12.80, blend weight 0.**
Per-well median 7.68, p90 15.51, max 61.9.

| fold | v1 | v2 | final |
|---|---|---|---|
| 0 | 11.41 | 10.25 | 10.16 |
| 1 | 12.64 | 12.59 | 12.60 |
| 2 | 11.75 | 10.90 | 10.70 |
| 3 | 12.49 | 11.47 | 11.02 |
| 4 | 11.86 | 10.94 | 10.76 |
| **OOF** | **12.04** | **11.26** | **11.08** |

## Artifact format notes

- LightGBM saved as native text (`booster_.save_model`, truncated to
  `best_iteration_`) — verified bit-identical predictions vs pickles.
- CatBoost saved as native `.cbm` — bit-identical, 5× smaller than pickles.
- KNN context as `.npz` plain arrays; scaler mean/scale as JSON.
- Scaler must be applied in **float32** to reproduce training arithmetic:
  float64 transform moved a few rows across tree split thresholds
  (max submission diff 0.15 ft).

## Open ideas (not yet tried)

- Discrete dip-offset classifier (Tucker-style, K≈10 classes) + cumsum,
  blended with the regression.
- Per-well b_well datum correction of the F cloud before plane fitting.
- Hard-well specialist model (train on worst-decile profile) or
  uncertainty-weighted blend with sp45/fleongg.
- Blend weight optimization of this model against sp45-proj + fleongg
  (expect ~0.1–0.2 weight given OOF 11.08 vs their LB ~7.5–7.9).
