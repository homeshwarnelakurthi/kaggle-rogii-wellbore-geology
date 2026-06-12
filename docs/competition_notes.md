# ROGII competition — domain notes

Key insights about the data and the geology, established empirically on the
773 training wells. Read this before touching the pipeline.

## The governing identity

```
TVT = -Z + F(x,y) + b_well
```

- `F(x,y)` is a smooth geological surface (effective formation depth).
- `b_well` is constant within a well (std ~0.01 ft). Every training well row
  is therefore an exact sample of `F` (up to datum): `F + b = TVT + Z`.
- Consequence: the competition is really *surface interpolation plus
  along-well extrapolation*, not log correlation. GR is a secondary,
  fine-tuning signal.

## Data invariants

- MD step is uniformly 1 ft in every well (train and test).
- The eval zone (TVT_input = NaN) is always one contiguous block at the
  **end** of the well — pure forward extrapolation, never interpolation.
- Eval zone averages 73% of the well (~4,895 rows); eval TVT range averages
  only 29.4 ft (wells are near-horizontal).
- Submission id = `{well_id}_{row_index}` into the horizontal well CSV.
- Formation columns (ANCC…BUDA) and typewell `Geology` exist only in train.
- The public `test/` folder is 3 truncated copies of train wells — local
  test scores are meaningless; trust grouped OOF.

## GR is sparse and the error is concentrated

- GR missing: ~24% of known-zone rows, ~32% of eval-zone rows; 188/773 wells
  miss >50% of eval-zone GR.
- The worst decile of wells carries ~51% of total squared error. Profile:
  long eval zones, distant offset wells, high GR missingness. Any future
  gain has to come from these wells.
- Dip magnitude correlates r ≈ +0.52 with per-well RMSE.

## Stratigraphic-domain lessons (hard-won)

1. **MD-domain GR window matching does not work.** In the lateral, 51 MD-ft
   spans ~1 TVT-ft; in the build section the same window spans ~40 TVT-ft.
   Cross-correlating raw MD windows compares incommensurate stratigraphic
   intervals — measured to be pure noise (MAE 55 ft even at high
   self-confidence).
2. **GR character repeats across cycles.** Any GR matching must be
   band-constrained around a structural prior, or the argmax snaps to the
   wrong stratigraphic cycle (this is what destroyed the Z-cumsum
   experiments, LB 60–120).
3. **The working recipe is the geosteering workflow**: build structure from
   offset wells first (local plane fit of F, path-integrated gradient), then
   fine-tune with GR(TVT) curve alignment in a tight ±15 ft band. Order
   matters; reversing it fails.
4. **The official hint (PPTX slide 9)** — "use the horizontal well's own
   pre-PS GR as reference" — is right about the instrument (same tool, same
   borehole, exact TVT labels) but only works after resampling to the TVT
   domain, for the scale reason in (1).

## Surface interpolation lessons

- IDW-KNN of F oversmooths: MAE 14 ft even with neighbors <500 ft. A local
  weighted plane fit (gradient + intercept) at the query point cuts that to
  ~4–5 ft on neighbor-dense wells.
- Path-integrating the local gradient along the lateral (`plane_dF_path`)
  beats evaluating the plane directly — but it **diverges** on
  sparse-neighbor wells (up to 927 ft error). It must be paired with
  trust features (`knn_dist_mean`, `plane_fit_rms`) so the model can gate it.
- Leave-own-well-out for train-well features cannot be done by masking a
  fixed k-NN result: the well's own points dominate any neighbor list.
  Rebuild the KD-tree without the well's points.

## Modeling lessons

- Detrend the target: learn `dF = TVT − last_tvt + (Z − Z_anchor)`, add the
  exact ramp back at reconstruction. Equivalent to the
  `cumsum(-dZ - offset·dMD)` formulation used by top public solutions.
- GroupKFold by well is mandatory: rows within a well are extremely
  correlated; ungrouped CV is fantasy (learn/test gap 7 vs 12 in CatBoost).
- CatBoost adds nothing over LightGBM here (optimal blend weight 0.0).
- Per-well smoothing of predictions: ≤0.04 RMSE. Not the lever.
- Scale features in float32 at inference to match training arithmetic;
  float64 moves rows across split thresholds (~0.15 ft prediction shifts).

## Reference points

- Leaderboard (as of June 2026): #1 5.986, #2 5.992, #3 6.487;
  our blend 7.535; this model OOF 11.08.
- #2 (Tucker Arrants) approach: `TVT = last_tvt + cumsum(-dZ - offset·dMD)`
  with discrete GR-classified offset (K=10) — discrete picker, not
  continuous search.
- fle3n v5 (LB 7.528): 128-seed PF + beam (engine A) ⊕ GBDT on rich
  features (engine B) + gated exact-match recovery.
