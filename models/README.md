# Trained model weights

The trained weights are committed in this folder (everything is pickle-free
and each file is well under GitHub's size limits). They are also published
as the Kaggle dataset for notebook attachment:

**[`homeshwarrao/rogii-lgb-weights-v1`](https://www.kaggle.com/datasets/homeshwarrao/rogii-lgb-weights-v1)**

## Contents

| file | format | loader |
|---|---|---|
| `lgb_fold{0..4}.txt`, `lgb_fold{0..4}_s1.txt` | native LightGBM text (10 models: 2 seeds × 5 folds, truncated to early-stopped best iteration) | `lgb.Booster(model_file=path)` |
| `cb_fold{0..4}.cbm` | native CatBoost binary | `CatBoostRegressor().load_model(path)` |
| `knn_context.npz` | plain numpy arrays — 1M-pt spatial cloud of F = TVT + Z from the 773 train wells, per-well anchors and plane dips | `features.load_context(path)` |
| `features.json` | feature list (73), scaler mean/scale arrays, blend weight, reconstruction formula | `json.load` |
| `cv_rmse.txt` | OOF scores of this training run | text |

Everything is pickle-free: no library version pinning is required to load.

## Usage

Attach the dataset (plus the competition data) to a Kaggle notebook and run
[`kaggle_inference.py`](../kaggle_inference.py) from the repo root. The
dataset also contains a copy of `src/features.py`, which the notebook imports
for feature extraction. `knn_context.npz` is required at inference: test-well
features query the train-well cloud.

Apply the scaler in **float32** — `(X.astype(np.float32) - mean32) / scale32`
— to reproduce the training-time arithmetic exactly.

## Reconstruction

Models predict the detrended residual `dF`. Final prediction:

```
TVT = last_tvt - (Z - Z_anchor) + dF_pred
```

OOF TVT RMSE of this set: **11.08** (LightGBM, GroupKFold(5) by well).
