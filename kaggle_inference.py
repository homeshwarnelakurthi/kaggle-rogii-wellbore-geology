"""Kaggle inference notebook for ROGII Wellbore Geology Prediction.
Paste into a Kaggle notebook (or upload as a script notebook).

Setup on Kaggle:
  1. Create a Kaggle dataset from the local `models/` folder PLUS
     `src/features.py`, e.g. dataset slug: <user>/rogii-lgb-weights
     Contents: lgb_fold*.txt (native LightGBM), cb_fold*.cbm (native
               CatBoost), knn_context.npz, features.json, features.py
  2. Attach that dataset + the competition data to the notebook.
  3. Run. Inference only — no training. ~5 min for ~200 wells.

All artifacts are pickle-free (native model formats, npz arrays, JSON),
so no library version pinning is required for loading.
"""
import glob
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

WEIGHTS = "/kaggle/input/rogii-lgb-weights"          # <- your dataset slug
COMP = "/kaggle/input/rogii-wellbore-geology-prediction"

sys.path.insert(0, WEIGHTS)                          # features.py lives here
from features import extract_well, load_context, FEATURE_COLS

ctx = load_context(f"{WEIGHTS}/knn_context.npz")
meta = json.load(open(f"{WEIGHTS}/features.json"))
w_lgb = meta.get("blend_w_lgb", 0.5)
# float32 to match the training-time sklearn transform arithmetic
sc_mean = np.array(meta["scaler_mean"], dtype=np.float32)
sc_scale = np.array(meta["scaler_scale"], dtype=np.float32)
lgbs = [lgb.Booster(model_file=p)
        for p in sorted(glob.glob(f"{WEIGHTS}/lgb_fold*.txt"))]
cbs = []
for p in sorted(glob.glob(f"{WEIGHTS}/cb_fold*.cbm")):
    m = CatBoostRegressor()
    m.load_model(p)
    cbs.append(m)

parts = []
for fp in sorted(glob.glob(f"{COMP}/test/*__horizontal_well.csv")):
    wid = os.path.basename(fp).split("__")[0]
    h = pd.read_csv(fp)
    tw = pd.read_csv(fp.replace("__horizontal_well", "__typewell"))
    feat = extract_well(h, tw, wid, ctx=ctx, exclude_well_idx=None,
                        is_train=False)
    Xs = (feat[FEATURE_COLS].values.astype(np.float32) - sc_mean) / sc_scale
    p_lgb = np.mean([m.predict(Xs) for m in lgbs], axis=0)
    p_cb = np.mean([m.predict(Xs) for m in cbs], axis=0)
    dF = w_lgb * p_lgb + (1 - w_lgb) * p_cb
    parts.append(pd.DataFrame({"id": feat["id"],
                               "tvt": feat["tvt_flat"].values + dF}))

sub = pd.concat(parts, ignore_index=True)
sample = pd.read_csv(f"{COMP}/sample_submission.csv")
sub = sample[["id"]].merge(sub, on="id", how="left")
assert sub["tvt"].notna().all()
sub.to_csv("submission.csv", index=False)
print("submission.csv:", len(sub), "rows")
