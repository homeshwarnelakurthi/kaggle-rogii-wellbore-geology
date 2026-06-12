"""Predict the test wells -> submission.csv.

Identical feature path as training, except KNN uses ALL 773 train wells
(no exclusion — test wells are not in the context cloud).
"""
import glob
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_well, load_context, FEATURE_COLS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "rogii-wellbore-geology-prediction")


def main():
    ctx = load_context("models/knn_context.npz")
    meta = json.load(open("models/features.json"))
    w_lgb = meta.get("blend_w_lgb", 0.5)
    # float32 to match the training-time sklearn transform arithmetic
    sc_mean = np.array(meta["scaler_mean"], dtype=np.float32)
    sc_scale = np.array(meta["scaler_scale"], dtype=np.float32)
    lgbs = [lgb.Booster(model_file=p)
            for p in sorted(glob.glob("models/lgb_fold*.txt"))]
    cbs = []
    for p in sorted(glob.glob("models/cb_fold*.cbm")):
        m = CatBoostRegressor()
        m.load_model(p)
        cbs.append(m)
    print(f"loaded {len(lgbs)} LGB + {len(cbs)} CB models")

    files = sorted(glob.glob(os.path.join(DATA, "test", "*__horizontal_well.csv")))
    print(f"{len(files)} test wells, blend w_lgb={w_lgb:.2f}")
    parts = []
    t0 = time.time()
    for fp in files:
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
        print(f"  {wid}: {len(feat)} rows")
    sub = pd.concat(parts, ignore_index=True)

    sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    sub = sample[["id"]].merge(sub, on="id", how="left")
    assert sub["tvt"].notna().all(), "missing predictions for some ids"
    sub.to_csv("submission.csv", index=False)
    print(f"submission.csv: {len(sub)} rows  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
