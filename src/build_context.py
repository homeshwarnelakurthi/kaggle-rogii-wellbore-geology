"""Build the spatial KNN context from all 773 train wells.

Uses full train TVT labels (allowed: at Kaggle inference time the train set
is available and this pickle ships inside the model dataset).
Output: models/knn_context.npz (plain arrays, no pickle)
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import KnnContext, fit_plane, save_context

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "rogii-wellbore-geology-prediction")
SUBSAMPLE = 5

def main():
    files = sorted(glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv")))
    print(f"{len(files)} train wells")
    pts_xy, pts_F, pts_well, pts_eg = [], [], [], []
    well_ids, anchors, gxs, gys = [], [], [], []
    t0 = time.time()
    for wi, fp in enumerate(files):
        wid = os.path.basename(fp).split("__")[0]
        df = pd.read_csv(fp, usecols=["MD", "X", "Y", "Z", "TVT", "TVT_input", "EGFDU"])
        X = df["X"].values; Y = df["Y"].values; Z = df["Z"].values
        F = df["TVT"].values + Z
        eg = df["EGFDU"].values
        s = slice(None, None, SUBSAMPLE)
        pts_xy.append(np.column_stack([X[s], Y[s]]))
        pts_F.append(F[s])
        pts_well.append(np.full(len(F[s]), wi, dtype=np.int32))
        pts_eg.append(eg[s])
        # anchor = last known row (same definition as feature extraction)
        known = np.isfinite(df["TVT_input"].values)
        a = int(np.argmax(~known)) - 1
        anchors.append([X[a], Y[a]])
        gx, gy = fit_plane(X, Y, F)   # full-well plane dip
        gxs.append(gx); gys.append(gy)
        well_ids.append(wid)
        if (wi + 1) % 100 == 0:
            print(f"  {wi+1}/{len(files)}  {time.time()-t0:.0f}s")

    ctx = KnnContext(
        pts_xy=np.concatenate(pts_xy).astype(np.float64),
        pts_F=np.concatenate(pts_F).astype(np.float64),
        pts_well=np.concatenate(pts_well),
        pts_egfdu=np.concatenate(pts_eg).astype(np.float64),
        well_ids=well_ids,
        well_anchor_xy=np.array(anchors, dtype=np.float64),
        well_gx=np.array(gxs), well_gy=np.array(gys),
    )
    os.makedirs("models", exist_ok=True)
    save_context(ctx, os.path.join("models", "knn_context.npz"))
    print(f"points: {len(ctx.pts_F):,}  wells: {len(well_ids)}  "
          f"saved models/knn_context.npz  ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
