"""Extract eval-zone features for all train wells -> features_train.parquet.

KNN features use leave-own-well-out so OOF stays honest while matching
test-time conditions (test wells query all 773 train wells).
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import extract_well, load_context

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "rogii-wellbore-geology-prediction")

def main():
    ctx = load_context(os.path.join("models", "knn_context.npz"))
    wid_to_idx = {w: i for i, w in enumerate(ctx.well_ids)}
    files = sorted(glob.glob(os.path.join(DATA, "train", "*__horizontal_well.csv")))
    parts = []
    t0 = time.time()
    for k, fp in enumerate(files):
        wid = os.path.basename(fp).split("__")[0]
        h = pd.read_csv(fp)
        tw = pd.read_csv(fp.replace("__horizontal_well", "__typewell"))
        parts.append(extract_well(h, tw, wid, ctx=ctx,
                                  exclude_well_idx=wid_to_idx[wid],
                                  is_train=True))
        if (k + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {k+1}/{len(files)}  {el:.0f}s  eta {el/(k+1)*(len(files)-k-1):.0f}s",
                  flush=True)
    df = pd.concat(parts, ignore_index=True)
    print(f"rows: {len(df):,}  cols: {len(df.columns)}")
    df.to_parquet("features_train.parquet", index=False)
    print(f"saved features_train.parquet  ({time.time()-t0:.0f}s total)")

if __name__ == "__main__":
    main()
