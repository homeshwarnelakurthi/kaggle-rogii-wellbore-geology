"""TASK C — Level-2 soft router between LGB and Tucker.

Target (per row): 1 if err_tucker < err_lgb else 0.
Router features: plane_fit_rms, |anchor_b|, knn_dist, gr_missing_rate,
                 model_disagreement = |tvt_lgb - tvt_tucker|  (per-row, key).
Shallow LGB (max_depth=3, num_leaves=7), GroupKFold(5) by well -> OOF tucker_prob.
Soft route: tvt = tucker_prob*tvt_tucker + (1-tucker_prob)*tvt_lgb.
Leakage-safe: router trained only on wells outside each val fold; its inputs
(tvt_lgb OOF, tvt_tucker per-well) are themselves out-of-fold.
"""
import os, sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gp_kriging as gk
from tail_rescue import anchor_b_per_well

ROOT = gk.ROOT


def rmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)))


def main():
    t0 = time.time()
    oof = pd.read_parquet(os.path.join(ROOT, "models", "oof_components.parquet"))
    feat = pd.read_parquet(os.path.join(ROOT, "features_train.parquet"),
                           columns=["well", "row_idx", "plane_fit_rms",
                                    "knn_dist_mean", "gr_missing_frac_well"])
    df = oof.merge(feat, on=["well", "row_idx"], how="left")
    df = df[np.isfinite(df.tvt_true)].reset_index(drop=True)
    ab = anchor_b_per_well()
    df["anchor_b"] = df.well.map(ab).abs()
    df["model_disagreement"] = np.abs(df.tvt_lgb - df.tvt_tucker)
    print(f"rows {len(df):,}  ({time.time()-t0:.0f}s)")

    # target + features
    err_lgb = np.abs(df.tvt_lgb - df.tvt_true)
    err_tuck = np.abs(df.tvt_tucker - df.tvt_true)
    y = (err_tuck < err_lgb).astype(int).values
    print(f"  base rate tucker-better rows: {y.mean():.3f}")
    fcols = ["plane_fit_rms", "anchor_b", "knn_dist_mean", "gr_missing_frac_well",
             "model_disagreement"]
    X = df[fcols].to_numpy(np.float64)
    groups = df.well.values

    base = rmse(df.tvt_lgb.values, df.tvt_true.values)
    print(f"\n  base LGB pooled RMSE = {base:.4f}")

    # router OOF probabilities (GroupKFold by well -> leakage-safe)
    tucker_prob = np.full(len(df), np.nan)
    params = dict(objective="binary", max_depth=3, num_leaves=7, n_estimators=300,
                  learning_rate=0.05, min_child_samples=200, subsample=0.8,
                  colsample_bytree=0.8, verbose=-1)
    imp = np.zeros(len(fcols))
    for fold, (tr, va) in enumerate(GroupKFold(5).split(X, y, groups)):
        m = lgb.LGBMClassifier(**params)
        m.fit(X[tr], y[tr])
        tucker_prob[va] = m.predict_proba(X[va])[:, 1]
        imp += m.feature_importances_
    print(f"  router trained ({time.time()-t0:.0f}s)  prob[min/mean/max]="
          f"{tucker_prob.min():.2f}/{tucker_prob.mean():.2f}/{tucker_prob.max():.2f}")

    tl = df.tvt_lgb.values; tt = df.tvt_tucker.values; tr_true = df.tvt_true.values

    # soft route (spec)
    soft = tucker_prob * tt + (1 - tucker_prob) * tl
    r_soft = rmse(soft, tr_true)
    # hard route threshold sweep
    print("\n=== routing results ===")
    print(f"  base LGB                          {base:.4f}")
    print(f"  soft (prob blend, spec)           {r_soft:.4f}")
    best = (r_soft, "soft")
    for thr in [0.4, 0.5, 0.6, 0.7, 0.8]:
        hard = np.where(tucker_prob > thr, tt, tl)
        r = rmse(hard, tr_true)
        if r < best[0]:
            best = (r, f"hard>{thr}")
        print(f"  hard route prob>{thr:.1f}                 {r:.4f}")
    # power-sharpened soft (reduce bleed on confident-lgb rows)
    for g in [1.5, 2.0, 3.0]:
        p = tucker_prob ** g
        r = rmse(p * tt + (1 - p) * tl, tr_true)
        if r < best[0]:
            best = (r, f"soft^{g}")
        print(f"  soft prob^{g}                       {r:.4f}")

    print(f"\n  >>> best route = {best[1]} -> {best[0]:.4f}  "
          f"(improvement {base-best[0]:+.4f} vs 11.074)")
    print(f"      oracle ceiling 10.05 | target break 10.8: "
          f"{'MET' if best[0] < 10.8 else 'NOT met'}")
    print("\n  router feature importance:")
    for f, v in sorted(zip(fcols, imp), key=lambda z: -z[1]):
        print(f"    {f:24s} {int(v)}")
    print(f"\ntotal time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
