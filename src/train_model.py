"""Train LightGBM + CatBoost on detrended residuals with GroupKFold(5).

Target: target_dF = TVT - (last_tvt - (Z - Z_anchor))
Reconstruction: TVT_pred = tvt_flat + dF_pred
OOF metric: RMSE on absolute TVT.

Outputs in models/ (all pickle-free):
  lgb_fold{0..4}[_s1].txt (native LightGBM), cb_fold{0..4}.cbm (native
  CatBoost), features.json (feature list + scaler mean/scale + blend weight),
  cv_rmse.txt
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from catboost import CatBoostRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURE_COLS

LGB_PARAMS = dict(
    objective="regression", metric="rmse", num_leaves=255,
    learning_rate=0.02, n_estimators=8000, min_child_samples=400,
    subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
    reg_alpha=1.0, reg_lambda=10.0, n_jobs=-1, verbose=-1,
)
LGB_SEEDS = [0, 2027]
CB_PARAMS = dict(
    loss_function="RMSE", iterations=1000, learning_rate=0.05,
    depth=8, l2_leaf_reg=3.0, random_seed=0, verbose=200,
    allow_writing_files=False,
)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    df = pd.read_parquet("features_train.parquet")
    print(f"rows {len(df):,}  wells {df.well.nunique()}")

    X = df[FEATURE_COLS].astype(np.float32)
    y = df["target_dF"].values.astype(np.float32)
    groups = df["well"].values
    tvt_true = df["tvt_true"].values
    tvt_flat = df["tvt_flat"].values

    print(f"flat-physics baseline RMSE: {rmse(tvt_true, tvt_flat):.4f}")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    os.makedirs("models", exist_ok=True)
    gkf = GroupKFold(n_splits=5)
    oof_lgb = np.full(len(df), np.nan)
    oof_cb = np.full(len(df), np.nan)
    fold_of = np.full(len(df), -1, dtype=np.int8)

    for fold, (tr, va) in enumerate(gkf.split(Xs, y, groups)):
        t0 = time.time()
        fold_of[va] = fold
        Xtr, ytr, Xva, yva = Xs[tr], y[tr], Xs[va], y[va]

        seed_preds = []
        for si, seed in enumerate(LGB_SEEDS):
            m = lgb.LGBMRegressor(**LGB_PARAMS, random_state=seed)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
            seed_preds.append(m.predict(Xva))
            suffix = "" if si == 0 else f"_s{si}"
            # native text format: portable across numpy/lightgbm versions,
            # truncated to the early-stopped best iteration
            m.booster_.save_model(f"models/lgb_fold{fold}{suffix}.txt",
                                  num_iteration=m.best_iteration_)
        oof_lgb[va] = np.mean(seed_preds, axis=0)
        t_lgb = time.time() - t0

        cb = CatBoostRegressor(**CB_PARAMS)
        cb.fit(Xtr, ytr, eval_set=(Xva, yva),
               early_stopping_rounds=50, use_best_model=True)
        oof_cb[va] = cb.predict(Xva)
        cb.save_model(f"models/cb_fold{fold}.cbm")

        tl = tvt_flat[va] + oof_lgb[va]
        tc = tvt_flat[va] + oof_cb[va]
        te = tvt_flat[va] + 0.5 * (oof_lgb[va] + oof_cb[va])
        print(f"fold {fold}: lgb_iters={m.best_iteration_} "
              f"TVT rmse lgb={rmse(tvt_true[va], tl):.4f} "
              f"cb={rmse(tvt_true[va], tc):.4f} ens={rmse(tvt_true[va], te):.4f} "
              f"(lgb {t_lgb:.0f}s, total {time.time()-t0:.0f}s)", flush=True)

    oof_ens = 0.5 * (oof_lgb + oof_cb)
    r_lgb = rmse(tvt_true, tvt_flat + oof_lgb)
    r_cb = rmse(tvt_true, tvt_flat + oof_cb)
    r_ens = rmse(tvt_true, tvt_flat + oof_ens)
    r_flat = rmse(tvt_true, tvt_flat)
    print(f"\nOOF TVT RMSE  lgb={r_lgb:.4f}  cb={r_cb:.4f}  ens={r_ens:.4f}  "
          f"(flat baseline {r_flat:.4f})")

    # optimal blend weight on OOF
    best_w, best_r = 0.5, r_ens
    for w in np.arange(0.0, 1.01, 0.05):
        r = rmse(tvt_true, tvt_flat + w * oof_lgb + (1 - w) * oof_cb)
        if r < best_r:
            best_w, best_r = float(w), r
    print(f"best blend: {best_w:.2f}*lgb + {1-best_w:.2f}*cb -> {best_r:.4f}")

    with open("models/features.json", "w") as fh:
        json.dump({"features": FEATURE_COLS, "target": "target_dF",
                   "reconstruction": "tvt = tvt_flat + pred = last_tvt - (Z - Z_anchor) + pred",
                   "blend_w_lgb": best_w,
                   "scaler_mean": scaler.mean_.tolist(),
                   "scaler_scale": scaler.scale_.tolist()}, fh, indent=2)
    with open("models/cv_rmse.txt", "w") as fh:
        fh.write(f"flat_baseline_rmse {r_flat:.4f}\n"
                 f"oof_tvt_rmse_lgb {r_lgb:.4f}\noof_tvt_rmse_cb {r_cb:.4f}\n"
                 f"oof_tvt_rmse_ens50 {r_ens:.4f}\n"
                 f"oof_tvt_rmse_best_blend {best_r:.4f} (w_lgb={best_w:.2f})\n")

    # feature importance summary
    imp = np.mean([lgb.Booster(model_file=f"models/lgb_fold{f}.txt")
                   .feature_importance() for f in range(5)], axis=0)
    order = np.argsort(imp)[::-1]
    print("\nTop 20 LGB features (gain split count):")
    for i in order[:20]:
        print(f"  {FEATURE_COLS[i]:28s} {imp[i]:.0f}")

    # per-well RMSE distribution
    oof = pd.DataFrame({"well": df["well"],
                        "err2": (tvt_true - (tvt_flat + best_w * oof_lgb
                                             + (1 - best_w) * oof_cb)) ** 2})
    pw = oof.groupby("well")["err2"].mean().pow(0.5).sort_values()
    print(f"\nper-well RMSE: median={pw.median():.2f} p90={pw.quantile(0.9):.2f} "
          f"max={pw.max():.2f}")
    print("done")


if __name__ == "__main__":
    main()
