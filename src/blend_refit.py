"""TASK A — blend re-fit on OOF.

Regenerate LGB OOF (predict each GroupKFold val fold with its saved fold model),
build align2_self and Tucker per-row predictions on the same rows, grid-search
Final = w1*lgb + w2*align2_self + w3*tucker on pooled OOF TVT RMSE.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURE_COLS
import gp_kriging as gk

ROOT = gk.ROOT


def rmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)))


def tucker_per_well():
    """Per-well Tucker (own_dip cumsum, physics sign), keyed by (well,row_idx)."""
    recs = []
    for wid in gk.list_wells("train"):
        w = gk.load_well(wid, "train")
        ps, n = w["ps"], w["n"]; ev = np.arange(ps, n)
        Z, MD, ti = w["Z"], w["MD"], w["TVT_input"]
        lo = max(1, ps - 200)
        dF = np.diff(ti[lo - 1:ps]) + np.diff(Z[lo - 1:ps]); dM = np.diff(MD[lo - 1:ps]); mk = dM > 0.5
        own_dip = np.median(dF[mk] / dM[mk]) if mk.any() else 0.0
        a = ps - 1
        dZ = np.diff(np.concatenate([[Z[a]], Z[ev]])); dMD = np.diff(np.concatenate([[MD[a]], MD[ev]]))
        tuck = ti[a] + np.cumsum(-dZ + own_dip * dMD)
        recs.append(pd.DataFrame({"well": wid, "row_idx": ev, "tucker": tuck}))
    return pd.concat(recs, ignore_index=True)


def main():
    t0 = time.time()
    jcfg = json.load(open(os.path.join(ROOT, "models", "features.json")))
    mean = np.array(jcfg["scaler_mean"], dtype=np.float32)
    scale = np.array(jcfg["scaler_scale"], dtype=np.float32)

    cols = list(dict.fromkeys(FEATURE_COLS + ["well", "row_idx", "tvt_flat",
                                              "tvt_true", "align2_own_est", "align2_tw_est"]))
    df = pd.read_parquet(os.path.join(ROOT, "features_train.parquet"), columns=cols)
    print(f"loaded parquet {len(df):,} rows ({time.time()-t0:.0f}s)")

    X = df[FEATURE_COLS].to_numpy(np.float32)
    Xs = ((X - mean) / scale).astype(np.float32)       # float32 to match training
    groups = df["well"].values
    y = df["target_dF"].values if "target_dF" in df.columns else None

    # regenerate LGB OOF with the SAME GroupKFold(5) split
    oof_lgb = np.full(len(df), np.nan, dtype=np.float64)
    gkf = GroupKFold(5)
    for fold, (tr, va) in enumerate(gkf.split(Xs, groups, groups)):
        preds = []
        for suf in ["", "_s1"]:
            b = lgb.Booster(model_file=os.path.join(ROOT, "models", f"lgb_fold{fold}{suf}.txt"))
            preds.append(b.predict(Xs[va]))
        oof_lgb[va] = np.mean(preds, axis=0)
    print(f"LGB OOF regenerated ({time.time()-t0:.0f}s)")

    tvt_flat = df["tvt_flat"].values
    tvt_true = df["tvt_true"].values
    fin = np.isfinite(tvt_true)

    tvt_lgb = tvt_flat + oof_lgb
    tvt_align2 = tvt_flat + df["align2_own_est"].values
    # merge tucker
    tk = tucker_per_well()
    df2 = df[["well", "row_idx"]].merge(tk, on=["well", "row_idx"], how="left")
    tvt_tucker = df2["tucker"].values
    # fallback for any unmatched / nan
    tvt_align2 = np.where(np.isfinite(tvt_align2), tvt_align2, tvt_flat)
    tvt_tucker = np.where(np.isfinite(tvt_tucker), tvt_tucker, tvt_flat)

    print("\n=== component OOF pooled RMSE (TVT) ===")
    print(f"  LGB (regenerated) : {rmse(tvt_lgb[fin], tvt_true[fin]):.4f}   (target 11.0751)")
    print(f"  align2_self       : {rmse(tvt_align2[fin], tvt_true[fin]):.4f}")
    print(f"  tucker            : {rmse(tvt_tucker[fin], tvt_true[fin]):.4f}")
    print(f"  flat baseline     : {rmse(tvt_flat[fin], tvt_true[fin]):.4f}")

    # correlation of residuals (does any component decorrelate from LGB?)
    rl = tvt_lgb[fin] - tvt_true[fin]
    ra = tvt_align2[fin] - tvt_true[fin]
    rt = tvt_tucker[fin] - tvt_true[fin]
    print(f"\n  resid corr  lgb~align2={np.corrcoef(rl,ra)[0,1]:.3f}  "
          f"lgb~tucker={np.corrcoef(rl,rt)[0,1]:.3f}")

    print("\n=== TASK A: grid search Final = w1*lgb + w2*align2 + w3*tucker ===")
    best = (1e9, None)
    lf, af, tf = tvt_lgb[fin], tvt_align2[fin], tvt_tucker[fin]; tt = tvt_true[fin]
    results = []
    for w1 in [0.6, 0.7, 0.8, 0.9, 1.0]:
        for w2 in [0.0, 0.1, 0.2, 0.3]:
            w3 = 1.0 - w1 - w2
            blend = w1 * lf + w2 * af + w3 * tf
            r = rmse(blend, tt)
            results.append((w1, w2, w3, r))
            if r < best[0]:
                best = (r, (w1, w2, w3))
    # show a few around the best
    results.sort(key=lambda z: z[3])
    print("  top 8 weight combos:")
    for w1, w2, w3, r in results[:8]:
        print(f"    w_lgb={w1:.1f} w_align2={w2:.1f} w_tucker={w3:+.1f} -> {r:.4f}")
    print(f"\n  >>> best blend {best[1]} -> pooled OOF {best[0]:.4f}  "
          f"(target <10.5: {'MET' if best[0] < 10.5 else 'NOT met'})")

    # unconstrained least-squares blend (reference upper bound on linear blend)
    A = np.column_stack([lf, af, tf])
    coef, *_ = np.linalg.lstsq(A, tt, rcond=None)
    r_ls = rmse(A @ coef, tt)
    print(f"  (unconstrained LS blend weights {np.round(coef,3)} -> {r_ls:.4f})")

    # save oof components for Task B
    out = df[["well", "row_idx", "tvt_flat", "tvt_true"]].copy()
    out["tvt_lgb"] = tvt_lgb; out["tvt_align2"] = tvt_align2; out["tvt_tucker"] = tvt_tucker
    out.to_parquet(os.path.join(ROOT, "models", "oof_components.parquet"))
    print(f"\n  saved models/oof_components.parquet  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
