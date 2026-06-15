"""Tucker offset-classifier: LightGBM regression of offset_drift from known-zone
GR + trajectory + spatial features, then constant-dip TVT reconstruction.

offset_drift = true_eval_offset - known_dip   (target; eval-zone median dip drift)
predicted_offset = known_dip + lgb_drift_pred
TVT = last_tvt + cumsum(-dZ_ev + predicted_offset*dMD_ev)  [physics sign] + savgol

Standalone; no existing model weights used. Oracle ceiling (perfect offset) is
pooled 15.06 ft, so this design cannot reach <11 — built to report the realized
number and the offset_drift predictability.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import skew
from scipy.spatial import cKDTree
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gp_kriging as gk

SUBDIR = os.path.join(gk.ROOT, "submissions"); os.makedirs(SUBDIR, exist_ok=True)
MODDIR = os.path.join(gk.ROOT, "models")

FEATURES = ["own_dip", "own_dip_std", "gr_mean", "gr_std", "gr_skew", "gr_trend",
            "gr_vs_tw_rmse", "gr_missing_eval", "knn_dist", "known_len", "eval_len",
            "traj_dip_mean", "traj_dip_std"]


def ols_slope(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return 0.0
    x = x[m] - x[m].mean(); y = y[m] - y[m].mean()
    d = (x * x).sum()
    return float((x * y).sum() / d) if d > 0 else 0.0


def feats_and_target(w):
    ps, n = w["ps"], w["n"]
    Z, MD, GR, ti, tvt = w["Z"], w["MD"], w["GR"], w["TVT_input"], w["TVT"]
    ev = np.arange(ps, n)
    # known-zone dip (last 200)
    lo = max(1, ps - 200)
    dF = np.diff(ti[lo - 1:ps]) + np.diff(Z[lo - 1:ps]); dMDk = np.diff(MD[lo - 1:ps])
    mk = dMDk > 0.5
    dip_series = dF[mk] / dMDk[mk] if mk.any() else np.array([0.0])
    known_dip = float(np.median(dip_series)); own_dip_std = float(np.std(dip_series))
    # trajectory dip last 50
    lo2 = max(1, ps - 50)
    dz50 = np.diff(Z[lo2 - 1:ps]); dmd50 = np.diff(MD[lo2 - 1:ps]); m50 = dmd50 > 0.5
    traj = dz50[m50] / dmd50[m50] if m50.any() else np.array([0.0])
    # GR known-zone stats
    kn_gr = GR[:ps]; kn_gr_f = kn_gr[np.isfinite(kn_gr)]
    gr_mean = float(kn_gr_f.mean()) if len(kn_gr_f) > 5 else np.nan
    gr_std = float(kn_gr_f.std()) if len(kn_gr_f) > 5 else np.nan
    gr_skew = float(skew(kn_gr_f)) if len(kn_gr_f) > 5 else 0.0
    gr_trend = ols_slope(MD[lo:ps], GR[lo:ps])
    # GR vs typewell rmse at known TVT
    tw_t, tw_g = w["tw_tvt"], w["tw_gr"]; tm = np.isfinite(tw_t) & np.isfinite(tw_g)
    if tm.sum() > 10:
        o = np.argsort(tw_t[tm])
        look = np.interp(ti[:ps], tw_t[tm][o], tw_g[tm][o], left=np.nan, right=np.nan)
        r = kn_gr - look; rf = r[np.isfinite(r)]
        gr_vs_tw_rmse = float(np.sqrt(np.mean(rf ** 2))) if len(rf) > 5 else np.nan
    else:
        gr_vs_tw_rmse = np.nan
    # GR missing rate in eval
    gr_missing_eval = float(np.mean(~np.isfinite(GR[ev]))) if len(ev) else 1.0

    f = dict(own_dip=known_dip, own_dip_std=own_dip_std,
             gr_mean=gr_mean, gr_std=gr_std, gr_skew=gr_skew, gr_trend=gr_trend,
             gr_vs_tw_rmse=gr_vs_tw_rmse, gr_missing_eval=gr_missing_eval,
             known_len=float(ps), eval_len=float(n - ps),
             traj_dip_mean=float(np.mean(traj)), traj_dip_std=float(np.std(traj)),
             knn_dist=np.nan)   # filled later
    f["_known_dip"] = known_dip
    # target (train only)
    if tvt is not None:
        dT = np.diff(tvt[ev]); dZ = np.diff(Z[ev]); dM = np.diff(MD[ev])
        me = (dM > 0.5) & np.isfinite(dT) & np.isfinite(dZ)
        if me.sum() >= 5:
            true_off = float(np.median((dT[me] + dZ[me]) / dM[me]))
            f["_offset_drift"] = true_off - known_dip
        else:
            f["_offset_drift"] = np.nan
    return f


def reconstruct(w, predicted_offset):
    ps, n = w["ps"], w["n"]; ev = np.arange(ps, n)
    a = ps - 1
    Zseq = np.concatenate([[w["Z"][a]], w["Z"][ev]]); MDseq = np.concatenate([[w["MD"][a]], w["MD"][ev]])
    dZ_ev = np.diff(Zseq); dMD_ev = np.diff(MDseq)
    tvt = w["TVT_input"][ps - 1] + np.cumsum(-dZ_ev + predicted_offset * dMD_ev)
    wl = min(61, len(tvt));  wl = wl - 1 if wl % 2 == 0 else wl
    if wl >= 5:
        tvt = savgol_filter(tvt, wl, 3)
    return ev, tvt


def main():
    t0 = time.time()
    ids = gk.list_wells("train")
    wells = [gk.load_well(w, "train") for w in ids]
    print(f"loaded {len(wells)} wells ({time.time()-t0:.0f}s)")

    # spatial isolation feature: mean dist to 5 nearest OTHER well rep points
    reps = np.array([[w["rep_x"], w["rep_y"]] for w in wells])
    rtree = cKDTree(reps); dd, _ = rtree.query(reps, k=6)
    knn_dist = dd[:, 1:].mean(1)

    rows = []
    for i, w in enumerate(wells):
        f = feats_and_target(w); f["knn_dist"] = float(knn_dist[i]); f["well"] = w["well"]; f["wi"] = i
        rows.append(f)
    df = pd.DataFrame(rows)
    print("\nSTEP 1 — offset_drift distribution:")
    od = df["_offset_drift"].dropna()
    print(f"  mean={od.mean():.5f} std={od.std():.5f} p5={od.quantile(.05):.4f} "
          f"p95={od.quantile(.95):.4f}  (target var)")

    valid = df["_offset_drift"].notna().values
    X = df[FEATURES].values.astype(np.float64)
    y = df["_offset_drift"].values.astype(np.float64)
    groups = df["wi"].values

    # STEP 2 — LGB GroupKFold(5) on offset_drift
    print("\nSTEP 2 — LightGBM offset_drift regression, GroupKFold(5):")
    gkf = GroupKFold(5)
    drift_oof = np.full(len(df), np.nan)
    params = dict(objective="regression", num_leaves=31, learning_rate=0.05,
                  n_estimators=500, min_child_samples=20, subsample=0.8,
                  colsample_bytree=0.8, verbose=-1)
    idx_valid = np.where(valid)[0]
    for fold, (tr, va) in enumerate(gkf.split(idx_valid, idx_valid, groups[idx_valid])):
        tri, vai = idx_valid[tr], idx_valid[va]
        m = lgb.LGBMRegressor(**params)
        m.fit(X[tri], y[tri])
        drift_oof[vai] = m.predict(X[vai])
    drift_rmse = np.sqrt(np.nanmean((drift_oof[valid] - y[valid]) ** 2))
    base_rmse = np.sqrt(np.nanmean(y[valid] ** 2))   # predict drift=0 (use known_dip)
    print(f"  offset_drift OOF RMSE = {drift_rmse:.5f}  (vs predict-zero {base_rmse:.5f}, "
          f"i.e. {'better' if drift_rmse<base_rmse else 'WORSE'})")

    # STEP 3/4 — reconstruct TVT with OOF predicted offset, score
    print("\nSTEP 3/4 — TVT reconstruction (OOF):")
    psq = psn = 0.0; perwell = []; oracle_sq = oracle_n = 0.0
    for i, w in enumerate(wells):
        if not valid[i]:
            continue
        pred_off = df["_known_dip"].iloc[i] + (drift_oof[i] if np.isfinite(drift_oof[i]) else 0.0)
        ev, tvt_pred = reconstruct(w, pred_off)
        tt = w["TVT"][ev]; m = np.isfinite(tt)
        if m.sum() < 5:
            continue
        sq = float(np.sum((tvt_pred[m] - tt[m]) ** 2)); psq += sq; psn += m.sum()
        perwell.append((w["well"], np.sqrt(sq / m.sum())))
        # oracle for reference
        oracle_off = df["_known_dip"].iloc[i] + y[i]
        _, tvt_or = reconstruct(w, oracle_off)
        oracle_sq += float(np.sum((tvt_or[m] - tt[m]) ** 2)); oracle_n += m.sum()
    pooled = np.sqrt(psq / psn); med = np.median([r[1] for r in perwell])
    pw = pd.DataFrame(perwell, columns=["well", "rmse"])
    print(f"  pooled RMSE  = {pooled:.2f} ft")
    print(f"  per-well med = {med:.2f} ft")
    print(f"  worst 10: ", [(r.well, round(r.rmse, 1)) for _, r in pw.nlargest(10, 'rmse').iterrows()])
    print(f"  (oracle ceiling for ref = {np.sqrt(oracle_sq/oracle_n):.2f} ft)")

    print("\n  comparison:")
    print(f"    naive baseline      107.50")
    print(f"    simple own_dip tuck  44.82")
    print(f"    THIS (lgb offset)    {pooled:6.2f}")
    print(f"    oracle (perfect)     15.06")
    print(f"    existing LGB         11.08")
    print(f"    target               <11")

    # feature importance (refit on all valid)
    mfull = lgb.LGBMRegressor(**params); mfull.fit(X[valid], y[valid])
    imp = sorted(zip(FEATURES, mfull.feature_importances_), key=lambda z: -z[1])
    print("\n  top features:", [(f, int(v)) for f, v in imp[:6]])

    # STEP 5 — save only if <11
    if pooled < 11:
        print("\nSTEP 5 — OOF<11, saving submission + model")
        # ... (would build test submission here)
        mfull.booster_.save_model(os.path.join(MODDIR, "tucker_lgb_offset.txt"))
        with open(os.path.join(MODDIR, "tucker_scaler.json"), "w") as fh:
            json.dump({"features": FEATURES}, fh)
    else:
        print(f"\nSTEP 5 — SKIPPED: OOF {pooled:.2f} >= 11 (per spec condition). "
              f"Capped by within-eval F curvature (oracle 15.06).")
    print(f"\ntotal time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
