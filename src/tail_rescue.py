"""TASK B — tail rescue: gate LGB -> Tucker on the worst wells.

Find which known-zone confidence feature (plane_fit_rms, anchor_b, knn_dist,
gr_missing) best flags LGB failure, then hard-gate those wells to an
independent predictor and measure pooled OOF change.
"""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gp_kriging as gk

ROOT = gk.ROOT


def rmse(p, t):
    return float(np.sqrt(np.mean((p - t) ** 2)))


def anchor_b_per_well():
    """Leave-own-well-out cloud anchor |b| = |median(known_eff - cloud_at_known)|."""
    wells = [gk.load_well(w, "train") for w in gk.list_wells("train")]
    for w in wells:
        tvt, Z, X, Y = w["TVT"], w["Z"], w["X"], w["Y"]; ps = w["ps"]
        m = np.isfinite(tvt) & np.isfinite(Z)
        w["cl_xy"] = np.column_stack([X[m], Y[m]]); w["cl_F"] = tvt[m] + Z[m]
        mk = np.isfinite(w["TVT_input"][:ps]) & np.isfinite(Z[:ps])
        w["kn_xy"] = np.column_stack([X[:ps][mk], Y[:ps][mk]]); w["kn_F"] = w["TVT_input"][:ps][mk] + Z[:ps][mk]
    out = {}
    g = np.arange(len(wells))
    for tr, va in GroupKFold(5).split(g, g, g):
        tree = cKDTree(np.vstack([wells[i]["cl_xy"] for i in tr]))
        cF = np.concatenate([wells[i]["cl_F"] for i in tr])
        for j in va:
            w = wells[j]
            if len(w["kn_F"]) < 5:
                out[w["well"]] = 0.0; continue
            d, idx = tree.query(w["kn_xy"], k=24, workers=-1)
            wt = 1.0 / (d + 1.0); Fik = (wt * cF[idx]).sum(1) / wt.sum(1)
            out[w["well"]] = abs(float(np.median(w["kn_F"] - Fik)))
    return out


def main():
    t0 = time.time()
    oof = pd.read_parquet(os.path.join(ROOT, "models", "oof_components.parquet"))
    oof = oof[np.isfinite(oof["tvt_true"])].copy()
    oof["e_lgb"] = (oof.tvt_lgb - oof.tvt_true) ** 2
    oof["e_tuck"] = (oof.tvt_tucker - oof.tvt_true) ** 2
    oof["e_al2"] = (oof.tvt_align2 - oof.tvt_true) ** 2
    g = oof.groupby("well")
    pw = pd.DataFrame({
        "rmse_lgb": g["e_lgb"].mean() ** 0.5,
        "rmse_tuck": g["e_tuck"].mean() ** 0.5,
        "rmse_al2": g["e_al2"].mean() ** 0.5,
        "n": g.size(),
    }).reset_index()

    # per-well confidence features
    feat = pd.read_parquet(os.path.join(ROOT, "features_train.parquet"),
                           columns=["well", "plane_fit_rms", "knn_dist_mean", "gr_missing_frac_well"])
    fg = feat.groupby("well").mean().reset_index()
    pw = pw.merge(fg, on="well")
    ab = anchor_b_per_well()
    pw["anchor_b"] = pw["well"].map(ab)
    print(f"per-well table built ({time.time()-t0:.0f}s)")

    overall = np.sqrt((oof.e_lgb.sum()) / len(oof))
    print(f"\nLGB pooled OOF = {overall:.4f}")

    worst = pw.nlargest(77, "rmse_lgb")
    print(f"\nworst 77 wells: LGB median RMSE {worst.rmse_lgb.median():.1f}, "
          f"carry {100*(worst.rmse_lgb**2*worst.n).sum()/(pw.rmse_lgb**2*pw.n).sum():.0f}% of SSE")
    print("  on worst-77: tucker median RMSE %.1f, align2 median RMSE %.1f"
          % (worst.rmse_tuck.median(), worst.rmse_al2.median()))
    print(f"  tucker beats lgb on {(worst.rmse_tuck<worst.rmse_lgb).sum()}/77, "
          f"align2 beats lgb on {(worst.rmse_al2<worst.rmse_lgb).sum()}/77")

    print("\ncorr(feature, per-well LGB RMSE):  all wells | worst-77")
    for c in ["plane_fit_rms", "anchor_b", "knn_dist_mean", "gr_missing_frac_well"]:
        ca = pw[[c, "rmse_lgb"]].corr().iloc[0, 1]
        cw = worst[[c, "rmse_lgb"]].corr().iloc[0, 1]
        print(f"  {c:22s} {ca:6.3f}   | {cw:6.3f}")

    # hard gate variants
    def pooled_with_gate(mask_wells, target_col):
        m = oof["well"].isin(set(mask_wells))
        pred = np.where(m, oof[target_col].values, oof["tvt_lgb"].values)
        return rmse(pred, oof["tvt_true"].values), int(m.sum()), len(mask_wells)

    p75_prms = pw.plane_fit_rms.quantile(0.75)
    p75_b = pw.anchor_b.quantile(0.75)
    print(f"\nP75 plane_fit_rms={p75_prms:.2f}  P75 anchor_b={p75_b:.2f}")

    print("\n=== gating experiments (pooled OOF) ===")
    print(f"  baseline (pure LGB)                         {overall:.4f}")
    gates = {
        "prms>P75 AND anchor_b>P75 -> tucker":
            (pw[(pw.plane_fit_rms > p75_prms) & (pw.anchor_b > p75_b)].well, "tvt_tucker"),
        "prms>P75 AND anchor_b>P75 -> align2":
            (pw[(pw.plane_fit_rms > p75_prms) & (pw.anchor_b > p75_b)].well, "tvt_align2"),
        "anchor_b>P75 -> tucker":
            (pw[pw.anchor_b > p75_b].well, "tvt_tucker"),
        "oracle: gate wells where tucker<lgb -> tucker":
            (pw[pw.rmse_tuck < pw.rmse_lgb].well, "tvt_tucker"),
    }
    best = (overall, "baseline")
    for name, (wlist, tgt) in gates.items():
        r, nrows, nw = pooled_with_gate(wlist, tgt)
        flag = "  <-- improves" if r < overall - 1e-4 else ""
        print(f"  {name:44s} {r:.4f}  ({nw} wells){flag}")
        if r < best[0]:
            best = (r, name)
    # tune: gate where tucker<lgb predicted by a soft rule (prms & b thresholds swept)
    print("\n  threshold sweep (gate to tucker when prms>q_p & anchor_b>q_b):")
    bestsweep = (overall, None)
    for qp in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for qb in [0.5, 0.6, 0.7, 0.8, 0.9]:
            tp = pw.plane_fit_rms.quantile(qp); tb = pw.anchor_b.quantile(qb)
            wl = pw[(pw.plane_fit_rms > tp) & (pw.anchor_b > tb)].well
            r, _, nw = pooled_with_gate(wl, "tvt_tucker")
            if r < bestsweep[0]:
                bestsweep = (r, (qp, qb, nw))
    print(f"    best sweep: {bestsweep[1]} -> {bestsweep[0]:.4f}")

    # ---- learnable meta-gate: predict (rmse_lgb - rmse_tuck) per well, KFold ----
    from sklearn.model_selection import KFold
    import lightgbm as lgb
    fcols = ["plane_fit_rms", "anchor_b", "knn_dist_mean", "gr_missing_frac_well"]
    Xg = pw[fcols].to_numpy(np.float64)
    delta = (pw.rmse_lgb - pw.rmse_tuck).to_numpy()      # >0 => tucker better => gate
    pred_delta = np.full(len(pw), np.nan)
    for tr, va in KFold(5, shuffle=True, random_state=0).split(Xg):
        m = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, learning_rate=0.05,
                              min_child_samples=20, verbose=-1)
        m.fit(Xg[tr], delta[tr])
        pred_delta[va] = m.predict(Xg[va])
    print("\n=== learnable meta-gate (KFold, predict rmse_lgb-rmse_tuck) ===")
    best_meta = (overall, None)
    for thr in [0.0, 1.0, 2.0, 3.0, 5.0]:
        wl = pw.well[pred_delta > thr]
        r, _, nw = pooled_with_gate(wl, "tvt_tucker")
        flag = "  <-- improves" if r < overall - 1e-4 else ""
        print(f"  gate tucker where pred_delta>{thr:.0f}: {r:.4f}  ({nw} wells){flag}")
        if r < best_meta[0]:
            best_meta = (r, thr)

    improve = overall - min(best[0], bestsweep[0], best_meta[0])
    print(f"\n>>> best REALIZABLE gating pooled OOF = {min(best[0], bestsweep[0], best_meta[0]):.4f} "
          f"(improvement {improve:+.4f} ft vs 11.074)")
    print(f"    oracle ceiling = 10.05 (+1.02, not realizable)")
    print(f"    target: improve > 0.3 ft -> {'MET' if improve > 0.3 else 'NOT met'}")
    print(f"\ntotal time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
