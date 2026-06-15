"""Standalone Tucker cumsum predictor + confidence routing vs cloud IDW.

Tucker (own-dip extrapolation):
  own_dip = median(dF/dMD) over last 200 known rows,  dF = d(TVT+Z)
  TVT_tucker = last_tvt + cumsum(-dZ_ev + own_dip*dMD_ev)   [physics sign]
  (spec wrote '- own_dip*dMD'; both tested, physics sign reported as primary)

Routing:
  confidence = 1/(1 + |anchor_b|/100 + own_std*20)
  tvt_final  = confidence*cloud_idw + (1-confidence)*tucker

cloud_idw = leave-own-well-out IDW on the 5.09M-pt F=TVT+Z cloud, anchored by
median(known_eff - cloud_at_known).  No existing LGB/GP weights used.
"""
import os, sys, time, json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gp_kriging as gk

SUBDIR = os.path.join(gk.ROOT, "submissions")
os.makedirs(SUBDIR, exist_ok=True)


def prep(w):
    tvt, Z, X, Y, MD = w["TVT"], w["Z"], w["X"], w["Y"], w["MD"]
    ps = w["ps"]; ti = w["TVT_input"]; ev = w["ev"]
    if tvt is not None:   # train well: build cloud + eval truth
        m = np.isfinite(tvt) & np.isfinite(Z)
        w["cl_xy"] = np.column_stack([X[m], Y[m]]); w["cl_F"] = tvt[m] + Z[m]
        me = np.isfinite(tvt[ev]) & np.isfinite(Z[ev])
        w["ev_idx"] = ev[me]
        w["ev_xy"] = np.column_stack([X[ev][me], Y[ev][me]])
        w["ev_Z"] = Z[ev][me]; w["ev_t"] = tvt[ev][me]
        w["naive"] = (ti[ps - 1] + Z[ps - 1]) - w["ev_Z"]
    mk = np.isfinite(ti[:ps]) & np.isfinite(Z[:ps])
    w["kn_xy"] = np.column_stack([X[:ps][mk], Y[:ps][mk]]); w["kn_F"] = ti[:ps][mk] + Z[:ps][mk]
    w["last_tvt"] = ti[ps - 1]

    # --- own dip from last 200 known rows ---
    lo = max(1, ps - 200)
    kZ = Z[lo - 1:ps]; kMD = MD[lo - 1:ps]; kT = ti[lo - 1:ps]
    dmk = np.diff(kMD); good = dmk != 0
    dF = (np.diff(kT) + np.diff(kZ))[good] / dmk[good]
    dF = dF[np.isfinite(dF)]
    w["own_dip"] = float(np.median(dF)) if len(dF) else 0.0
    w["own_std"] = float(np.std(dF)) if len(dF) else 0.0

    # --- tucker over eval rows (use anchor-prepended diffs, then mask) ---
    a = ps - 1
    Zseq = np.concatenate([[Z[a]], Z[ev]]); MDseq = np.concatenate([[MD[a]], MD[ev]])
    dZ = np.diff(Zseq); dMD = np.diff(MDseq)
    tuck_plus = w["last_tvt"] + np.cumsum(-dZ + w["own_dip"] * dMD)   # physics sign
    tuck_minus = w["last_tvt"] + np.cumsum(-dZ - w["own_dip"] * dMD)  # spec sign
    if tvt is not None:
        w["tuck_plus"] = tuck_plus[me]; w["tuck_minus"] = tuck_minus[me]
    return w


def knn_idw(tree, F, q, k=24):
    d, idx = tree.query(q, k=k, workers=-1)
    wts = 1.0 / (d + 1.0)
    return (wts * F[idx]).sum(1) / wts.sum(1), d.mean(1)


def per_well_rmse(pred, t):
    return float(np.sqrt(np.mean((pred - t) ** 2)))


def apply_router(router, cloud, tuck, dm, absb, ostd):
    """Route between cloud and tucker per row given a router spec tuple."""
    kind = router[0]
    if kind == "hardb":
        return np.where(absb < router[1], cloud, tuck)
    if kind == "rowgate":
        _, D, T = router
        return np.where((dm < D) & (absb < T), cloud, tuck)
    if kind == "steep":
        wc = 1.0 / (1.0 + (absb / 30.0) ** router[1])
        return wc * cloud + (1 - wc) * tuck
    return tuck


def report_block(name, pw, psq, pn):
    a = np.array([pw[w] for w in pw])
    print(f"  {name:14s}: pooled {np.sqrt(psq/pn):7.2f} | median {np.median(a):6.2f} | "
          f"p90 {np.percentile(a,90):6.1f} | worst {a.max():6.0f}")


def main():
    t0 = time.time()
    ids = gk.list_wells("train")
    wells = [prep(gk.load_well(w, "train")) for w in ids]
    print(f"loaded+prepped {len(wells)} wells ({time.time()-t0:.0f}s)")

    gkf = GroupKFold(5); g = np.arange(len(wells))
    # per-row accumulators (concatenated across all OOF wells) for flexible routing
    A = {k: [] for k in ["cloud", "tuck", "naive", "t", "dm", "absb", "ostd", "well"]}
    wellrows = []

    for fold, (tr, va) in enumerate(gkf.split(g, g, g)):
        tree = cKDTree(np.vstack([wells[i]["cl_xy"] for i in tr]))
        cF = np.concatenate([wells[i]["cl_F"] for i in tr])
        for j in va:
            w = wells[j]
            if len(w["ev_t"]) < 5:
                continue
            Fi, dm = knn_idw(tree, cF, w["ev_xy"])
            Fik, _ = knn_idw(tree, cF, w["kn_xy"])
            anchor_b = float(np.median(w["kn_F"] - Fik)) if len(w["kn_F"]) >= 5 else 0.0
            cloud = Fi + anchor_b - w["ev_Z"]
            t = w["ev_t"]; n = len(t)
            A["cloud"].append(cloud); A["tuck"].append(w["tuck_plus"]); A["naive"].append(w["naive"])
            A["t"].append(t); A["dm"].append(dm)
            A["absb"].append(np.full(n, abs(anchor_b))); A["ostd"].append(np.full(n, w["own_std"]))
            A["well"].append(np.full(n, j))
            wellrows.append(dict(well=w["well"], anchor_b=anchor_b, own_dip=w["own_dip"],
                                 own_std=w["own_std"], rmse_cloud=per_well_rmse(cloud, t),
                                 rmse_tuck=per_well_rmse(w["tuck_plus"], t), n=n))
        print(f"  fold {fold} ({time.time()-t0:.0f}s)")

    for k in A:
        A[k] = np.concatenate(A[k])
    cloud, tuck, naive, t = A["cloud"], A["tuck"], A["naive"], A["t"]
    absb, ostd, dm, wellid = A["absb"], A["ostd"], A["dm"], A["well"]
    N = len(t)

    def pooled(p):
        return float(np.sqrt(np.mean((p - t) ** 2)))

    print("\n=== GroupKFold(5) OOF — pooled RMSE (per-row) ===")
    print(f"  naive            : {pooled(naive):7.2f}")
    print(f"  tuck_plus        : {pooled(tuck):7.2f}")
    print(f"  cloud            : {pooled(cloud):7.2f}")
    # spec routing (per-well confidence)
    conf_spec = 1.0 / (1.0 + absb / 100.0 + ostd * 20.0)
    conf_b = 1.0 / (1.0 + absb / 100.0)
    print(f"  routed_spec      : {pooled(conf_spec*cloud+(1-conf_spec)*tuck):7.2f}")
    print(f"  routed_b         : {pooled(conf_b*cloud+(1-conf_b)*tuck):7.2f}")

    # ---- search better routers (tucker-alone is a candidate) ----
    print("\n  hard |b|-gate sweep (cloud if |b|<T else tucker):")
    best = (pooled(tuck), ("tuck",))
    for T in [10, 20, 30, 40, 50, 75, 100, 150]:
        p = np.where(absb < T, cloud, tuck)
        r = pooled(p)
        print(f"    T={T:4d}: {r:7.2f}")
        if r < best[0]:
            best = (r, ("hardb", T))
    # per-row gate: tucker when local neighbor dist large OR |b| large
    print("\n  per-row gate (cloud if dm<D and |b|<T else tucker):")
    for D in [400, 600, 800]:
        for T in [30, 50, 75]:
            p = np.where((dm < D) & (absb < T), cloud, tuck)
            r = pooled(p)
            if r < best[0]:
                best = (r, ("rowgate", D, T)); print(f"    D={D},T={T}: {r:7.2f}  *")
    # smooth inverse-error blend on |b| (steeper)
    for p_exp in [2, 3, 4]:
        wc = 1.0 / (1.0 + (absb / 30.0) ** p_exp)
        r = pooled(wc * cloud + (1 - wc) * tuck)
        if r < best[0]:
            best = (r, ("steep", p_exp))
    # oracle bounds
    def rmse_sub(p, msk):
        return np.sqrt(np.mean((p[msk] - t[msk]) ** 2))
    owell = np.empty(N)
    for j in np.unique(wellid):
        msk = wellid == j
        owell[msk] = cloud[msk] if rmse_sub(cloud, msk) <= rmse_sub(tuck, msk) else tuck[msk]
    print(f"\n  oracle per-well (cloud|tuck) : {pooled(owell):7.2f}")
    prow = np.where(np.abs(cloud - t) <= np.abs(tuck - t), cloud, tuck)
    print(f"  oracle per-row  (cloud|tuck) : {pooled(prow):7.2f}  (unlearnable lower bound)")
    print(f"\n  >>> best practical router: {best[1]} -> pooled {best[0]:.2f}")

    meta = pd.DataFrame(wellrows)
    meta.to_csv(os.path.join(SUBDIR, "tucker_routed_oof.csv"), index=False)
    best_tuck = "tuck_plus"
    best_router = best[1]

    # ---------------- final fit + test submission ----------------
    print("\n=== final fit on all 773 + test submission ===")
    full_tree = cKDTree(np.vstack([w["cl_xy"] for w in wells]))
    full_F = np.concatenate([w["cl_F"] for w in wells])
    sample = pd.read_csv(os.path.join(gk.DATA, "sample_submission.csv"))
    pred_map = {}
    for wid in gk.list_wells("test"):
        d = prep(gk.load_well(wid, "test"))
        # eval rows for test: all eval rows (truth hidden) -> use full ev (finite Z)
        ev = d["ev"]; Zev = d["Z"][ev]
        ev_xy = np.column_stack([d["X"][ev], d["Y"][ev]])
        Fi, dm = knn_idw(full_tree, full_F, ev_xy)
        Fik, _ = knn_idw(full_tree, full_F, d["kn_xy"])
        anchor_b = float(np.median(d["kn_F"] - Fik)) if len(d["kn_F"]) >= 5 else 0.0
        cloud = Fi + anchor_b - Zev
        # tucker over all eval rows
        a = d["ps"] - 1
        Zseq = np.concatenate([[d["Z"][a]], d["Z"][ev]]); MDseq = np.concatenate([[d["MD"][a]], d["MD"][ev]])
        dZ = np.diff(Zseq); dMD = np.diff(MDseq)
        sign = 1.0 if best_tuck == "tuck_plus" else -1.0
        tuck = d["last_tvt"] + np.cumsum(-dZ + sign * d["own_dip"] * dMD)
        ab = np.full(len(ev), abs(anchor_b)); ostd = np.full(len(ev), d["own_std"])
        final = apply_router(best_router, cloud, tuck, dm, ab, ostd)
        for i, idx in zip(ev, final):
            pred_map[f"{wid}_{i}"] = float(idx)
    sample["tvt"] = sample["id"].map(pred_map).fillna(0.0)
    out = os.path.join(SUBDIR, "tucker_routed_submission.csv")
    sample.to_csv(out, index=False)
    print(f"  wrote {out} ({sample['id'].isin(pred_map).sum()}/{len(sample)} ids)  "
          f"using router {best_router}")
    print(f"\n  total time {time.time()-t0:.0f}s")
    print(f"  best pooled OOF = {best[0]:.2f} (target was <11; per-row oracle floor "
          f"for cloud|tucker = {pooled(prow):.2f})")


if __name__ == "__main__":
    main()
