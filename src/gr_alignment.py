"""Standalone per-row GR alignment model + fusion with cloud/tucker.

GR alignment: for each eval row, match its local GR window against (a) the
well's own known-zone GR ("self", pre-PS, slide-9 high-res) and (b) the
typewell GR, reading off the TVT where the character matches.  Vectorized
(windowed MSE / NCC via matmul; the spec's nested loop is ~2.6B iters).

Metrics compared: spec MSE unconstrained vs NCC band-constrained around the
flat trajectory.  Confidence gating blends the GR estimate onto tucker.

Fusion: route GR-alignment / cloud-IDW / tucker by confidence.  No existing
model weights used.
"""
import os, sys, time, json
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gp_kriging as gk

SUBDIR = os.path.join(gk.ROOT, "submissions"); os.makedirs(SUBDIR, exist_ok=True)
EPS = 1e-9
W = 101          # GR window length (~ spec window=51 -> 2*50+1)
ESTRIDE = 3
RSTRIDE = 3


def interp_nan(a):
    a = np.asarray(a, float).copy(); m = np.isfinite(a)
    if m.sum() == 0:
        return np.zeros_like(a), m
    if not m.all():
        idx = np.arange(len(a)); a[~m] = np.interp(idx[~m], idx[m], a[m])
    return a, m


def smooth(y):
    wl = min(61, len(y)); wl = wl - 1 if wl % 2 == 0 else wl
    return savgol_filter(y, wl, 3) if wl >= 5 else y


def align(ev_gr, ref_gr, ref_tvt, n, metric="mse", band=None, center=None):
    """Return (tvt_est, score) length n. metric in {mse,ncc}."""
    ev_i, _ = interp_nan(ev_gr); ref_i, _ = interp_nan(ref_gr)
    if len(ev_i) < W or len(ref_i) < W:
        return np.full(n, np.nan), np.zeros(n)
    EW = sliding_window_view(ev_i, W)[::ESTRIDE]
    ec = np.arange(0, len(ev_i) - W + 1, ESTRIDE) + W // 2
    RW = sliding_window_view(ref_i, W)[::RSTRIDE]
    rt = sliding_window_view(np.asarray(ref_tvt, float), W)[::RSTRIDE][:, W // 2]
    rmask = np.isfinite(rt)
    RW, rt = RW[rmask], rt[rmask]
    if len(rt) == 0:
        return np.full(n, np.nan), np.zeros(n)
    if metric == "ncc":
        En = EW - EW.mean(1, keepdims=True); En /= (En.std(1, keepdims=True) + EPS)
        Rn = RW - RW.mean(1, keepdims=True); Rn /= (Rn.std(1, keepdims=True) + EPS)
        S = En @ Rn.T / W                                # higher = better
    else:
        sE = (EW ** 2).sum(1); sR = (RW ** 2).sum(1)
        S = -(sE[:, None] + sR[None, :] - 2 * EW @ RW.T) / W   # higher = better (neg MSE)
    if band is not None and center is not None:
        cen = center[ec]
        ok = np.abs(rt[None, :] - cen[:, None]) <= band
        S = np.where(ok, S, -np.inf)
        has = ok.any(1)
    else:
        has = np.ones(len(ec), bool)
    best = np.argmax(S, 1)
    sc = S[np.arange(len(best)), best]
    est = np.where(has, rt[best], np.nan)
    # ncc score is already 0..1-ish; mse score map to pseudo-conf via rank later
    tvt_est = np.interp(np.arange(n), ec, est)
    score = np.interp(np.arange(n), ec, np.where(np.isfinite(sc), sc, 0.0))
    return tvt_est, score


def prep(w):
    ps, n = w["ps"], w["n"]; ev = np.arange(ps, n)
    Z, MD, GR, ti, tvt = w["Z"], w["MD"], w["GR"], w["TVT_input"], w["TVT"]
    a = ps - 1; last = ti[a]
    # tucker
    lo = max(1, ps - 200)
    dFk = np.diff(ti[lo - 1:ps]) + np.diff(Z[lo - 1:ps]); dMk = np.diff(MD[lo - 1:ps]); mk = dMk > 0.5
    own_dip = np.median(dFk[mk] / dMk[mk]) if mk.any() else 0.0
    Zseq = np.concatenate([[Z[a]], Z[ev]]); MDseq = np.concatenate([[MD[a]], MD[ev]])
    dZ_ev = np.diff(Zseq); dMD_ev = np.diff(MDseq)
    w["tvt_flat"] = last - np.cumsum(dZ_ev)
    w["tucker"] = last + np.cumsum(-dZ_ev + own_dip * dMD_ev)
    w["ev"] = ev; w["ev_Z"] = Z[ev]; w["ev_GR"] = GR[ev]
    w["kn_gr"] = GR[:ps]; w["kn_tvt"] = ti[:ps]
    if tvt is not None:
        w["ev_t"] = tvt[ev]; w["valid"] = np.isfinite(tvt[ev])
        m = np.isfinite(tvt) & np.isfinite(Z)
        w["cl_xy"] = np.column_stack([w["X"][m], w["Y"][m]]); w["cl_F"] = tvt[m] + Z[m]
    mk2 = np.isfinite(ti[:ps]) & np.isfinite(Z[:ps])
    w["kn_xy"] = np.column_stack([w["X"][:ps][mk2], w["Y"][:ps][mk2]]); w["kn_F"] = ti[:ps][mk2] + Z[:ps][mk2]
    return w


def gr_preds(w):
    ev_gr = w["ev_GR"]; n = len(ev_gr)
    flat = w["tvt_flat"]
    # spec: unconstrained MSE, self + typewell
    self_mse, sc_self = align(ev_gr, w["kn_gr"], w["kn_tvt"], n, "mse")
    tw_mse, sc_tw = align(ev_gr, w["tw_gr"], w["tw_tvt"], n, "mse")
    # improved: band-constrained NCC around flat trajectory
    self_nb, scn_self = align(ev_gr, w["kn_gr"], w["kn_tvt"], n, "ncc", band=60.0, center=flat)
    tw_nb, scn_tw = align(ev_gr, w["tw_gr"], w["tw_tvt"], n, "ncc", band=60.0, center=flat)
    blend = 0.7 * np.where(np.isfinite(self_mse), self_mse, flat) + \
            0.3 * np.where(np.isfinite(tw_mse), tw_mse, flat)
    blend_nb = 0.7 * np.where(np.isfinite(self_nb), self_nb, flat) + \
               0.3 * np.where(np.isfinite(tw_nb), tw_nb, flat)
    # confidence gating (step 3): gr local variance
    gr_i, gr_m = interp_nan(ev_gr)
    var = pd.Series(gr_i).rolling(51, center=True, min_periods=5).std().values
    conf = np.clip(var / 30.0, 0, 1)
    miss = ~gr_m
    best_gr = np.where(np.isfinite(self_nb), self_nb, flat)
    corr = np.clip(best_gr - w["tucker"], -20, 20)
    gated = np.where(miss | (conf < 0.2), w["tucker"], w["tucker"] + conf * corr)
    return dict(self_mse=self_mse, tw_mse=tw_mse, self_nb=self_nb, tw_nb=tw_nb,
                blend=blend, blend_nb=blend_nb, gated=gated,
                score_self=scn_self, score_tw=scn_tw, conf=conf)


def main():
    t0 = time.time()
    wells = [prep(gk.load_well(w, "train")) for w in gk.list_wells("train")]
    print(f"loaded {len(wells)} wells ({time.time()-t0:.0f}s)")

    # GR predictions (per-well, no training) + cloud via GroupKFold
    GP = [gr_preds(w) for w in wells]
    print(f"GR alignment computed ({time.time()-t0:.0f}s)")

    # cloud IDW (leave-own-well-out)
    cloud = [None] * len(wells)
    gkf = GroupKFold(5); g = np.arange(len(wells))
    for tr, va in gkf.split(g, g, g):
        tree = cKDTree(np.vstack([wells[i]["cl_xy"] for i in tr]))
        cF = np.concatenate([wells[i]["cl_F"] for i in tr])
        for j in va:
            w = wells[j]; ev = w["ev"]
            d, idx = tree.query(np.column_stack([w["X"][ev], w["Y"][ev]]), k=24, workers=-1)
            wt = 1.0 / (d + 1.0); Fi = (wt * cF[idx]).sum(1) / wt.sum(1)
            dk, ik = tree.query(w["kn_xy"], k=24, workers=-1)
            wtk = 1.0 / (dk + 1.0); Fik = (wtk * cF[ik]).sum(1) / wtk.sum(1)
            b = np.median(w["kn_F"] - Fik) if len(w["kn_F"]) >= 5 else 0.0
            cloud[j] = dict(pred=Fi + b - w["ev_Z"], absb=abs(b), dm=d.mean(1))
    print(f"cloud computed ({time.time()-t0:.0f}s)")

    # ---- score standalone variants ----
    def pooled(key, transform):
        sse = nn = 0.0
        for j, w in enumerate(wells):
            v = w["valid"]
            if v.sum() < 5:
                continue
            p = smooth(transform(j, w))[v]; t = w["ev_t"][v]
            sse += np.sum((p - t) ** 2); nn += v.sum()
        return np.sqrt(sse / nn)

    def perwell_med(transform):
        rs = []
        for j, w in enumerate(wells):
            v = w["valid"]
            if v.sum() < 5:
                continue
            p = smooth(transform(j, w))[v]; t = w["ev_t"][v]
            rs.append(np.sqrt(np.mean((p - t) ** 2)))
        return np.median(rs), rs

    variants = {
        "tucker":      lambda j, w: w["tucker"],
        "self_mse":    lambda j, w: np.where(np.isfinite(GP[j]["self_mse"]), GP[j]["self_mse"], w["tvt_flat"]),
        "tw_mse":      lambda j, w: np.where(np.isfinite(GP[j]["tw_mse"]), GP[j]["tw_mse"], w["tvt_flat"]),
        "self_nb":     lambda j, w: np.where(np.isfinite(GP[j]["self_nb"]), GP[j]["self_nb"], w["tvt_flat"]),
        "tw_nb":       lambda j, w: np.where(np.isfinite(GP[j]["tw_nb"]), GP[j]["tw_nb"], w["tvt_flat"]),
        "blend_mse":   lambda j, w: GP[j]["blend"],
        "blend_nb":    lambda j, w: GP[j]["blend_nb"],
        "gated":       lambda j, w: GP[j]["gated"],
        "cloud":       lambda j, w: cloud[j]["pred"],
    }
    print("\n=== STEP 4: standalone OOF (pooled / per-well median) ===")
    res = {}
    for name, fn in variants.items():
        p = pooled(name, fn); m, _ = perwell_med(fn)
        res[name] = p
        print(f"  {name:11s}: pooled {p:7.2f} | median {m:6.2f}")

    # ---- fusion: GR(self_nb) primary, cloud & tucker fallback by confidence ----
    print("\n=== STEP 5: fusion (GR + cloud + tucker) ===")
    def fuse(j, w):
        gp = GP[j]; cl = cloud[j]
        gr = np.where(np.isfinite(gp["self_nb"]), gp["self_nb"], np.nan)
        sc = gp["score_self"]                      # ncc 0..1
        wg = np.clip(sc, 0, 1) ** 2                # GR weight by alignment score
        wg = np.where(np.isfinite(gr), wg, 0.0)
        # cloud weight high when anchor agrees (small |b|); else low
        wc = 1.0 / (1.0 + cl["absb"] / 30.0)
        wc = np.full(len(w["ev"]), wc)
        base = w["tucker"]
        gr_f = np.where(np.isfinite(gr), gr, base)
        cl_f = cl["pred"]
        num = wg * gr_f + 0.5 * wc * cl_f + 0.3 * base
        den = wg + 0.5 * wc + 0.3
        return num / den
    pf = pooled("fuse", fuse); mf, _ = perwell_med(fuse)
    print(f"  fusion     : pooled {pf:7.2f} | median {mf:6.2f}")

    # ---- subset analysis ----
    print("\n=== per-well subset analysis (self_nb GR) ===")
    rows = []
    for j, w in enumerate(wells):
        v = w["valid"]
        if v.sum() < 5:
            continue
        sc = np.nanmean(GP[j]["score_self"][v]); miss = np.mean(~np.isfinite(w["ev_GR"][v]))
        gr_r = np.sqrt(np.mean((smooth(variants["self_nb"](j, w))[v] - w["ev_t"][v]) ** 2))
        tk_r = np.sqrt(np.mean((w["tucker"][v] - w["ev_t"][v]) ** 2))
        rows.append((sc, miss, gr_r, tk_r))
    R = pd.DataFrame(rows, columns=["score", "miss", "gr", "tuck"])
    for label, msk in [("easy score>0.5", R.score > 0.5), ("hard score<0.3", R.score < 0.3),
                       ("missGR>50%", R.miss > 0.5)]:
        s = R[msk]
        if len(s):
            print(f"  {label:16s} n={len(s):3d}: GR median {s.gr.median():6.2f}  vs tucker {s.tuck.median():6.2f}")

    print(f"\nself vs typewell standalone: self_nb {res['self_nb']:.2f} vs tw_nb {res['tw_nb']:.2f} "
          f"(mse: self {res['self_mse']:.2f} vs tw {res['tw_mse']:.2f})")
    print("\ncomparison: tucker 44.82 | align2(existing) ~30.7 | existing LGB 11.08 | target fusion <12")
    print(f"  best GR standalone = {min(res[k] for k in ['self_nb','tw_nb','blend_nb','gated']):.2f}  fusion = {pf:.2f}")

    best_gr = min(["self_nb", "tw_nb", "blend_nb", "gated"], key=lambda k: res[k])
    if res[best_gr] < 20:
        sub = pd.read_csv(os.path.join(gk.DATA, "sample_submission.csv"))
        # build test predictions with full cloud + GR fusion
        full_tree = cKDTree(np.vstack([w["cl_xy"] for w in wells]))
        full_F = np.concatenate([w["cl_F"] for w in wells])
        pmap = {}
        for wid in gk.list_wells("test"):
            d = prep(gk.load_well(wid, "test"))
            gpv = gr_preds(d)
            ev = d["ev"]
            dd, ii = full_tree.query(np.column_stack([d["X"][ev], d["Y"][ev]]), k=24, workers=-1)
            wt = 1.0 / (dd + 1.0); Fi = (wt * full_F[ii]).sum(1) / wt.sum(1)
            dk, ik = full_tree.query(d["kn_xy"], k=24, workers=-1)
            wtk = 1.0 / (dk + 1.0); Fik = (wtk * full_F[ik]).sum(1) / wtk.sum(1)
            b = np.median(d["kn_F"] - Fik) if len(d["kn_F"]) >= 5 else 0.0
            cl = dict(pred=Fi + b - d["Z"][ev], absb=abs(b), dm=dd.mean(1))
            gr = np.where(np.isfinite(gpv["self_nb"]), gpv["self_nb"], d["tucker"])
            wg = np.clip(gpv["score_self"], 0, 1) ** 2
            wc = 1.0 / (1.0 + cl["absb"] / 30.0)
            num = wg * gr + 0.5 * wc * cl["pred"] + 0.3 * d["tucker"]
            den = wg + 0.5 * wc + 0.3
            final = smooth(num / den)
            for i, p in zip(ev, final):
                pmap[f"{wid}_{i}"] = float(p)
        sub["tvt"] = sub["id"].map(pmap).fillna(0.0)
        out = os.path.join(SUBDIR, "gr_alignment_submission.csv")
        sub.to_csv(out, index=False)
        print(f"\nSTEP 5: saved {out} ({sub['id'].isin(pmap).sum()}/{len(sub)} ids)")
    print(f"\ntotal time {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
