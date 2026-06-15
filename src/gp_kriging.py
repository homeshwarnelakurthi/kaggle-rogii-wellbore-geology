"""Standalone Gaussian-Process kriging model for ROGII wellbore TVT.

Physics (exact): TVT = -Z + F(x,y) + b_well, b_well a true per-well constant.
  => eff_depth = TVT + Z = F(x,y) + b_well.
Z is known in the eval zone, so the TVT error EQUALS the F(x,y) error.

This module learns the regional surface F(x,y) with a GP on per-well
representative points, then per well anchors the absolute level using the
well's own known-zone eff_depth (TVT_input + Z). Two anchoring modes:

  A (plan spec): b_well = median(known_eff) - GP(rep_xy)
  B (residual kriging): b_well = median(known_eff - GP(known_xy))

An optional GR self-correlation correction nudges each eval row toward the
typewell TVT whose GR character matches (vectorized NCC, band-constrained).

Does NOT import any existing features.py model. Standalone from scratch.
"""
import os, sys, json, time, glob
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import savgol_filter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

EPS = 1e-9
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "rogii-wellbore-geology-prediction")
# Kaggle fallback
if not os.path.isdir(DATA):
    DATA = "/kaggle/input/competitions/rogii-wellbore-geology-prediction"
TRAIN = os.path.join(DATA, "train")
TEST = os.path.join(DATA, "test")
N_RESTARTS = int(os.environ.get("GP_RESTARTS", "5"))


# ----------------------------------------------------------------------------- helpers
def interp_nan(a):
    a = np.asarray(a, dtype=np.float64).copy()
    m = np.isfinite(a)
    if m.sum() == 0:
        return np.zeros_like(a)
    if not m.all():
        idx = np.arange(len(a))
        a[~m] = np.interp(idx[~m], idx[m], a[m])
    return a


def smooth(y):
    n = len(y)
    wl = min(61, n)
    if wl % 2 == 0:
        wl -= 1
    if wl >= 5:
        return savgol_filter(y, wl, 3)
    return y


def load_well(well_id, split):
    base = TRAIN if split == "train" else TEST
    h = pd.read_csv(os.path.join(base, f"{well_id}__horizontal_well.csv"))
    tw = pd.read_csv(os.path.join(base, f"{well_id}__typewell.csv"))
    ti = h["TVT_input"].values.astype(np.float64)
    known = np.isfinite(ti)
    n = len(h)
    ps = int(np.argmax(~known)) if (~known).any() else n
    ev = np.arange(ps, n)
    X = h["X"].values.astype(np.float64)
    Y = h["Y"].values.astype(np.float64)
    Z = h["Z"].values.astype(np.float64)
    MD = h["MD"].values.astype(np.float64)
    GR = h["GR"].values.astype(np.float64)
    tvt = h["TVT"].values.astype(np.float64) if "TVT" in h.columns else None
    kn_eff = ti[:ps] + Z[:ps]
    return dict(
        well=well_id, n=n, ps=ps, ev=ev,
        X=X, Y=Y, Z=Z, MD=MD, GR=GR, TVT_input=ti, TVT=tvt,
        tw_tvt=tw["TVT"].values.astype(np.float64),
        tw_gr=tw["GR"].values.astype(np.float64),
        rep_x=float(np.median(X)), rep_y=float(np.median(Y)),
        kn_eff=kn_eff[np.isfinite(kn_eff)],
        ids=[f"{well_id}_{i}" for i in ev],
    )


def list_wells(split):
    base = TRAIN if split == "train" else TEST
    fs = sorted(glob.glob(os.path.join(base, "*__horizontal_well.csv")))
    return [os.path.basename(f).split("__")[0] for f in fs]


# ----------------------------------------------------------------------------- GP
def build_gp(X_gp, y_gp, n_restarts=N_RESTARTS):
    scaler = StandardScaler().fit(X_gp)
    Xs = scaler.transform(X_gp)
    kernel = (ConstantKernel(1.0, (1e-2, 1e4))
              * Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
              + WhiteKernel(1e-2, (1e-5, 1e2)))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts,
                                  normalize_y=True, alpha=1e-6, random_state=0)
    gp.fit(Xs, y_gp)
    return gp, scaler


# ----------------------------------------------------------------------------- GR correction
def gr_correction(tvt_pred, ev_gr, tw_tvt, tw_gr, window=51, weight=0.2,
                  min_corr=0.3, clip=15.0, band=60.0):
    """Vectorized normalized-cross-correlation of eval GR windows against
    typewell GR windows; nudge tvt_pred toward the best-matching typewell TVT.
    Band-constrained to +/- `band` ft around the current prediction so the
    match cannot snap to a different stratigraphic cycle."""
    n = len(tvt_pred)
    W = window if window % 2 == 1 else window + 1
    if n < W:
        return tvt_pred
    ev = interp_nan(ev_gr)
    mt = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    if mt.sum() < W + 5:
        return tvt_pred
    tw_t = tw_tvt[mt]; tw_g = interp_nan(tw_gr[mt])
    o = np.argsort(tw_t); tw_t = tw_t[o]; tw_g = tw_g[o]

    ev_win = sliding_window_view(ev, W)
    tw_win = sliding_window_view(tw_g, W)
    tw_ctr = sliding_window_view(tw_t, W).mean(1)
    centers = np.arange(W // 2, W // 2 + ev_win.shape[0])

    en = ev_win - ev_win.mean(1, keepdims=True)
    en /= (en.std(1, keepdims=True) + EPS)
    tn = tw_win - tw_win.mean(1, keepdims=True)
    tn /= (tn.std(1, keepdims=True) + EPS)
    C = en @ tn.T / W

    cpred = tvt_pred[centers]
    if band is not None:
        inb = np.abs(tw_ctr[None, :] - cpred[:, None]) <= band
        C = np.where(inb, C, -np.inf)
        has = inb.any(1)
    else:
        has = np.ones(len(centers), bool)
    best = np.argmax(C, axis=1)
    bestcorr = np.where(has, C[np.arange(len(best)), best], -1.0)
    best_tvt = np.where(has, tw_ctr[best], cpred)

    corr_full = np.interp(np.arange(n), centers, bestcorr)
    btvt_full = np.interp(np.arange(n), centers, best_tvt)
    correction = np.clip(btvt_full - tvt_pred, -clip, clip)
    return tvt_pred + np.where(corr_full > min_corr, weight * correction, 0.0)


# ----------------------------------------------------------------------------- per-well prediction
def predict_configs(d, gp, scaler, configs, n_known_anchor=300, band=60.0):
    """Return {config_name: tvt_pred over eval rows}. Shares the GP evaluation
    across configs.  config name grammar: <anchor>[_gr][_raw]
      anchor in {A,B}; _gr adds GR correction; _raw skips savgol."""
    ev = d["ev"]
    Zev = d["Z"][ev]
    ev_gr = d["GR"][ev]
    if len(d["kn_eff"]) == 0:
        base_level = float(np.median(d["TVT_input"][np.isfinite(d["TVT_input"])] +
                                     d["Z"][np.isfinite(d["TVT_input"])]))
    else:
        base_level = float(np.median(d["kn_eff"]))

    ev_xy = scaler.transform(np.column_stack([d["X"][ev], d["Y"][ev]]))
    ev_gp = gp.predict(ev_xy)

    # anchor A: GP at whole-well representative location
    rep_xy = scaler.transform([[d["rep_x"], d["rep_y"]]])
    gp_rep = float(gp.predict(rep_xy)[0])
    b_A = base_level - gp_rep

    # anchor B: median GP residual over (subsampled) known-zone points
    ps = d["ps"]
    kn_mask = np.isfinite(d["TVT_input"][:ps]) & np.isfinite(d["Z"][:ps])
    kn_idx = np.arange(ps)[kn_mask]
    if len(kn_idx) > n_known_anchor:
        kn_idx = kn_idx[np.linspace(0, len(kn_idx) - 1, n_known_anchor).astype(int)]
    if len(kn_idx) >= 5:
        kn_xy = scaler.transform(np.column_stack([d["X"][kn_idx], d["Y"][kn_idx]]))
        kn_gp = gp.predict(kn_xy)
        kn_eff_pts = d["TVT_input"][kn_idx] + d["Z"][kn_idx]
        b_B = float(np.median(kn_eff_pts - kn_gp))
    else:
        b_B = b_A

    out = {}
    cache = {}
    for cfg in configs:
        anchor = cfg[0]
        b = b_A if anchor == "A" else b_B
        key = (anchor, "gr" in cfg)
        if key not in cache:
            pred = ev_gp + b - Zev
            if "gr" in cfg:
                pred = gr_correction(pred, ev_gr, d["tw_tvt"], d["tw_gr"], band=band)
            cache[key] = pred
        pred = cache[key]
        out[cfg] = pred if "raw" in cfg else smooth(pred)
    return out


# ----------------------------------------------------------------------------- scoring
def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def well_truth(d):
    ev = d["ev"]
    t = d["TVT"][ev]
    m = np.isfinite(t)
    return t, m
