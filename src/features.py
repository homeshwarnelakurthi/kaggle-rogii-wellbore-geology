"""Feature extraction for ROGII wellbore TVT prediction.

Physics: TVT = -Z + F(x,y) + b_well  (b_well constant per well)
  => TVT - last_tvt = -(Z - Z_anchor) + dF,  dF = F(x,y) - F(anchor)

Model target = dF (detrended residual). Reconstruction:
  TVT_pred = last_tvt - (Z - Z_anchor) + dF_pred

Every feature here uses only columns present in BOTH train and test
horizontal wells (MD, X, Y, Z, GR, TVT_input) and typewells (TVT, GR),
plus the spatial KNN context built from full train-well labels
(available at Kaggle inference time as part of the model dataset).
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial import cKDTree

EPS = 1e-6


def interp_nan(a):
    """Linearly interpolate NaNs; returns (filled, finite_mask)."""
    a = np.asarray(a, dtype=np.float64).copy()
    m = np.isfinite(a)
    if 0 < m.sum() < len(a):
        idx = np.arange(len(a))
        a[~m] = np.interp(idx[~m], idx[m], a[m])
    return a, m


def self_corr(kn_gr, kn_tvt, ev_gr, ev_tvt_flat, window=51, stride=5,
              band=40.0):
    """'The well is its own typewell': match eval-zone GR windows against
    pre-PS GR windows (same tool, same hole) and read off the TVT where the
    character matches. Vectorized normalized cross-correlation via matmul.

    The search is constrained to known-zone windows whose TVT lies within
    +/- band ft of the flat-trajectory estimate at the eval window center —
    GR character repeats across stratigraphic cycles, so an unconstrained
    argmax snaps to the wrong cycle.

    Returns (tvt_estimate, confidence) arrays of len(ev_gr).
    """
    n_ev = len(ev_gr)
    out_tvt = np.full(n_ev, np.nan)
    out_conf = np.full(n_ev, np.nan)
    if len(kn_gr) < window + stride or n_ev < window:
        return out_tvt, out_conf
    kn_i, kn_m = interp_nan(kn_gr)
    ev_i, ev_m = interp_nan(ev_gr)
    if kn_m.mean() < 0.3 or ev_m.sum() < window:
        return out_tvt, out_conf

    kw = sliding_window_view(kn_i, window)[::stride]               # (Nk, w)
    kt = sliding_window_view(np.asarray(kn_tvt, dtype=np.float64),
                             window)[::stride].mean(axis=1)        # (Nk,)
    ew = sliding_window_view(ev_i, window)[::stride]               # (Ne, w)
    centers = np.arange(0, n_ev - window + 1, stride) + window // 2

    kn_n = kw - kw.mean(axis=1, keepdims=True)
    kn_n = kn_n / (kn_n.std(axis=1, keepdims=True) + EPS)
    ev_n = ew - ew.mean(axis=1, keepdims=True)
    ev_n = ev_n / (ev_n.std(axis=1, keepdims=True) + EPS)

    C = (ev_n @ kn_n.T) / window                                   # (Ne, Nk)
    flat_c = ev_tvt_flat[centers]
    in_band = np.abs(kt[None, :] - flat_c[:, None]) <= band
    C = np.where(in_band, C, -np.inf)
    has_cand = in_band.any(axis=1)
    best = np.argmax(C, axis=1)
    conf = C[np.arange(len(best)), best]
    est = kt[best]
    conf = np.where(has_cand, conf, 0.0)
    est = np.where(has_cand, est, flat_c)

    # windows that are mostly interpolated (missing GR) are unreliable
    ev_valid = sliding_window_view(ev_m.astype(np.float64), window)[::stride].mean(axis=1)
    conf = np.where(ev_valid > 0.5, conf, 0.0)

    out_tvt = np.interp(np.arange(n_ev), centers, est)
    out_conf = np.interp(np.arange(n_ev), centers, conf)
    return out_tvt, out_conf


def make_gr_curve(tvt, gr, binsize=0.5, smooth=5):
    """Reference GR(TVT) curve: binned median, lightly smoothed.
    Returns (grid_tvt, grid_gr) or (None, None)."""
    m = np.isfinite(tvt) & np.isfinite(gr)
    if m.sum() < 20:
        return None, None
    b = np.round(tvt[m] / binsize).astype(np.int64)
    s = pd.DataFrame({"b": b, "gr": gr[m]}).groupby("b")["gr"].median()
    full = np.arange(s.index.min(), s.index.max() + 1)
    vals = s.reindex(full).interpolate(limit_direction="both").values
    vals = pd.Series(vals).rolling(smooth, center=True, min_periods=1).mean().values
    return full * binsize, vals


def seq_align(ref_tvt, ref_gr, ev_gr, ev_tvt_flat, max_shift=50.0,
              step=1.0, win=201):
    """TVT-domain alignment of the lateral against a GR(TVT) reference curve.

    The eval zone is near-horizontal: a smoothed eval GR sample reads GR at
    a single stratigraphic depth. For each row choose the shift d minimizing
    |GR_ev - ref(tvt_flat + d)| averaged over a +/-win/2 row neighborhood.
    The chosen d is a direct estimate of target_dF = TVT - tvt_flat.

    Returns (delta, confidence) arrays of len(ev_gr).
    """
    from scipy.ndimage import uniform_filter1d
    n = len(ev_gr)
    if ref_tvt is None or n < 20:
        return np.full(n, np.nan), np.full(n, np.nan)
    ev_s, ev_m = interp_nan(ev_gr)
    if ev_m.sum() < 20:
        return np.full(n, np.nan), np.full(n, np.nan)
    ev_s = pd.Series(ev_s).rolling(21, center=True, min_periods=1).mean().values

    deltas = np.arange(-max_shift, max_shift + step, step)
    grid = ev_tvt_flat[:, None] + deltas[None, :]            # (n, nd)
    ref_v = np.interp(grid, ref_tvt, ref_gr,
                      left=np.nan, right=np.nan)
    cost = np.abs(ev_s[:, None] - ref_v)
    w = (ev_m.astype(np.float64))[:, None] * np.isfinite(cost)
    num = uniform_filter1d(np.nan_to_num(cost) * w, size=win, axis=0)
    den = uniform_filter1d(w, size=win, axis=0)
    c = num / (den + EPS)
    c[den < 0.05] = np.nan                                   # too little support

    valid = np.isfinite(c).any(axis=1)
    delta = np.full(n, np.nan)
    conf = np.full(n, np.nan)
    if valid.any():
        cv = np.where(np.isfinite(c), c, np.inf)
        best = np.argmin(cv[valid], axis=1)
        delta[valid] = deltas[best]
        cmin = cv[valid, best]
        cmed = np.nanmedian(np.where(np.isfinite(c[valid]), c[valid], np.nan),
                            axis=1)
        conf[valid] = (cmed - cmin) / (cmed + EPS)
    return delta, conf


def gr_by_tvt_lookup(tvt, gr, q_tvt, binsize=1.0):
    """Median GR per TVT bin from the known zone, looked up at query TVT."""
    m = np.isfinite(tvt) & np.isfinite(gr)
    if m.sum() < 10:
        return np.full(len(q_tvt), np.nan)
    b = np.round(tvt[m] / binsize).astype(np.int64)
    df = pd.DataFrame({"b": b, "gr": gr[m]}).groupby("b")["gr"].median()
    centers = df.index.values * binsize
    vals = df.values
    out = np.interp(q_tvt, centers, vals, left=np.nan, right=np.nan)
    return out


def tail_slope(y, x, n):
    """OLS slope of y vs x over the last n samples (NaN-safe)."""
    y = y[-n:]; x = x[-n:]
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < max(5, n // 4):
        return np.nan
    xv, yv = x[m], y[m]
    xc = xv - xv.mean()
    den = (xc ** 2).sum()
    return float((xc * (yv - yv.mean())).sum() / den) if den > 0 else np.nan


class KnnContext:
    """Spatial context from the 773 train wells (rebuilt from raw arrays so
    the pickle is portable across scipy versions)."""

    def __init__(self, pts_xy, pts_F, pts_well, pts_egfdu,
                 well_ids, well_anchor_xy, well_gx, well_gy):
        self.pts_xy = pts_xy            # (Np,2)
        self.pts_F = pts_F              # (Np,)  F = TVT + Z (full train labels)
        self.pts_well = pts_well        # (Np,) int well index
        self.pts_egfdu = pts_egfdu      # (Np,)  EGFDU formation depth (may be NaN)
        self.well_ids = well_ids        # list[str]
        self.well_anchor_xy = well_anchor_xy  # (Nw,2)
        self.well_gx = well_gx          # (Nw,) per-well plane dip dF/dx
        self.well_gy = well_gy
        self._tree = None
        self._wtree = None

    def wtree(self):
        if self._wtree is None:
            self._wtree = cKDTree(self.well_anchor_xy)
        return self._wtree

    def make_view(self, exclude_well=None):
        """Query view; pass exclude_well (int index) for leave-own-well-out.
        Builds a KD-tree that physically excludes that well's points so the
        k nearest neighbors are always from other wells."""
        if exclude_well is None:
            keep = slice(None)
        else:
            keep = self.pts_well != exclude_well
        return CtxView(self.pts_xy[keep], self.pts_F[keep], self.pts_egfdu[keep])

    def neighbor_dip(self, anchor_xy, exclude_well=None, k=8):
        """Average plane dip (dF/dx, dF/dy) of the k nearest wells."""
        kq = min(k + 1, len(self.well_anchor_xy))
        d, idx = self.wtree().query(anchor_xy.reshape(1, 2), k=kq)
        d, idx = d[0], idx[0]
        if exclude_well is not None:
            keep = idx != exclude_well
            d, idx = d[keep], idx[keep]
        d, idx = d[:k], idx[:k]
        w = 1.0 / (d + 1.0)
        gx = self.well_gx[idx]; gy = self.well_gy[idx]
        m = np.isfinite(gx) & np.isfinite(gy)
        if m.sum() == 0:
            return np.nan, np.nan
        return (float((w[m] * gx[m]).sum() / w[m].sum()),
                float((w[m] * gy[m]).sum() / w[m].sum()))


def save_context(ctx, path):
    """Persist KnnContext as plain arrays (.npz) — no pickle involved."""
    np.savez_compressed(
        path, pts_xy=ctx.pts_xy, pts_F=ctx.pts_F, pts_well=ctx.pts_well,
        pts_egfdu=ctx.pts_egfdu, well_ids=np.array(ctx.well_ids),
        well_anchor_xy=ctx.well_anchor_xy, well_gx=ctx.well_gx,
        well_gy=ctx.well_gy)


def load_context(path):
    d = np.load(path, allow_pickle=False)
    return KnnContext(
        pts_xy=d["pts_xy"], pts_F=d["pts_F"], pts_well=d["pts_well"],
        pts_egfdu=d["pts_egfdu"], well_ids=[str(w) for w in d["well_ids"]],
        well_anchor_xy=d["well_anchor_xy"], well_gx=d["well_gx"],
        well_gy=d["well_gy"])


class CtxView:
    """KD-tree over a (possibly own-well-excluded) point cloud."""

    def __init__(self, pts_xy, pts_F, pts_egfdu):
        self.pts_F = pts_F
        self.pts_egfdu = pts_egfdu
        self.tree = cKDTree(pts_xy)

    def query_F(self, q_xy, k=12):
        """Inverse-distance-weighted F and EGFDU at query points."""
        d, idx = self.tree.query(q_xy, k=k, workers=-1)
        w = 1.0 / (d + 1.0)
        wsum = w.sum(axis=1) + EPS
        F = (w * self.pts_F[idx]).sum(axis=1) / wsum
        eg_vals = self.pts_egfdu[idx]
        weg = np.where(np.isfinite(eg_vals), w, 0.0)
        egsum = weg.sum(axis=1)
        eg = np.where(egsum > 0,
                      np.nansum(weg * eg_vals, axis=1) / (egsum + EPS), np.nan)
        return F, eg, d.mean(axis=1)

    def query_plane(self, q_xy, k=32, scale=400.0):
        """Local weighted plane fit of F around each query point.

        Returns (F_plane, gx, gy, fit_rms): plane value at the query,
        local surface gradient, and residual scatter of the fit.
        Batched closed-form 3x3 normal equations.
        """
        d, idx = self.tree.query(q_xy, k=k, workers=-1)
        x1 = self.tree.data[idx, 0] - q_xy[:, 0:1]                 # (n,k)
        x2 = self.tree.data[idx, 1] - q_xy[:, 1:2]
        f = self.pts_F[idx]
        w = np.exp(-(d / scale) ** 2) + 1e-4                       # gaussian kernel

        Sw = w.sum(1)
        M = np.empty((len(q_xy), 3, 3))
        M[:, 0, 0] = (w * x1 * x1).sum(1)
        M[:, 0, 1] = M[:, 1, 0] = (w * x1 * x2).sum(1)
        M[:, 0, 2] = M[:, 2, 0] = (w * x1).sum(1)
        M[:, 1, 1] = (w * x2 * x2).sum(1)
        M[:, 1, 2] = M[:, 2, 1] = (w * x2).sum(1)
        M[:, 2, 2] = Sw
        # ridge on gradient terms guards collinear neighbor geometry
        lam = 1e-4 * (M[:, 0, 0] + M[:, 1, 1]) + 1e-6
        M[:, 0, 0] += lam
        M[:, 1, 1] += lam
        v = np.stack([(w * x1 * f).sum(1), (w * x2 * f).sum(1),
                      (w * f).sum(1)], axis=1)
        try:
            c = np.linalg.solve(M, v)                              # (n,3)
        except np.linalg.LinAlgError:
            c = np.stack([np.linalg.lstsq(M[i], v[i], rcond=None)[0]
                          for i in range(len(M))])
        F_plane = c[:, 2]
        resid = f - (c[:, 0:1] * x1 + c[:, 1:2] * x2 + c[:, 2:3])
        fit_rms = np.sqrt((w * resid ** 2).sum(1) / Sw)
        return F_plane, c[:, 0], c[:, 1], fit_rms


def fit_plane(x, y, f):
    """Least-squares plane f ~ gx*x + gy*y + c. Returns (gx, gy)."""
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(f)
    if m.sum() < 30:
        return np.nan, np.nan
    A = np.column_stack([x[m] - x[m].mean(), y[m] - y[m].mean(),
                         np.ones(m.sum())])
    try:
        coef, *_ = np.linalg.lstsq(A, f[m] - f[m].mean(), rcond=None)
        return float(coef[0]), float(coef[1])
    except np.linalg.LinAlgError:
        return np.nan, np.nan


def extract_well(h, tw, well_id, ctx=None, exclude_well_idx=None,
                 is_train=False):
    """Extract eval-zone features for one well.

    h:  horizontal well df (MD, X, Y, Z, GR, TVT_input [, TVT if train])
    tw: typewell df (TVT, GR)
    Returns DataFrame with id, features, and (if train) target columns.
    """
    md = h["MD"].values.astype(np.float64)
    X = h["X"].values.astype(np.float64)
    Y = h["Y"].values.astype(np.float64)
    Z = h["Z"].values.astype(np.float64)
    gr = h["GR"].values.astype(np.float64)
    tvt_in = h["TVT_input"].values.astype(np.float64)
    n = len(h)

    known = np.isfinite(tvt_in)
    ps = int(np.argmax(~known))          # first eval row
    ev = np.arange(ps, n)
    n_ev = len(ev)
    a = ps - 1                            # anchor row

    last_tvt = tvt_in[a]
    Za, Xa, Ya = Z[a], X[a], Y[a]
    F_anchor = last_tvt + Za

    dmd = np.gradient(md)
    dz = np.gradient(Z) / dmd
    dx = np.gradient(X) / dmd
    dy = np.gradient(Y) / dmd

    # flat-physics baseline (model learns the correction dF on top of this)
    dz_cum = Z[ev] - Za                       # cumulative Z change from anchor
    tvt_flat = last_tvt - dz_cum
    md_since = md[ev] - md[a]
    frac = (np.arange(n_ev) + 1) / n_ev

    f = {}
    # --- trajectory geometry ---
    f["dz_per_ft"] = dz[ev]
    f["dx_per_ft"] = dx[ev]
    f["dy_per_ft"] = dy[ev]
    dip_ang = np.arctan(dz[ev])
    f["sin_dip"] = np.sin(dip_ang)
    f["cos_dip"] = np.cos(dip_ang)
    f["neg_dz_cum"] = -dz_cum
    f["md_since"] = md_since
    f["md_since_norm"] = md_since / 5000.0
    f["frac_along_eval"] = frac
    f["horiz_dist"] = np.hypot(X[ev] - Xa, Y[ev] - Ya)

    # --- GR features (full-well rolling, sliced to eval) ---
    gr_s = pd.Series(gr)
    for w in (5, 21, 51, 101):
        f[f"gr_roll_{w}"] = gr_s.rolling(w, center=True, min_periods=3).mean().values[ev]
    gr_i, gr_m = interp_nan(gr)
    grad = np.gradient(gr_i) if np.isfinite(gr_i).all() else np.full(n, np.nan)
    f["gr_grad"] = np.where(gr_m, grad, np.nan)[ev]
    f["gr"] = gr[ev]
    f["gr_missing"] = (~gr_m[ev]).astype(np.float64)
    f["gr_missing_frac_well"] = np.full(n_ev, (~gr_m[ev]).mean())

    # --- known-zone GR statistics ---
    kn_gr = gr[:ps]
    kn_gr_fin = kn_gr[np.isfinite(kn_gr)]
    gr_kn_mean = kn_gr_fin.mean() if len(kn_gr_fin) > 5 else np.nan
    gr_kn_std = kn_gr_fin.std() if len(kn_gr_fin) > 5 else np.nan
    tail = kn_gr[-50:]
    tail_fin = tail[np.isfinite(tail)]
    f["gr_kn_mean"] = np.full(n_ev, gr_kn_mean)
    f["gr_kn_std"] = np.full(n_ev, gr_kn_std)
    f["gr_kn_tail_mean"] = np.full(n_ev, tail_fin.mean() if len(tail_fin) > 3 else np.nan)
    f["gr_z"] = (gr[ev] - gr_kn_mean) / (gr_kn_std + EPS)

    # --- GR vs typewell at flat TVT ---
    tw_tvt = tw["TVT"].values.astype(np.float64)
    tw_gr = tw["GR"].values.astype(np.float64)
    twm = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
    if twm.sum() > 10:
        o = np.argsort(tw_tvt[twm])
        tw_lookup = np.interp(tvt_flat, tw_tvt[twm][o], tw_gr[twm][o],
                              left=np.nan, right=np.nan)
    else:
        tw_lookup = np.full(n_ev, np.nan)
    f["gr_vs_typewell_flat"] = gr[ev] - tw_lookup

    # --- GR vs own known zone at flat TVT ---
    f["gr_vs_known_flat"] = gr[ev] - gr_by_tvt_lookup(tvt_in[:ps], kn_gr, tvt_flat)

    # --- own known-zone dip / slope / velocity (per-well scalars) ---
    F_kn = tvt_in[:ps] + Z[:ps]
    dF_kn = np.gradient(F_kn) / dmd[:ps]
    dip50 = np.nanmedian(dF_kn[-50:]) if ps >= 50 else np.nan
    dip200 = np.nanmedian(dF_kn[-200:]) if ps >= 200 else np.nan
    f["dip_own_tail_50"] = np.full(n_ev, dip50)
    f["dip_own_tail_200"] = np.full(n_ev, dip200)
    f["dip_own_std"] = np.full(n_ev, np.nanstd(dF_kn))
    s30 = tail_slope(tvt_in[:ps], md[:ps], 30)
    s100 = tail_slope(tvt_in[:ps], md[:ps], 100)
    f["slope_tail_30"] = np.full(n_ev, s30)
    f["slope_tail_100"] = np.full(n_ev, s100)

    # velocity model: dTVT = beta*dZ + c  (per 1-ft MD)
    dtvt_kn = np.diff(tvt_in[:ps]) / np.diff(md[:ps])
    dz_kn = np.diff(Z[:ps]) / np.diff(md[:ps])
    vm = np.isfinite(dtvt_kn) & np.isfinite(dz_kn)
    if vm.sum() > 50:
        A = np.column_stack([dz_kn[vm], np.ones(vm.sum())])
        (beta, c), *_ = np.linalg.lstsq(A, dtvt_kn[vm], rcond=None)
    else:
        beta, c = np.nan, np.nan
    f["velocity_beta"] = np.full(n_ev, beta)
    f["velocity_intercept"] = np.full(n_ev, c)

    # direct dF candidates the trees can pick from
    f["proj_dip50"] = dip50 * md_since
    f["proj_dip200"] = dip200 * md_since
    f["proj_slope30"] = (s30 * md_since + dz_cum) if np.isfinite(s30) else np.full(n_ev, np.nan)
    f["proj_slope100"] = (s100 * md_since + dz_cum) if np.isfinite(s100) else np.full(n_ev, np.nan)
    f["proj_velocity"] = (beta + 1.0) * dz_cum + c * md_since if np.isfinite(beta) else np.full(n_ev, np.nan)

    # own-well plane fit on known-zone F
    gx_own, gy_own = fit_plane(X[:ps], Y[:ps], F_kn)
    f["own_plane_dF"] = (gx_own * (X[ev] - Xa) + gy_own * (Y[ev] - Ya)
                         if np.isfinite(gx_own) else np.full(n_ev, np.nan))

    # --- self-correlation (pre-PS GR is its own typewell) ---
    # multi-scale, band-constrained around the flat-trajectory estimate
    sc31, cf31 = self_corr(kn_gr, tvt_in[:ps], gr[ev], tvt_flat, window=31)
    sc81, cf81 = self_corr(kn_gr, tvt_in[:ps], gr[ev], tvt_flat, window=81)
    f["self_corr_conf"] = cf31
    f["self_corr_conf_81"] = cf81
    f["self_corr_resid_flat"] = sc31 - tvt_flat
    f["self_corr_resid_flat_81"] = sc81 - tvt_flat
    f["self_corr_resid_anchor"] = sc31 - last_tvt
    f["self_corr_gated"] = np.where(cf31 > 0.6, sc31 - tvt_flat, np.nan)
    w31 = np.clip(cf31, 0, None) ** 2
    w81 = np.clip(cf81, 0, None) ** 2
    den = w31 + w81
    sc_mix = np.where(den > 0.1,
                      (w31 * np.nan_to_num(sc31) + w81 * np.nan_to_num(sc81))
                      / (den + EPS), np.nan)
    f["self_corr_mix_resid"] = sc_mix - tvt_flat

    # --- TVT-domain GR alignment (the physically correct correlation) ---
    tw_curve = make_gr_curve(tw_tvt, tw_gr)
    own_curve = make_gr_curve(tvt_in[:ps], kn_gr)
    d_tw, c_tw = seq_align(tw_curve[0], tw_curve[1], gr[ev], tvt_flat)
    d_own, c_own = seq_align(own_curve[0], own_curve[1], gr[ev], tvt_flat)
    f["align_tw_delta"] = d_tw
    f["align_tw_conf"] = c_tw
    f["align_own_delta"] = d_own
    f["align_own_conf"] = c_own
    f["align_tw_gated"] = np.where(np.nan_to_num(c_tw) > 0.3, d_tw, np.nan)
    f["align_own_gated"] = np.where(np.nan_to_num(c_own) > 0.3, d_own, np.nan)

    # --- known-zone tail curvature / dip trend ---
    s500 = tail_slope(tvt_in[:ps], md[:ps], 500)
    f["slope_tail_500"] = np.full(n_ev, s500)
    f["dip_trend"] = np.full(n_ev, (s100 - s500) if np.isfinite(s500) else np.nan)
    dip_change = (np.nanmedian(dF_kn[-50:]) - np.nanmedian(dF_kn[-500:-300])
                  if ps >= 500 else np.nan)
    f["dip_change"] = np.full(n_ev, dip_change)

    # --- spatial KNN context ---
    if ctx is not None:
        view = ctx.make_view(exclude_well=exclude_well_idx)
        q_xy = np.column_stack([X[ev], Y[ev]])
        a_xy = np.array([[Xa, Ya]])
        Fq, egq, dq = view.query_F(q_xy)
        Fa_knn, ega, _ = view.query_F(a_xy)
        f["knn_dF"] = Fq - Fa_knn[0]
        f["knn_F_minus_anchor"] = Fq - F_anchor
        f["knn_dist_mean"] = dq
        f["egfdu_drift"] = egq - ega[0]
        f["z_minus_egfdu"] = Z[ev] - egq

        # local plane fit of the F surface at every eval row
        Fp, gxq, gyq, prms = view.query_plane(q_xy)
        Fpa, gxa, gya, _ = view.query_plane(a_xy)
        f["plane_dF"] = Fp - Fpa[0]
        f["plane_F_minus_anchor"] = Fp - F_anchor
        f["plane_gx"] = gxq
        f["plane_gy"] = gyq
        f["plane_fit_rms"] = prms
        # path integral of the local surface gradient along the lateral
        dXe = np.diff(np.concatenate([[Xa], X[ev]]))
        dYe = np.diff(np.concatenate([[Ya], Y[ev]]))
        f["plane_dF_path"] = np.cumsum(gxq * dXe + gyq * dYe)

        # second-stage GR alignment centered on the structural estimate:
        # plane_dF_path is usually within ~5 ft, so a tight band avoids
        # wrong-cycle locking and only fine-tunes stratigraphic position
        center = tvt_flat + f["plane_dF_path"]
        d_tw2, c_tw2 = seq_align(tw_curve[0], tw_curve[1], gr[ev], center,
                                 max_shift=15.0)
        d_own2, c_own2 = seq_align(own_curve[0], own_curve[1], gr[ev], center,
                                   max_shift=15.0)
        f["align2_tw_delta"] = d_tw2
        f["align2_tw_conf"] = c_tw2
        f["align2_own_delta"] = d_own2
        f["align2_own_conf"] = c_own2
        f["align2_tw_est"] = f["plane_dF_path"] + d_tw2
        f["align2_own_est"] = f["plane_dF_path"] + d_own2

        ndx, ndy = ctx.neighbor_dip(np.array([Xa, Ya]),
                                    exclude_well=exclude_well_idx)
        f["nb_dip_x"] = np.full(n_ev, ndx)
        f["nb_dip_y"] = np.full(n_ev, ndy)
        f["nb_plane_dF"] = (ndx * (X[ev] - Xa) + ndy * (Y[ev] - Ya)
                            if np.isfinite(ndx) else np.full(n_ev, np.nan))

    out = pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})
    out.insert(0, "id", [f"{well_id}_{i}" for i in ev])
    out["well"] = well_id
    out["row_idx"] = ev.astype(np.int32)
    out["last_tvt"] = np.float64(last_tvt)
    out["tvt_flat"] = tvt_flat
    if is_train and "TVT" in h.columns:
        tvt_true = h["TVT"].values[ev].astype(np.float64)
        out["tvt_true"] = tvt_true
        out["target_dF"] = tvt_true - tvt_flat   # = (TVT - last_tvt) + dz_cum
    return out


FEATURE_COLS = [
    "dz_per_ft", "dx_per_ft", "dy_per_ft", "sin_dip", "cos_dip",
    "neg_dz_cum", "md_since", "md_since_norm", "frac_along_eval", "horiz_dist",
    "gr_roll_5", "gr_roll_21", "gr_roll_51", "gr_roll_101", "gr_grad", "gr",
    "gr_missing", "gr_missing_frac_well",
    "gr_kn_mean", "gr_kn_std", "gr_kn_tail_mean", "gr_z",
    "gr_vs_typewell_flat", "gr_vs_known_flat",
    "dip_own_tail_50", "dip_own_tail_200", "dip_own_std",
    "slope_tail_30", "slope_tail_100", "velocity_beta", "velocity_intercept",
    "proj_dip50", "proj_dip200", "proj_slope30", "proj_slope100", "proj_velocity",
    "own_plane_dF",
    "self_corr_conf", "self_corr_conf_81", "self_corr_resid_flat",
    "self_corr_resid_flat_81", "self_corr_resid_anchor", "self_corr_gated",
    "self_corr_mix_resid",
    "align_tw_delta", "align_tw_conf", "align_own_delta", "align_own_conf",
    "align_tw_gated", "align_own_gated",
    "slope_tail_500", "dip_trend", "dip_change",
    "knn_dF", "knn_F_minus_anchor", "knn_dist_mean",
    "plane_dF", "plane_F_minus_anchor", "plane_gx", "plane_gy",
    "plane_fit_rms", "plane_dF_path",
    "align2_tw_delta", "align2_tw_conf", "align2_own_delta", "align2_own_conf",
    "align2_tw_est", "align2_own_est",
    "egfdu_drift", "z_minus_egfdu", "nb_dip_x", "nb_dip_y", "nb_plane_dF",
]
