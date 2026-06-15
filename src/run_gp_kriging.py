"""Run the standalone GP kriging pipeline end-to-end:
  Step 1  load + analyze (b_well, correlations, baselines)
  Step 4  GroupKFold(5) OOF comparing anchor/GR configs
  Step 5  final fit on all 773 wells -> test submission + per-well stats
  Step 6  save artifacts (gp pkl, scaler pkl, stats json) + RMSE histogram
"""
import os, sys, json, time, glob
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gp_kriging as gk

ROOT = gk.ROOT
SUBDIR = os.path.join(ROOT, "submissions")
MODDIR = os.path.join(ROOT, "models")
os.makedirs(SUBDIR, exist_ok=True)

CONFIGS = ["A", "A_gr", "B", "B_gr"]   # compared on OOF


def load_all_train():
    ids = gk.list_wells("train")
    wells = []
    t0 = time.time()
    for i, w in enumerate(ids):
        wells.append(gk.load_well(w, "train"))
        if (i + 1) % 200 == 0:
            print(f"  loaded {i+1}/{len(ids)}  ({time.time()-t0:.0f}s)")
    print(f"  loaded {len(wells)} train wells in {time.time()-t0:.0f}s")
    return wells


def rep_arrays(wells):
    X_gp = np.array([[w["rep_x"], w["rep_y"]] for w in wells])
    y_gp = np.array([float(np.median(w["TVT"][w["ev"]][np.isfinite(w["TVT"][w["ev"]])]))
                     for w in wells])
    return X_gp, y_gp


# ----------------------------------------------------------------------------- Step 1
def step1(wells):
    print("\n" + "=" * 70)
    print("STEP 1 — data analysis (773 wells)")
    print("=" * 70)
    rep_x, rep_y, rep_eff, rep_z, b_well = [], [], [], [], []
    naive_sq, naive_n = [], []
    for w in wells:
        ev = w["ev"]
        t = w["TVT"][ev]; Z = w["Z"][ev]
        m = np.isfinite(t) & np.isfinite(Z)
        if m.sum() < 5 or len(w["kn_eff"]) < 5:
            continue
        ev_eff = t[m] + Z[m]
        rep_x.append(w["rep_x"]); rep_y.append(w["rep_y"])
        rep_eff.append(np.median(ev_eff)); rep_z.append(np.median(Z[m]))
        b_well.append(np.median(w["kn_eff"]) - np.median(ev_eff))
        last_eff = w["TVT_input"][w["ps"] - 1] + w["Z"][w["ps"] - 1]
        err = (last_eff - Z[m]) - t[m]
        naive_sq.append((err ** 2).sum()); naive_n.append(m.sum())
    b = np.array(b_well)
    print(f"  b_well := median(known_eff)-median(eval_eff)  [conflates F(x,y) drift]")
    print(f"    mean={b.mean():.3f}  std={b.std():.3f}  |b|>5ft in {(np.abs(b)>5).sum()} wells")
    re = np.array(rep_eff)
    print(f"  corr(rep_eff,X)={np.corrcoef(re,rep_x)[0,1]:.3f}  "
          f"corr(rep_eff,Y)={np.corrcoef(re,rep_y)[0,1]:.3f}  "
          f"corr(rep_eff,Z)={np.corrcoef(re,rep_z)[0,1]:.3f}")
    ns = np.array(naive_sq); nn = np.array(naive_n)
    print(f"  NAIVE (F=const=last known eff) pooled RMSE = {np.sqrt(ns.sum()/nn.sum()):.3f} ft")
    return dict(b_well_mean=float(b.mean()), b_well_std=float(b.std()))


# ----------------------------------------------------------------------------- Step 4
def step4(wells):
    print("\n" + "=" * 70)
    print("STEP 4 — GroupKFold(5) OOF")
    print("=" * 70)
    X_gp, y_gp = rep_arrays(wells)
    groups = np.arange(len(wells))
    gkf = GroupKFold(n_splits=5)
    # accumulate pooled sq-error and per-well rmse per config
    pooled_sq = {c: 0.0 for c in CONFIGS}
    pooled_n = {c: 0 for c in CONFIGS}
    perwell = {c: [] for c in CONFIGS}
    well_names = []
    well_meta = []   # (well, rep_x, rep_y, best-config rmse) filled for primary
    t0 = time.time()
    for fold, (tr, va) in enumerate(gkf.split(X_gp, y_gp, groups)):
        gp, scaler = gk.build_gp(X_gp[tr], y_gp[tr])
        for j in va:
            d = wells[j]
            t, m = gk.well_truth(d)
            if m.sum() < 5:
                continue
            preds = gk.predict_configs(d, gp, scaler, CONFIGS)
            well_names.append(d["well"])
            row = dict(well=d["well"], rep_x=d["rep_x"], rep_y=d["rep_y"], n=int(m.sum()))
            for c in CONFIGS:
                p = preds[c][m]
                sq = float(np.sum((p - t[m]) ** 2))
                pooled_sq[c] += sq; pooled_n[c] += int(m.sum())
                rw = float(np.sqrt(sq / m.sum()))
                perwell[c].append(rw)
                row[f"rmse_{c}"] = rw
            well_meta.append(row)
        print(f"  fold {fold}: {len(va)} val wells, kernel={gp.kernel_}  "
              f"({time.time()-t0:.0f}s)")
    print("\n  config |  pooled RMSE | per-well median | per-well p90")
    print("  -------+-------------+-----------------+-------------")
    results = {}
    for c in CONFIGS:
        pr = np.sqrt(pooled_sq[c] / pooled_n[c])
        pw = np.array(perwell[c])
        results[c] = dict(pooled=float(pr), med=float(np.median(pw)),
                          p90=float(np.percentile(pw, 90)))
        print(f"  {c:6s} |  {pr:8.3f}   |   {np.median(pw):8.3f}      |  {np.percentile(pw,90):7.2f}")
    best = min(CONFIGS, key=lambda c: results[c]["pooled"])
    print(f"\n  >>> best config by pooled RMSE: {best}  ({results[best]['pooled']:.3f} ft)")

    meta = pd.DataFrame(well_meta)
    meta["rmse_best"] = meta[f"rmse_{best}"]
    worst = meta.nlargest(10, "rmse_best")[["well", "rep_x", "rep_y", "rmse_best", "n"]]
    bestw = meta.nsmallest(10, "rmse_best")[["well", "rmse_best", "n"]]
    print("\n  WORST 10 wells (config %s):" % best)
    for _, r in worst.iterrows():
        print(f"    {r.well}  rmse={r.rmse_best:8.2f}  (x={r.rep_x:.0f}, y={r.rep_y:.0f}, n={int(r.n)})")
    print("  BEST 10 wells:")
    for _, r in bestw.iterrows():
        print(f"    {r.well}  rmse={r.rmse_best:6.3f}  (n={int(r.n)})")

    # isolation check: are worst wells spatially isolated?
    from scipy.spatial import cKDTree
    allxy = meta[["rep_x", "rep_y"]].values
    tree = cKDTree(allxy)
    d2, _ = tree.query(allxy, k=2)
    meta["nn_dist"] = d2[:, 1]
    worst_nn = meta.nlargest(int(0.1 * len(meta)), "rmse_best")["nn_dist"].median()
    all_nn = meta["nn_dist"].median()
    print(f"\n  worst-decile median nearest-well dist = {worst_nn:.0f} ft  "
          f"(all wells median = {all_nn:.0f} ft)  -> "
          f"{'ISOLATED' if worst_nn > 1.5*all_nn else 'NOT specially isolated'}")

    meta.to_csv(os.path.join(SUBDIR, "gp_oof_per_well.csv"), index=False)
    # histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(np.clip(meta["rmse_best"], 0, 60), bins=60, color="#4c72b0")
        ax.set_xlabel("per-well RMSE (ft, clipped 60)"); ax.set_ylabel("wells")
        ax.set_title(f"GP kriging OOF per-well RMSE — config {best}\n"
                     f"pooled {results[best]['pooled']:.2f}  median {results[best]['med']:.2f}")
        fig.tight_layout()
        fig.savefig(os.path.join(SUBDIR, "gp_per_well_rmse_hist.png"), dpi=110)
        print(f"  saved histogram -> submissions/gp_per_well_rmse_hist.png")
    except Exception as e:
        print(f"  (histogram skipped: {e})")
    return results, best


# ----------------------------------------------------------------------------- Step 5/6
def step5_6(wells, best_cfg, oof_results, step1_stats):
    print("\n" + "=" * 70)
    print("STEP 5/6 — final fit, test submission, artifacts")
    print("=" * 70)
    X_gp, y_gp = rep_arrays(wells)
    gp, scaler = gk.build_gp(X_gp, y_gp)
    print(f"  final kernel: {gp.kernel_}")

    test_ids = gk.list_wells("test")
    print(f"  test wells: {len(test_ids)}")
    rows = []
    stats = []
    for w in test_ids:
        d = gk.load_well(w, "test")
        preds = gk.predict_configs(d, gp, scaler, [best_cfg])[best_cfg]
        # GP uncertainty at eval points
        ev_xy = scaler.transform(np.column_stack([d["X"][d["ev"]], d["Y"][d["ev"]]]))
        _, sd = gp.predict(ev_xy, return_std=True)
        rep_xy = scaler.transform([[d["rep_x"], d["rep_y"]]])
        gp_rep = float(gp.predict(rep_xy)[0])
        b_well = float(np.median(d["kn_eff"])) - gp_rep
        for idx, p in zip(d["ids"], preds):
            rows.append((idx, float(p)))
        stats.append(dict(well_id=w, rep_x=d["rep_x"], rep_y=d["rep_y"],
                          gp_pred=gp_rep, b_well=b_well, n_eval=int(len(d["ev"])),
                          tvt_mean=float(np.mean(preds)), tvt_std=float(np.std(preds)),
                          gp_uncert_mean=float(np.mean(sd))))

    pred_map = dict(rows)
    sample = pd.read_csv(os.path.join(gk.DATA, "sample_submission.csv"))
    sample["tvt"] = sample["id"].map(pred_map).fillna(0.0)
    sub_path = os.path.join(SUBDIR, "gp_kriging_submission.csv")
    sample.to_csv(sub_path, index=False)
    n_filled = sample["id"].isin(pred_map).sum()
    print(f"  submission: {sub_path}  ({n_filled}/{len(sample)} ids predicted)")

    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(SUBDIR, "gp_per_well_stats.csv"), index=False)
    print(f"  per-well stats: submissions/gp_per_well_stats.csv")
    print("  test-well GP uncertainty (mean std at eval pts):")
    for _, r in stats_df.iterrows():
        print(f"    {r.well_id}  gp_uncert={r.gp_uncert_mean:6.2f}  "
              f"b_well={r.b_well:8.2f}  tvt[{r.tvt_mean:.0f}+/-{r.tvt_std:.0f}]")

    # artifacts
    import joblib
    joblib.dump(gp, os.path.join(MODDIR, "gp_kriging_model.pkl"))
    joblib.dump(scaler, os.path.join(MODDIR, "gp_scaler.pkl"))
    gp_stats = dict(
        oof_results=oof_results, best_config=best_cfg,
        oof_rmse=oof_results[best_cfg]["pooled"],
        kernel_params=str(gp.kernel_),
        training_wells=len(wells),
        b_well_mean=step1_stats["b_well_mean"],
        b_well_std=step1_stats["b_well_std"],
        n_restarts=gk.N_RESTARTS,
    )
    with open(os.path.join(MODDIR, "gp_stats.json"), "w") as f:
        json.dump(gp_stats, f, indent=2)
    print(f"  artifacts: models/gp_kriging_model.pkl, gp_scaler.pkl, gp_stats.json")
    return gp_stats


def main():
    t0 = time.time()
    wells = load_all_train()
    s1 = step1(wells)
    oof, best = step4(wells)
    gp_stats = step5_6(wells, best, oof, s1)

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"  OOF pooled RMSE ({best}): {oof[best]['pooled']:.3f} ft")
    print(f"    vs naive flat baseline ~107.5 ft")
    print(f"    vs prior own-model OOF 11.08 ft")
    print(f"    vs current best LB 7.535 (blend of public notebooks)")
    print(f"  total time: {time.time()-t0:.0f}s")
    print(f"  outputs in submissions/ and models/")


if __name__ == "__main__":
    main()
