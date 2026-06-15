# ============================================
# LGB INFERENCE CELL — numpy 2.x fixed, real extract_well API
# Paste AFTER the sp45+fleongg blend cell, BEFORE exact-match recovery.
# NOTE vs original spec: extract_well returns ONE DataFrame (cols incl
#   'id' and 'tvt_flat'), NOT a 3-tuple. Reconstruction is tvt_flat + dF,
#   which equals last_tvt - (Z - Z_anchor) + dF. Dataset path auto-detects.
# ============================================
import json as _j, sys as _sy, glob as _gl
import numpy as _np2, pandas as _pd2
import lightgbm as _lgb2
from pathlib import Path as _P2

# locate the uploaded weights dataset (path may vary; auto-detect on features.json)
_cand = list(_P2("/kaggle/input").glob("**/features.json"))
_LDIR = (_P2("/kaggle/input/rogii-lgb-weights-v1") if (_P2("/kaggle/input/rogii-lgb-weights-v1")/"features.json").exists()
         else (_cand[0].parent if _cand else _P2("/kaggle/input/rogii-lgb-weights-v1")))
_sy.path.insert(0, str(_LDIR))
print(f"LGB weights dir: {_LDIR}")

try:
    from features import extract_well, load_context as _lc

    with open(_LDIR / "features.json") as _f:
        _fc = _j.load(_f)
    _FN = _fc["features"]
    _FM = _np2.array(_fc["scaler_mean"], dtype=_np2.float32)
    _FS = _np2.array(_fc["scaler_scale"], dtype=_np2.float32)
    _cx = _lc(str(_LDIR / "knn_context.npz"))

    _ms = []
    for _fd in range(5):
        for _sx in ["", "_s1"]:
            _pp = _LDIR / f"lgb_fold{_fd}{_sx}.txt"
            if _pp.exists():
                _ms.append(_lgb2.Booster(model_file=str(_pp)))
    print(f"LGB: {len(_ms)} models loaded")

    _DD = _P2("/kaggle/input/competitions/rogii-wellbore-geology-prediction")
    _tids = [f.stem.replace("__horizontal_well", "")
             for f in sorted((_DD / "test").glob("*__horizontal_well.csv"))]

    _lp = {}
    for _wid in _tids:
        try:
            _hw = _pd2.read_csv(_DD / "test" / f"{_wid}__horizontal_well.csv")
            _tw = _pd2.read_csv(_DD / "test" / f"{_wid}__typewell.csv")
            _out = extract_well(_hw, _tw, _wid, ctx=_cx,
                                exclude_well_idx=None, is_train=False)
            if _out is None or len(_out) == 0:
                print(f"  {_wid}: empty"); continue
            _X = ((_out[_FN].fillna(0).values.astype(_np2.float32) - _FM)
                  / (_FS + 1e-9))
            _dF = _np2.mean([m.predict(_X) for m in _ms], axis=0)
            _tv = _out["tvt_flat"].values + _dF          # = last_tvt - (Z - Z_anchor) + dF
            for _id, _v in zip(_out["id"].values, _tv):
                _lp[_id] = float(_v)
            print(f"  {_wid}: {len(_tv)} rows [{_tv.min():.1f},{_tv.max():.1f}]")
        except Exception as _e:
            print(f"  {_wid}: {_e}")

    _LOK = len(_lp) > 0
    print(f"LGB OK={_LOK} | {len(_lp)} predictions")

except Exception as _e:
    import traceback as _tb; _tb.print_exc()
    print(f"LGB SKIPPED: {_e}")
    _LOK = False
    _lp = {}

# 3-way blend — only runs if LGB actually produced predictions
if _LOK:
    _WD = _P2("/kaggle/working")
    _cs = _pd2.read_csv(_WD / "submission.csv")
    _ls = _pd2.DataFrame([{"id": k, "tvt_lgb": v} for k, v in _lp.items()])
    _mg = _cs.merge(_ls, on="id", how="left")
    _mk = _mg.tvt_lgb.notna()
    print(f"matched {_mk.sum()}/{len(_mg)} ids for blend")
    _mg.loc[_mk, "tvt"] = (0.85 * _mg.loc[_mk, "tvt"].values
                           + 0.15 * _mg.loc[_mk, "tvt_lgb"].values)
    _mg[["id", "tvt"]].to_csv(_WD / "submission.csv", index=False)
    print(f"3-way blend: mean={_mg.tvt.mean():.2f} std={_mg.tvt.std():.2f}")
else:
    print("LGB failed — sp45+fleongg kept as final")
