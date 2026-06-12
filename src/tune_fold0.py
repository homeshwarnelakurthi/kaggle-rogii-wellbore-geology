"""Quick fold-0 comparison of LGB configs on the v2 features."""
import sys, os, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURE_COLS

df = pd.read_parquet("features_train.parquet")
X = df[FEATURE_COLS].values.astype(np.float32)
y = df["target_dF"].values.astype(np.float32)
groups = df["well"].values
tvt_true = df["tvt_true"].values
tvt_flat = df["tvt_flat"].values

tr, va = next(GroupKFold(n_splits=5).split(X, y, groups))
base = dict(objective="regression", metric="rmse", n_jobs=-1, verbose=-1,
            subsample=0.7, subsample_freq=1)
configs = {
    "v2 (63/0.03/mc200)": dict(num_leaves=63, learning_rate=0.03,
                               n_estimators=3000, min_child_samples=200,
                               colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=5.0),
    "big (255/0.02/mc400)": dict(num_leaves=255, learning_rate=0.02,
                                 n_estimators=8000, min_child_samples=400,
                                 colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=10.0),
    "mid (127/0.02/mc300)": dict(num_leaves=127, learning_rate=0.02,
                                 n_estimators=8000, min_child_samples=300,
                                 colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=5.0),
}
for name, cfg in configs.items():
    t0 = time.time()
    m = lgb.LGBMRegressor(**base, **cfg)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
          callbacks=[lgb.early_stopping(150, verbose=False)])
    p = m.predict(X[va])
    r = np.sqrt(np.mean((tvt_flat[va] + p - tvt_true[va]) ** 2))
    print(f"{name}: iters={m.best_iteration_} TVT rmse={r:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
