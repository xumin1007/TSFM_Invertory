"""LightGBM 分位数回归。对应 docs/06_model_hyperparams.md §6.1。

tau=0.50 与 tau=0.85 分别训练两个独立模型，不用共享树的多输出头。
类别特征走 f2d.encoding 的冻结词表；缺失值原样传入 NaN，不填充。
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..conventions import SEED_BASE, TAU_MAIN, TAU_UPPER

BASE_PARAMS = dict(
    objective="quantile",
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l1=0.1,
    lambda_l2=1.0,
    max_bin=255,
    seed=SEED_BASE,
    deterministic=True,
    force_row_wise=True,
    verbose=-1,
)


class QuantileGBDT:
    """两个 tau 各一个模型。early stopping 在显式验证窗上进行。"""

    def __init__(self, num_features: list[str], cat_features: list[str],
                 params: dict | None = None, taus=(TAU_MAIN, TAU_UPPER)):
        self.num_features = list(num_features)
        self.cat_features = list(cat_features)
        self.features = self.num_features + self.cat_features
        self.params = {**BASE_PARAMS, **(params or {})}
        self.taus = taus
        self.models: dict[float, lgb.LGBMRegressor] = {}
        self.best_iters: dict[float, int] = {}

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.features]

    def fit(self, train: pd.DataFrame, y_col: str, w_col: str | None = None,
            valid: pd.DataFrame | None = None, early_stopping_rounds: int = 100):
        Xtr, ytr = self._X(train), train[y_col].to_numpy(float)
        wtr = train[w_col].to_numpy(float) if w_col else None
        for tau in self.taus:
            m = lgb.LGBMRegressor(**{**self.params, "alpha": tau})
            kw = {}
            if valid is not None and len(valid):
                kw["eval_set"] = [(self._X(valid), valid[y_col].to_numpy(float))]
                kw["eval_sample_weight"] = (
                    [valid[w_col].to_numpy(float)] if w_col else None)
                kw["callbacks"] = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
            m.fit(Xtr, ytr, sample_weight=wtr,
                  categorical_feature=self.cat_features or "auto", **kw)
            self.models[tau] = m
            self.best_iters[tau] = int(getattr(m, "best_iteration_", 0) or self.params["n_estimators"])
        return self

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = self._X(df)
        return (self.models[self.taus[0]].predict(X),
                self.models[self.taus[1]].predict(X))
