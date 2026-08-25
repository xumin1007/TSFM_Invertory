"""多分位点 LightGBM。对应 docs/06_model_hyperparams.md §6.1。

**为何是 21 个分位点而非 2 个**：TSFM 走「日级分位网格 → 数值卷积 → 周/月」
这条路径，而两分位数 Gamma 在零膨胀目标上已实测失效（q85 MAE 0.298 vs
数值卷积 0.125，07 §2.3.1）。要让比较落在模型本身而非聚合方式上，GBDT
必须产出**同一形状**的输出。代价是每个 origin 训 21 个模型而非 2 个。

**为何特征只用序列自身历史**：Chronos-2 零样本只看目标序列的 context，
不使用任何外生特征。若 GBDT 用上静态属性、类别、在订量等结构化特征，
比较的就不再是「模型」而是「信息集」。故本类默认只用滞后/滚动统计，
与 TSFM 的信息集对齐；结构化特征作为**单独登记的臂**（gbdt-rich）而非
默认，且其结论须与 gbdt-lean 分开报告。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..conventions import SEED_BASE

# 与 f2d.models.chronos.NATIVE_LEVELS 保持一致，使两者输出同形
GRID_LEVELS = np.array([
    .01, .05, .10, .15, .20, .25, .30, .35, .40, .45, .50,
    .55, .60, .65, .70, .75, .80, .85, .90, .95, .99])

BASE_PARAMS = dict(
    objective="quantile", n_estimators=300, learning_rate=0.05,
    num_leaves=31, min_child_samples=40, feature_fraction=0.9,
    bagging_fraction=0.9, bagging_freq=1, lambda_l1=0.1, lambda_l2=1.0,
    max_bin=255, seed=SEED_BASE, deterministic=True,
    force_row_wise=True, verbose=-1)

# 与 TSFM 信息集对齐：只用目标序列自身的滞后/滚动统计
LEAN_FEATURES = ["lag1", "lag7", "roll7", "roll28", "roll7_std",
                 "roll28_std", "zero_rate_28", "days_since_nonzero", "dow"]


def make_lean_features(daily: pd.DataFrame, value_col: str = "y",
                       group_col: str = "series_id") -> pd.DataFrame:
    """日级 lean 特征。全部严格来自 origin 之前（shift(1) 保证）。"""
    d = daily.sort_values([group_col, "d"]).copy()
    g = d.groupby(group_col)[value_col]
    s1 = g.shift(1)

    def _roll(w, fn):
        return (s1.rolling(w, min_periods=1).agg(fn)
                .reset_index(level=0, drop=True))

    d["lag1"] = s1
    d["lag7"] = g.shift(7)
    d["roll7"] = _roll(7, "mean")
    d["roll28"] = _roll(28, "mean")
    d["roll7_std"] = _roll(7, "std")
    d["roll28_std"] = _roll(28, "std")
    d["zero_rate_28"] = (s1.eq(0).rolling(28, min_periods=1).mean()
                         .reset_index(level=0, drop=True))
    nz = d[value_col].gt(0)
    d["days_since_nonzero"] = (
        d.groupby(group_col).cumcount()
        - nz.groupby(d[group_col]).cummax().mul(0).add(
            nz.groupby(d[group_col]).cumsum().where(nz).ffill().fillna(0)) * 0)
    # 简化：距上次非零的天数，用分组内 forward-fill 的最后非零位置
    last_nz = d.assign(_i=d.groupby(group_col).cumcount()).assign(
        _p=lambda x: x["_i"].where(nz.shift(1, fill_value=False)))
    d["days_since_nonzero"] = (last_nz["_i"]
                               - last_nz.groupby(group_col)["_p"].ffill()).fillna(999)
    d["dow"] = d["d"].dt.dayofweek
    return d


class QuantileGridGBDT:
    """在 GRID_LEVELS 的每个分位点各训一个 LightGBM。"""

    def __init__(self, features: list[str] | None = None,
                 levels: np.ndarray = GRID_LEVELS, params: dict | None = None):
        self.features = list(features or LEAN_FEATURES)
        self.levels = np.asarray(levels, float)
        self.params = {**BASE_PARAMS, **(params or {})}
        self.models: list = []

    def fit(self, train: pd.DataFrame, y_col: str = "y"):
        import lightgbm as lgb
        X, y = train[self.features], train[y_col].to_numpy(float)
        self.models = []
        for tau in self.levels:
            m = lgb.LGBMRegressor(**{**self.params, "alpha": float(tau)})
            m.fit(X, y)
            self.models.append(m)
        return self

    def predict_grid(self, df: pd.DataFrame) -> np.ndarray:
        """返回 (n_rows, n_levels)，已做非负截断与单调重排。"""
        if not self.models:
            raise RuntimeError("尚未 fit")
        X = df[self.features]
        g = np.column_stack([m.predict(X) for m in self.models])
        return np.maximum.accumulate(np.clip(g, 0.0, None), axis=1)
