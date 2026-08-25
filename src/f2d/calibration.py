"""残差分位数校准器。对应 docs/00_global_conventions.md §6。

把点预测转为 (q50, q85)。加性残差、逐序列/类别/全局三层回退、
视界匹配、expanding 窗、每 origin 重拟合。校准器不加权。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .conventions import CALIB_N_MIN, QUANTILE_METHOD, TAU_MAIN, TAU_UPPER


class ResidualCalibrator:
    """三层回退（L0 series / L1 category / L2 global）的经验残差分位数。

    残差池的三条硬约束（§6.2）：
      1. 时点隔离：只纳入 target_end < origin 的残差（由调用方筛选后传入，此处再断言）
      2. 视界匹配：只纳入相同预测视界的残差
      3. expanding 窗：每 origin 重新拟合
    """

    def __init__(self, n_min: int = CALIB_N_MIN, taus: tuple[float, ...] = (TAU_MAIN, TAU_UPPER)):
        self.n_min = n_min
        self.taus = taus
        self._q: dict[str, dict] = {}
        self.horizon = None

    def fit(
        self,
        residuals: pd.DataFrame,
        origin,
        horizon,
        series_col: str = "series_id",
        cat_col: str = "category",
        resid_col: str = "residual",
        end_col: str = "target_end",
    ) -> "ResidualCalibrator":
        r = residuals
        # 约束 1：时点隔离（可断言，不靠约定）
        late = r[end_col] >= origin
        if bool(late.any()):
            raise ValueError(
                f"校准器泄漏：{int(late.sum())} 条残差的 target_end >= origin={origin}"
            )
        # 约束 2：视界匹配
        if "horizon" in r.columns:
            r = r[r["horizon"] == horizon]
        self.horizon = horizon

        vals = r[resid_col].to_numpy(float)
        self._q = {"L2": {t: self._quantile(vals, t) for t in self.taus}, "L1": {}, "L0": {}}
        if vals.size < self.n_min:
            raise ValueError(
                f"L2 全局残差仅 {vals.size} < n_min={self.n_min}；校准窗过短，属切分配置错误"
            )
        for key, grp in r.groupby(cat_col, observed=True)[resid_col]:
            a = grp.to_numpy(float)
            if a.size >= self.n_min:
                self._q["L1"][key] = {t: self._quantile(a, t) for t in self.taus}
        for key, grp in r.groupby(series_col, observed=True)[resid_col]:
            a = grp.to_numpy(float)
            if a.size >= self.n_min:
                self._q["L0"][key] = {t: self._quantile(a, t) for t in self.taus}
        return self

    @staticmethod
    def _quantile(a: np.ndarray, tau: float) -> float:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return 0.0
        return float(np.quantile(a, tau, method=QUANTILE_METHOD))

    def predict(
        self, point: np.ndarray, series: pd.Series, category: pd.Series
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (q50, q85, calib_level)。level 逐行记录，供事后审计。"""
        point = np.asarray(point, float)
        n = point.size
        out = {t: np.empty(n) for t in self.taus}
        level = np.empty(n, dtype=object)
        s = series.to_numpy()
        c = category.to_numpy()
        for i in range(n):
            if s[i] in self._q["L0"]:
                src, lv = self._q["L0"][s[i]], "L0"
            elif c[i] in self._q["L1"]:
                src, lv = self._q["L1"][c[i]], "L1"
            else:
                src, lv = self._q["L2"], "L2"
            level[i] = lv
            for t in self.taus:
                out[t][i] = point[i] + src[t]
        return out[self.taus[0]], out[self.taus[1]], level

    @property
    def level_sizes(self) -> dict[str, int]:
        return {"L0": len(self._q["L0"]), "L1": len(self._q["L1"]), "L2": 1}
