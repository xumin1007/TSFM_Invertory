"""点预测基线。分位数由 f2d.calibration 的残差校准器产生（quantile_source="calibrated"）。"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PointBaseline:
    """一个点预测基线 = 面板上的一列。刻意保持极简，可审计。"""

    def __init__(self, model_id: str, column: str):
        self.model_id = model_id
        self.column = column

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        p = df[self.column].to_numpy(float)
        # 无历史时退化为 0（该行同时会被记为 calib L1/L2 与 fallback）
        return np.where(np.isfinite(p), p, 0.0)


REGISTRY = {
    "naive-last": PointBaseline("naive-last", "sales_lag1"),
    "roll-mean-3": PointBaseline("roll-mean-3", "sales_roll3"),
    "roll-mean-6": PointBaseline("roll-mean-6", "sales_roll6"),
}
