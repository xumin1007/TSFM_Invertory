"""FreshRetailNet-LT 最小周级面板。对应 spec 01 + configs/fresh_lt.yaml。

本模块目前只支撑 MDE 估计与点预测基线；恢复模型、TSFM、协变量尚未接入。
一个样本 = (series_id, origin)，origin 为周一，目标为 [origin, origin+6] 的
observed_sales_daily 之和。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ARTIFACT_DIR, DATA_DIR

RAW = DATA_DIR / "external" / "freshretailnet_lt" / "data"
ART = ARTIFACT_DIR / "fresh_lt"
CACHE = ART / "_cache"

COLS = ["dt", "store_id", "product_id", "sale_amount"]


def load_weekly(use_cache: bool = True) -> pd.DataFrame:
    """读取 train+eval，按 (series_id, 周一起点) 聚合日销量。

    train 与 eval 在时间上不重叠（train 至 2025-07-13，eval 2025-07-14..20），
    故拼接不构成泄漏；split 由 origin 日期决定，见 configs/fresh_lt.yaml。
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / "weekly.parquet"
    if use_cache and cp.exists():
        return pd.read_parquet(cp)

    frames = []
    for name in ("train", "eval"):
        d = pd.read_parquet(RAW / f"{name}.parquet", columns=COLS)
        d["dt"] = pd.to_datetime(d["dt"])
        d["source"] = name
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["series_id"] = d["store_id"].astype(str) + "_" + d["product_id"].astype(str)
    # 周一起点：dt 所在周的周一
    d["origin"] = d["dt"] - pd.to_timedelta(d["dt"].dt.weekday, unit="D")

    w = (d.groupby(["series_id", "origin"], as_index=False)
          .agg(y=("sale_amount", "sum"), n_days=("sale_amount", "size")))
    w = w.sort_values(["series_id", "origin"]).reset_index(drop=True)
    w.to_parquet(cp, index=False)
    return w


def add_lags(w: pd.DataFrame) -> pd.DataFrame:
    """严格只用 origin 之前的周。shift(1) 保证不含当周。"""
    g = w.groupby("series_id")["y"]
    w = w.copy()
    w["lag1"] = g.shift(1)
    w["roll4"] = g.shift(1).rolling(4, min_periods=1).mean().reset_index(level=0, drop=True)
    w["roll8"] = g.shift(1).rolling(8, min_periods=1).mean().reset_index(level=0, drop=True)
    w["weeks_observed"] = g.transform(lambda s: np.arange(len(s), dtype=float))
    return w
