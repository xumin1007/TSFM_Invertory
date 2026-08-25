"""OSA 日级面板。对应 docs/implementation_specs/03_osa.md + configs/osa.yaml。

一个样本 = (store_id, sku, date)。目标为 total_sales_units。

OSA 特殊性：
  - 日期编码为整数 YYYYMMDD，须显式解析
  - 面板密度 ~48%，缺失是真正的状态未知（非零销量）
  - gap_decision = EXCLUDE_NOT_ZERO_FILL：缺失日保持缺失，
    对应 origin 排除出评估（00_global_conventions.md §8.2）
  - 100 条序列，统计功效受限：MECHANISM_DEMONSTRATOR_ONLY
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ARTIFACT_DIR, DATA_DIR

RAW = DATA_DIR / "external" / "osa-data"
ART = ARTIFACT_DIR / "osa"
CACHE = ART / "_cache"


def load_raw(use_cache: bool = True) -> pd.DataFrame:
    """读取 osa_raw_data.csv，解析整数日期，返回日级面板。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / "daily_raw.parquet"
    if use_cache and cp.exists():
        return pd.read_parquet(cp)

    df = pd.read_csv(RAW / "osa_raw_data.csv")
    df["d"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    assert df["d"].notna().all(), "日期解析失败"
    df["series_id"] = df["store_id"].astype(str) + "_" + df["sku"].astype(str)
    df = df.sort_values(["series_id", "d"]).reset_index(drop=True)
    df.to_parquet(cp, index=False)
    return df


def raw_audit(df: pd.DataFrame) -> dict:
    """步骤 1：原始审计。"""
    return {
        "n_rows": int(len(df)),
        "n_series": int(df.series_id.nunique()),
        "date_range": [str(df.d.min().date()), str(df.d.max().date())],
        "calendar_days": int((df.d.max() - df.d.min()).days + 1),
        "pk_duplicates": int(df.duplicated(["series_id", "d"]).sum()),
        "negative_sales": int((df.total_sales_units < 0).sum()),
        "negative_inventory": int((df.on_hand_inventory_units < 0).sum()),
        "zero_sales_share": round(float((df.total_sales_units == 0).mean()), 4),
        "per_series_obs": {
            "min": int(df.groupby("series_id").size().min()),
            "median": int(df.groupby("series_id").size().median()),
            "max": int(df.groupby("series_id").size().max()),
        },
        "columns": list(df.columns),
    }


def build_daily_panel(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """构建日级面板。不补零（EXCLUDE_NOT_ZERO_FILL）。

    返回 (panel, audit)。panel 列：series_id, d, y, store_id, sku,
    product_category, on_hand_inventory_units, ...
    """
    panel = df.copy()
    panel["y"] = panel["total_sales_units"].astype(float)

    per_series = panel.groupby("series_id").agg(
        first_date=("d", "min"), last_date=("d", "max"), n_obs=("d", "size"))
    per_series["span_days"] = (per_series.last_date - per_series.first_date).dt.days + 1
    per_series["density"] = per_series.n_obs / per_series.span_days

    audit = {
        "n_rows": int(len(panel)),
        "n_series": int(panel.series_id.nunique()),
        "gap_decision": "EXCLUDE_NOT_ZERO_FILL",
        "density_median": round(float(per_series.density.median()), 4),
        "zero_share": round(float((panel.y == 0).mean()), 4),
        "mean_obs_per_series": round(float(per_series.n_obs.mean()), 1),
    }
    return panel, audit


def aggregate_to_weekly(panel: pd.DataFrame) -> pd.DataFrame:
    """聚合到周级（周一起点）。只保留完整周（7 天全部有观测）。

    由于 gap_decision=EXCLUDE，不补零，故"完整周"要求该序列在目标窗
    7 天中每天都有原始观测记录。
    """
    df = panel.copy()
    df["origin"] = df.d - pd.to_timedelta(df.d.dt.weekday, unit="D")

    g = df.groupby(["series_id", "origin"], as_index=False).agg(
        y=("y", "sum"), n_days=("y", "size"))
    complete = g[g.n_days == 7].drop(columns=["n_days"]).reset_index(drop=True)
    complete.attrs["n_excluded_incomplete"] = int(len(g) - len(complete))
    return complete.sort_values(["series_id", "origin"]).reset_index(drop=True)


def write_audit(audit: dict, path: Path | None = None) -> Path:
    p = path or (ART / "raw_audit.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str),
                 encoding="utf-8")
    return p
