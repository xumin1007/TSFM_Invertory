"""Zhao 日级路径的测试。

这些函数封装了本轮反复踩过的构造错误，故测试重点在**前提校验**而非happy path：
  - 补零只在生命期内（生命期外补零 = 凭空造数据）
  - 「无记录=零销量」的前提必须由数据支持（交易日志），否则抛错
  - 聚合只保留完整周期
  - 经验分位数必须用 inverted_cdf（线性插值会抹平零原子）
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f2d.datasets.zhao import (add_lag_features, aggregate_to_period,
                               build_daily_panel, empirical_quantile_grid)


def _raw(rows, zero_rows: int = 0):
    """构造最小的 raw dict；rows = [(sku, 'YYYY-MM-DD', qty), ...]"""
    df = pd.DataFrame(rows, columns=["sku_ID", "date", "quantity"])
    if zero_rows:
        extra = pd.DataFrame({"sku_ID": ["z"] * zero_rows,
                              "date": ["2019-01-01"] * zero_rows,
                              "quantity": [0] * zero_rows})
        df = pd.concat([df, extra], ignore_index=True)
    return {"sales": df}


def _span(sku: str, start: str, days: int, step: int = 10):
    """每 step 天一笔交易，跨度 days 天。"""
    d0 = pd.Timestamp(start)
    return [(sku, str((d0 + pd.Timedelta(days=k)).date()), 1.0)
            for k in range(0, days, step)] + \
           [(sku, str((d0 + pd.Timedelta(days=days - 1)).date()), 2.0)]


# --------------------------------------------------------------------------
# 前提校验
# --------------------------------------------------------------------------

def test_rejects_when_zeros_are_recorded():
    """若销售表显式记录零（如 OSA 那样），「无记录=零销量」不成立，必须抛错。

    OSA 的日表有 48.45% 的行 total_sales_units==0；若对那类数据补零，
    会凭空造出大量假数据。
    """
    rows = _span("a", "2019-01-01", 200)
    raw = _raw(rows, zero_rows=len(rows))      # 50% 零行
    with pytest.raises(ValueError, match="交易日志"):
        build_daily_panel(raw)


def test_accepts_transaction_log():
    """零值占比极低时（Zhao 实测 0.11%）视为交易日志，正常构建。

    step=1 使交易行足够多（200 行），1 个零行占 0.5% < 1% 阈值。
    """
    raw = _raw(_span("a", "2019-01-01", 200, step=1), zero_rows=1)
    panel, audit = build_daily_panel(raw)
    assert audit["zero_qty_share"] < 0.01
    assert len(panel) > 0


# --------------------------------------------------------------------------
# 生命期补零
# --------------------------------------------------------------------------

def test_zero_fill_confined_to_lifetime():
    """补零范围必须恰为 [首次交易, 末次交易]，不得外扩。"""
    raw = _raw(_span("a", "2019-03-01", 150))
    panel, _ = build_daily_panel(raw, min_lifetime_days=100)
    assert panel.d.min() == pd.Timestamp("2019-03-01")
    assert panel.d.max() == pd.Timestamp("2019-03-01") + pd.Timedelta(days=149)
    assert len(panel) == 150                    # 生命期内每天一行


def test_short_lifetime_series_dropped():
    raw = _raw(_span("short", "2019-01-01", 50) + _span("long", "2019-01-01", 200))
    panel, audit = build_daily_panel(raw, min_lifetime_days=100)
    assert set(panel.sku_ID) == {"long"}
    assert audit["n_sku_total"] == 2 and audit["n_sku_kept"] == 1


def test_zero_fill_creates_zeros_not_nan():
    raw = _raw(_span("a", "2019-01-01", 200, step=20))
    panel, _ = build_daily_panel(raw, min_lifetime_days=100)
    assert panel.y.notna().all()
    assert (panel.y == 0).sum() > 0


# --------------------------------------------------------------------------
# 聚合
# --------------------------------------------------------------------------

def test_aggregate_keeps_only_complete_periods():
    """生命期从周三开始 -> 首周不完整，必须被排除。"""
    raw = _raw(_span("a", "2019-01-02", 200))   # 2019-01-02 是周三
    panel, _ = build_daily_panel(raw, min_lifetime_days=100)
    wk = aggregate_to_period(panel, "W")
    assert (wk.n_days == 7).all()
    assert wk.attrs["n_excluded_incomplete"] >= 1
    assert wk.origin.min() > pd.Timestamp("2018-12-31")   # 首个不完整周已排除


def test_aggregate_month_requires_full_calendar_month():
    raw = _raw(_span("a", "2019-01-15", 200))
    panel, _ = build_daily_panel(panel_raw := raw, min_lifetime_days=100)
    mo = aggregate_to_period(panel, "M")
    assert (mo.n_days == mo.origin.dt.days_in_month).all()
    assert pd.Timestamp("2019-01-01") not in set(mo.origin)   # 1 月不完整


def test_aggregate_sum_matches_daily():
    raw = _raw(_span("a", "2019-01-07", 210))   # 周一起
    panel, _ = build_daily_panel(raw, min_lifetime_days=100)
    wk = aggregate_to_period(panel, "W")
    covered = panel[panel.d.between(wk.origin.min(),
                                    wk.origin.max() + pd.Timedelta(days=6))]
    assert np.isclose(wk.y.sum(), covered.y.sum())


def test_aggregate_rejects_bad_freq():
    raw = _raw(_span("a", "2019-01-01", 200))
    panel, _ = build_daily_panel(raw, min_lifetime_days=100)
    with pytest.raises(ValueError, match="freq"):
        aggregate_to_period(panel, "Q")


# --------------------------------------------------------------------------
# 滞后特征
# --------------------------------------------------------------------------

def test_lags_exclude_current_period():
    """lag1 必须是上一期的值，绝不能含当期。"""
    df = pd.DataFrame({"sku_ID": ["a"] * 5, "y": [1., 2., 3., 4., 5.]})
    out = add_lag_features(df, lags=(1,), rolls=(2,))
    assert np.isnan(out.lag1.iloc[0])
    assert list(out.lag1.iloc[1:]) == [1., 2., 3., 4.]
    # roll2 在第 2 行应为 mean(y[0]) = 1.0，绝不含 y[1]
    assert out.roll2.iloc[1] == 1.0


def test_periods_observed_counts_prior_periods():
    df = pd.DataFrame({"sku_ID": ["a"] * 3 + ["b"] * 2, "y": [1., 2., 3., 4., 5.]})
    out = add_lag_features(df)
    assert list(out.periods_observed) == [0., 1., 2., 0., 1.]


# --------------------------------------------------------------------------
# 经验分位网格
# --------------------------------------------------------------------------

def test_quantile_grid_uses_only_history_before_origin():
    """历史窗不同则网格应不同。

    构造：前 60 天每日销量 1，其后每日销量 5。取 origin=第 60 天与第 190 天，
    q85 应分别落在两个水平上 —— 若函数误用了 origin 之后的数据，两者会相同。
    """
    d0 = pd.Timestamp("2019-01-01")
    rows = [("a", str((d0 + pd.Timedelta(days=k)).date()), 1.0 if k < 60 else 5.0)
            for k in range(200)]
    panel, _ = build_daily_panel(_raw(rows), min_lifetime_days=100)
    lv = np.array([.5, .85])
    _, g_early = empirical_quantile_grid(panel, lv, d0 + pd.Timedelta(days=60), min_obs=10)
    _, g_late = empirical_quantile_grid(panel, lv, d0 + pd.Timedelta(days=190), min_obs=10)
    assert g_early.shape == g_late.shape
    assert g_early[0, 1] == 1.0, f"早期 q85 应为 1，得 {g_early[0, 1]}"
    assert g_late[0, 1] == 5.0, f"晚期 q85 应为 5，得 {g_late[0, 1]}"


def test_quantile_grid_preserves_zero_atom():
    """必须用 inverted_cdf：线性插值会在 0 与 1 之间产生分数值，抹平零原子。"""
    raw = _raw(_span("a", "2019-01-01", 200, step=10))   # 90% 的日为 0
    panel, _ = build_daily_panel(raw, min_lifetime_days=100)
    lv = np.array([.1, .5, .85, .99])
    _, g = empirical_quantile_grid(panel, lv, pd.Timestamp("2019-07-01"), min_obs=10)
    assert g[0, 0] == 0.0 and g[0, 1] == 0.0
    assert np.all(g == np.rint(g)), "出现分数分位数，说明未使用 inverted_cdf"


def test_quantile_grid_drops_short_history():
    raw = _raw(_span("a", "2019-01-01", 200, step=5))
    panel, _ = build_daily_panel(raw, min_lifetime_days=100)
    sids, g = empirical_quantile_grid(panel, np.array([.5]),
                                      pd.Timestamp("2019-01-10"), min_obs=30)
    assert len(sids) == 0                        # origin 前不足 30 个观测
