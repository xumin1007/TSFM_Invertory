"""审计 monthly-reset switching value 与 carry-state long-run policy value。

本脚本不重新调用预测模型。它读取 ``audit_long_v2.csv`` 中已经冻结的每月
arm-specific 订至水平 S，并把同一组 S 放入两种不同的闭环状态设计：

1. ``switch_empty``：论文当前口径。每月重置到共同 logged on-hand，pipeline
   置零，只在月初复核一次。
2. ``switch_logged_pipeline``：仍然每月重置，但加入月初 logged
   on_order_inventory。pre-origin PO 记录提供到货日的分配比例，快照提供权威
   总量；它们对所有 arms 都是共同的 sunk commitments。origin 后的新订单
   仍完全由各 arm 自己产生。
3. ``carry_logged_pipeline``：从最早可用 origin 连续回放。每个 arm 独立继承
   on-hand 与 pipeline，月末不再重置；最早四个月只 warm up，7--10 月评分。

持有成本 ``h = kappa_h * unit_cost / 12`` 是月率。为与现有 R3 完全对账，
默认按每个自然月加 1 个 lead-time day 的保护窗口计分；同时输出不重叠的
calendar-month 口径作为长期政策的运营解释。

用法：``PYTHONPATH=src python -m f2d.audit_replay_estimands``
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable

import numpy as np
import pandas as pd

from . import config as cfgmod
from .checks import CheckResult
from .conventions import SEED_BASE
from .datasets import zhao
from .decision import costs_from_alpha
from .run_rolling_origins import HIGH_ALPHAS, ORIGINS
from .simulation import ReplayConfig, replay

ART = cfgmod.ARTIFACT_DIR / "zhao_audit"
LONG = ART / "audit_long_v2.csv"
SELECTION = cfgmod.ARTIFACT_DIR / "zhao_rolling_origins" / "baseline_selection.csv"
LEAD_DAYS = 1
KAPPA_H = 0.20
B = 10_000


def _demand_matrix(daily: pd.DataFrame, sids: np.ndarray,
                   start: pd.Timestamp, n_days: int) -> np.ndarray:
    dates = pd.date_range(start, periods=n_days, freq="D")
    d = daily[(daily.d >= dates[0]) & (daily.d <= dates[-1])
              & daily.series_id.isin(sids)]
    wide = d.pivot(index="series_id", columns="d", values="y")
    return wide.reindex(index=sids, columns=dates, fill_value=0.0).fillna(0.0).to_numpy(float)


def _logged_pipeline(raw_orders: pd.DataFrame, sids: np.ndarray,
                     start: pd.Timestamp, n_days: int,
                     snapshot_qty: np.ndarray | None = None) -> np.ndarray:
    """回放前已经下单的到货日程；总量以月初在途快照为权威。

    可重建 PO 提供实际到货日的相对分配；若快照有在途量而 PO 表无对应明细，
    余额按声明的 L 到货。origin 后的新订单绝不从日志读取。
    """
    o = raw_orders[(raw_orders.order_date < start)
                   & (raw_orders.arrival_date >= start)
                   & raw_orders.sku_ID.isin(sids)].copy()
    if not o.empty:
        o["day"] = (o.arrival_date - start).dt.days.astype(int)
    schedule_days = max(n_days, int(o.day.max()) + 1 if not o.empty else n_days)
    out = np.zeros((len(sids), schedule_days), float)
    if not o.empty:
        sid_idx = pd.Series(np.arange(len(sids)), index=sids)
        g = o.groupby(["sku_ID", "day"], as_index=False).quantity.sum()
        ii = g.sku_ID.map(sid_idx).to_numpy(int)
        np.add.at(out, (ii, g.day.to_numpy(int)), g.quantity.to_numpy(float))
    if snapshot_qty is not None:
        target = np.clip(np.asarray(snapshot_qty, float), 0.0, None)
        rebuilt = out.sum(axis=1)
        has_detail = rebuilt > 0
        out[has_detail] *= (target[has_detail] / rebuilt[has_detail])[:, None]
        fallback = (~has_detail) & (target > 0)
        out[fallback, min(LEAD_DAYS, out.shape[1] - 1)] = target[fallback]
    return out


def _selection_map() -> dict[tuple[float, int], str]:
    sel = pd.read_csv(SELECTION)
    return {(float(r.alpha), int(r.origin_idx)): str(r.retuned)
            for _, r in sel.iterrows()}


def _monthly_block(df: pd.DataFrame, alpha: float, oi: int,
                   base_arm: str) -> pd.DataFrame:
    d = df[(df.alpha == alpha) & (df.origin_idx == oi)
           & df.arm.isin(["chronos2-zs", base_arm])]
    p = d.pivot(index="series_id", columns="arm", values=["S", "L0", "inv0"])
    p = p.dropna(subset=[("S", "chronos2-zs"), ("S", base_arm)])
    out = pd.DataFrame(index=p.index)
    out["Sz"] = p[("S", "chronos2-zs")]
    out["Sb"] = p[("S", base_arm)]
    out["L0z"] = p[("L0", "chronos2-zs")]
    out["L0b"] = p[("L0", base_arm)]
    out["inv0"] = p[("inv0", "chronos2-zs")]
    return out


def _cost(result, h: np.ndarray, p: np.ndarray,
          start: int = 0, stop: int | None = None) -> np.ndarray:
    stop = result.n_days if stop is None else stop
    return (h * result.i_end[:, start:stop].mean(axis=1)
            + p * result.lost[:, start:stop].sum(axis=1))


def _switching_rows(df: pd.DataFrame, daily: pd.DataFrame, panel: pd.DataFrame,
                    raw_orders: pd.DataFrame, with_pipeline: bool) -> pd.DataFrame:
    base_of = _selection_map()
    rows = []
    for oi, month in enumerate(ORIGINS):
        snap = panel[panel.month == month].set_index("series_id")
        for alpha in HIGH_ALPHAS:
            base = base_of[(alpha, oi)]
            block = _monthly_block(df, alpha, oi, base)
            block = block.loc[block.index.intersection(snap.index)].sort_index()
            sids = block.index.to_numpy(str)
            n_days = month.days_in_month + LEAD_DAYS
            demand = _demand_matrix(daily, sids, month, n_days)
            inv0 = block.inv0.to_numpy(float)
            snapshot_pipeline = snap.loc[sids, "on_order_inventory"].to_numpy(float)
            pipeline = (_logged_pipeline(raw_orders, sids, month, n_days,
                                         snapshot_pipeline)
                        if with_pipeline else None)
            cfg = ReplayConfig(n_days=n_days, lead_time_days=LEAD_DAYS,
                               review_cadence_days=n_days)
            rz = replay(demand, block.Sz.to_numpy(float)[:, None], inv0, cfg,
                        initial_pipeline_arrivals=pipeline)
            rb = replay(demand, block.Sb.to_numpy(float)[:, None], inv0, cfg,
                        initial_pipeline_arrivals=pipeline)
            unit_cost = snap.loc[sids, "unit_cost_hist"].to_numpy(float)
            h, p = costs_from_alpha(unit_cost, alpha, KAPPA_H, 12)
            pipe0 = np.zeros(len(sids)) if pipeline is None else pipeline.sum(axis=1)
            rows.append(pd.DataFrame({
                "series_id": sids, "origin_idx": oi, "alpha": alpha,
                "L0z": block.L0z.to_numpy(float),
                "L0b": block.L0b.to_numpy(float),
                "dynz": _cost(rz, h, p), "dynb": _cost(rb, h, p),
                "order_z": rz.order[:, 0], "order_b": rb.order[:, 0],
                "pipeline0": pipe0,
            }))
    return pd.concat(rows, ignore_index=True)


def _common_sids(df: pd.DataFrame) -> np.ndarray:
    sets = []
    for oi in range(len(ORIGINS)):
        sets.append(set(df[(df.origin_idx == oi) & (df.arm == "chronos2-zs")].series_id))
    return np.asarray(sorted(set.intersection(*sets)), dtype=str)


def _carry_rows(df: pd.DataFrame, daily: pd.DataFrame, panel: pd.DataFrame,
                raw_orders: pd.DataFrame, with_pipeline: bool) -> pd.DataFrame:
    base_of = _selection_map()
    sids = _common_sids(df)
    start = ORIGINS[0]
    stop_exclusive = ORIGINS[-1] + pd.DateOffset(months=1) + pd.Timedelta(days=LEAD_DAYS)
    n_days = int((stop_exclusive - start).days)
    demand = _demand_matrix(daily, sids, start, n_days)
    snap0 = panel[panel.month == start].set_index("series_id")
    snapshot_pipeline = snap0.loc[sids, "on_order_inventory"].to_numpy(float)
    pipeline = (_logged_pipeline(raw_orders, sids, start, n_days,
                                 snapshot_pipeline)
                if with_pipeline else None)
    review_days = tuple(int((m - start).days) for m in ORIGINS)
    rows = []

    for alpha in HIGH_ALPHAS:
        z_levels, b_levels, l0z, l0b = [], [], [], []
        for oi, _ in enumerate(ORIGINS):
            base = base_of[(alpha, oi)]
            block = _monthly_block(df, alpha, oi, base).reindex(sids)
            if block.isna().any().any():
                raise ValueError(f"共同样本在 alpha={alpha}, origin={oi} 缺少 S/L0")
            z_levels.append(block.Sz.to_numpy(float))
            b_levels.append(block.Sb.to_numpy(float))
            l0z.append(block.L0z.to_numpy(float))
            l0b.append(block.L0b.to_numpy(float))

        inv0 = snap0.loc[sids, "beginning_inventory"].to_numpy(float)
        cfg = ReplayConfig(n_days=n_days, lead_time_days=LEAD_DAYS,
                           review_days=review_days)
        rz = replay(demand, np.column_stack(z_levels), inv0, cfg,
                    initial_pipeline_arrivals=pipeline)
        rb = replay(demand, np.column_stack(b_levels), inv0, cfg,
                    initial_pipeline_arrivals=pipeline)

        for oi, month in enumerate(ORIGINS):
            a = int((month - start).days)
            calendar_b = int((month + pd.DateOffset(months=1) - start).days)
            protection_b = min(calendar_b + LEAD_DAYS, n_days)
            snap = panel[panel.month == month].set_index("series_id")
            unit_cost = snap.loc[sids, "unit_cost_hist"].to_numpy(float)
            h, p = costs_from_alpha(unit_cost, alpha, KAPPA_H, 12)
            common = {
                "series_id": sids, "origin_idx": oi, "alpha": alpha,
                "L0z": l0z[oi], "L0b": l0b[oi],
                "order_z": rz.order[:, a], "order_b": rb.order[:, a],
                "pipeline0": (np.zeros(len(sids)) if pipeline is None
                              else pipeline.sum(axis=1)),
            }
            rows.append(pd.DataFrame({
                **common, "score_window": "calendar",
                "dynz": _cost(rz, h, p, a, calendar_b),
                "dynb": _cost(rb, h, p, a, calendar_b),
            }))
            rows.append(pd.DataFrame({
                **common, "score_window": "protection",
                "dynz": _cost(rz, h, p, a, protection_b),
                "dynb": _cost(rb, h, p, a, protection_b),
            }))
    return pd.concat(rows, ignore_index=True)


def _point(d: pd.DataFrame) -> np.ndarray:
    r0 = (d.L0z.sum() - d.L0b.sum()) / d.L0b.sum() * 100
    r3 = (d.dynz.sum() - d.dynb.sum()) / d.dynb.sum() * 100
    return np.array([r0, r3, r3 - r0])


def _cluster_ci(d: pd.DataFrame, b: int, seed: int = SEED_BASE):
    clusters = [g.index.to_numpy() for _, g in d.groupby("series_id", sort=False)]
    point = _point(d)
    rng = np.random.default_rng(seed)
    draws = np.empty((b, 3))
    for j in range(b):
        picks = rng.integers(0, len(clusters), len(clusters))
        rows = np.concatenate([clusters[k] for k in picks])
        draws[j] = _point(d.loc[rows])
    return point, np.quantile(draws, .025, axis=0), np.quantile(draws, .975, axis=0)


def _report(label: str, d: pd.DataFrame, b: int) -> dict:
    point, lo, hi = _cluster_ci(d.reset_index(drop=True), b)
    print(f"\n{label}  (n={len(d):,}, clusters={d.series_id.nunique():,})")
    for name, i, unit in (("R0 static", 0, "%"), ("R dynamic", 1, "%"),
                          ("attenuation", 2, " pp")):
        print(f"  {name:<13} {point[i]:+7.2f}{unit}  "
              f"[{lo[i]:+7.2f}, {hi[i]:+7.2f}]")
    return {
        "n": int(len(d)), "n_series": int(d.series_id.nunique()),
        "R0": round(float(point[0]), 3), "Rdyn": round(float(point[1]), 3),
        "attenuation_pp": round(float(point[2]), 3),
        "Rdyn_ci": [round(float(lo[1]), 3), round(float(hi[1]), 3)],
        "attenuation_ci": [round(float(lo[2]), 3), round(float(hi[2]), 3)],
    }


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=B)
    args = ap.parse_args(argv)

    if not LONG.exists():
        raise FileNotFoundError(f"先运行 audit_transmission --rebuild：{LONG}")
    df = pd.read_csv(LONG, dtype={"series_id": str})
    raw = zhao.load_raw()
    daily, _ = zhao.build_daily_panel(raw)
    panel, _ = zhao.build_panel(raw)
    panel["series_id"] = panel.series_id.astype(str)

    print("构造 monthly-reset short-run switching replay …")
    sw0 = _switching_rows(df, daily, panel, raw["orders"], with_pipeline=False)
    swp = _switching_rows(df, daily, panel, raw["orders"], with_pipeline=True)
    sw0.to_csv(ART / "switch_empty.csv", index=False)
    swp.to_csv(ART / "switch_logged_pipeline.csv", index=False)

    print("构造 continuous carry-state replay …")
    car0 = _carry_rows(df, daily, panel, raw["orders"], with_pipeline=False)
    carp = _carry_rows(df, daily, panel, raw["orders"], with_pipeline=True)
    car0.to_csv(ART / "carry_empty_pipeline.csv", index=False)
    carp.to_csv(ART / "carry_logged_pipeline.csv", index=False)

    chk = CheckResult(step_id="zhao_audit_replay_estimands", dataset="zhao",
                      seed=SEED_BASE)
    notes = {}
    notes["switch_all_empty"] = _report("Current switch / empty pipeline / Mar-Oct", sw0, args.b)
    notes["switch_all_logged_pipeline"] = _report(
        "Switch / logged pre-existing pipeline / Mar-Oct", swp, args.b)
    notes["switch_jul_oct_empty"] = _report(
        "Current switch / empty pipeline / Jul-Oct", sw0[sw0.origin_idx >= 4], args.b)
    notes["switch_jul_oct_logged_pipeline"] = _report(
        "Switch / logged pre-existing pipeline / Jul-Oct",
        swp[swp.origin_idx >= 4], args.b)

    for frame_name, frame in (("carry_empty", car0), ("carry_logged_pipeline", carp)):
        for score_window in ("calendar", "protection"):
            for first_oi, warm_label in ((3, "jun_oct_92d_warmup"),
                                         (4, "jul_oct_122d_warmup")):
                d = frame[(frame.score_window == score_window)
                          & (frame.origin_idx >= first_oi)]
                key = f"{frame_name}_{score_window}_{warm_label}"
                notes[key] = _report(key.replace("_", " "), d, args.b)

    pipe_share = float((swp.pipeline0 > 0).mean())
    notes["switch_logged_pipeline_positive_share"] = round(pipe_share, 4)
    notes["switch_logged_pipeline_mean_qty"] = round(float(swp.pipeline0.mean()), 3)
    chk.notes.update(notes)
    chk.n_rows = len(swp) + len(carp)
    # 现有空-pipeline 口径必须与已冻结的 −3.78% 对账。
    chk.assert_true("reproduce_current_R3",
                    abs(notes["switch_all_empty"]["Rdyn"] - (-3.780)) < 0.01,
                    f"got {notes['switch_all_empty']['Rdyn']}")
    chk.finish(ART / "checks_replay_estimands")
    print(f"\n结果已保存到 {ART}")
    return chk.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
