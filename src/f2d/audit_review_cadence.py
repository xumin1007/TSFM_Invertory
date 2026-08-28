"""Review-cadence sensitivity for the arm-specific closed-loop replay.

The forecasting origin remains monthly: within a month, all replenishment reviews use
the daily predictive path frozen at that month's origin. This isolates replenishment
frequency from forecast-update frequency. For each R in {1, 7, 30} days, the base-stock
target is recomputed from the corresponding R+L protection-interval distribution; the
monthly S is never reused at a shorter cadence.

Two state estimands are reported:
  * monthly reset to logged on-hand with empty pipeline (short-run switching);
  * continuous arm-specific carry state from March, with the logged on-order snapshot
    shared only at the initial origin and July--October scored after a 122-day warm-up.

Use ``--rebuild`` to refresh cached Chronos daily quantile paths.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable

import numpy as np
import pandas as pd

from . import config as cfgmod
from .aggregation import convolve_varying_pmf, pmf_quantile
from .checks import CheckResult
from .conventions import SEED_BASE
from .datasets import zhao
from .decision import costs_from_alpha
from .models.chronos import (BASE_CHECKPOINT, NATIVE_LEVELS, QuantileRepair,
                             to_grid)
from .run_rolling_origins import (HIGH_ALPHAS, ORIGINS, _emp_grid)
from .simulation import ReplayConfig, replay

ART = cfgmod.ARTIFACT_DIR / "zhao_audit"
LONG = ART / "audit_long_v2.csv"
SELECTION = cfgmod.ARTIFACT_DIR / "zhao_rolling_origins" / "baseline_selection.csv"
CADENCES = (1, 7, 30)
LEAD_DAYS = 1
KAPPA_H = 0.20
VMAX = 60
B = 10_000


def _selection_map() -> dict[tuple[float, int], str]:
    d = pd.read_csv(SELECTION)
    return {(float(r.alpha), int(r.origin_idx)): str(r.retuned)
            for _, r in d.iterrows()}


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
    o = raw_orders[(raw_orders.order_date < start)
                   & (raw_orders.arrival_date >= start)
                   & raw_orders.sku_ID.isin(sids)].copy()
    if not o.empty:
        o["day"] = (o.arrival_date - start).dt.days.astype(int)
    out = np.zeros((len(sids), max(n_days, int(o.day.max()) + 1 if not o.empty else n_days)),
                   float)
    if not o.empty:
        pos = pd.Series(np.arange(len(sids)), index=sids)
        g = o.groupby(["sku_ID", "day"], as_index=False).quantity.sum()
        np.add.at(out, (g.sku_ID.map(pos).to_numpy(int), g.day.to_numpy(int)),
                  g.quantity.to_numpy(float))
    if snapshot_qty is not None:
        target = np.clip(np.asarray(snapshot_qty, float), 0.0, None)
        rebuilt = out.sum(axis=1)
        has_detail = rebuilt > 0
        out[has_detail] *= (target[has_detail] / rebuilt[has_detail])[:, None]
        fallback = (~has_detail) & (target > 0)
        out[fallback, min(LEAD_DAYS, out.shape[1] - 1)] = target[fallback]
    return out


def _prediction_length(n_month_days: int) -> int:
    ends = []
    for r in CADENCES:
        ends.extend(t + r + LEAD_DAYS for t in range(0, n_month_days, r))
    return max(ends)


def _chronos_grid(pipe, sids: np.ndarray, ctx: dict[str, np.ndarray],
                  oi: int, pred_len: int, args):
    cache = ART / f"cadence_chronos_oi{oi}_n{args.n_series}_h{pred_len}.npz"
    if cache.exists() and not args.rebuild:
        z = np.load(cache, allow_pickle=False)
        cached_sids = z["sids"].astype(str)
        pos = pd.Series(np.arange(len(cached_sids)), index=cached_sids)
        if set(sids).issubset(pos.index):
            return z["grid"][pos.loc[sids].to_numpy(int)], pipe

    if pipe is None:
        from chronos import BaseChronosPipeline
        print(f"加载 Chronos-2 ({BASE_CHECKPOINT}) …")
        pipe = BaseChronosPipeline.from_pretrained(
            BASE_CHECKPOINT, device_map=args.device, local_files_only=True)
    import torch
    q, _ = pipe.predict_quantiles(
        [torch.tensor(ctx[s], dtype=torch.float32) for s in sids],
        prediction_length=pred_len, quantile_levels=list(NATIVE_LEVELS),
        batch_size=args.batch_size)
    grid, _ = QuantileRepair()(to_grid(q))
    grid = grid.reshape(len(sids), pred_len, -1)
    np.savez_compressed(cache, sids=sids.astype("U"), grid=grid)
    return grid, pipe


def _base_stock_paths(grid: np.ndarray, review_days: tuple[int, ...],
                      cadence: int, alpha: float) -> np.ndarray:
    levels = []
    pi_len = cadence + LEAD_DAYS
    for t in review_days:
        pmf = convolve_varying_pmf(
            NATIVE_LEVELS, [grid[:, k, :] for k in range(t, t + pi_len)],
            vmax=VMAX)
        levels.append(pmf_quantile(pmf, [alpha])[alpha])
    return np.column_stack(levels)


def _monthly_policy_blocks(daily: pd.DataFrame, panel: pd.DataFrame,
                           args) -> dict[tuple[int, int, float], dict]:
    # 与 transmission 审计严格使用同一冻结样本；缓存可以是它的超集。
    if LONG.exists():
        frozen = pd.read_csv(LONG, usecols=["series_id", "origin_idx"],
                             dtype={"series_id": str})
        keep = frozen.series_id.unique()
        frozen_by_origin = {
            int(oi): np.asarray(sorted(g.series_id.unique()), dtype=str)
            for oi, g in frozen.groupby("origin_idx")}
    else:
        rng = np.random.default_rng(SEED_BASE)
        pool = np.asarray(sorted(set(daily.series_id)))
        keep = rng.choice(pool, size=min(args.n_series, len(pool)), replace=False)
        frozen_by_origin = None
    daily = daily[daily.series_id.isin(keep)]
    sku_of = dict(zip(daily.series_id, daily.sku_ID))
    base_of = _selection_map()
    blocks = {}
    pipe = None

    for oi, month in enumerate(ORIGINS):
        snap = panel[panel.month == month].set_index("sku_ID")
        hist = daily[daily.d < month]
        ctx = {s: g.y.to_numpy() for s, g in hist.groupby("series_id")}
        if frozen_by_origin is not None:
            sids = frozen_by_origin[oi]
        else:
            sids = np.asarray([s for s in sorted(ctx)
                               if len(ctx[s]) >= 30 and sku_of[s] in snap.index], dtype=str)
        pred_len = _prediction_length(month.days_in_month)
        gz, pipe = _chronos_grid(pipe, sids, ctx, oi, pred_len, args)
        print(f"{month:%Y-%m}: {len(sids)} series, horizon={pred_len} "
              f"({time.time()-args._t0:.0f}s)")

        skus = np.asarray([sku_of[s] for s in sids])
        cur = snap.loc[skus]
        inv0 = cur.beginning_inventory.to_numpy(float)
        unit_cost = cur.unit_cost_hist.to_numpy(float)
        n_eval = month.days_in_month + LEAD_DAYS
        demand = _demand_matrix(daily, sids, month, n_eval)

        emp_grids = {}
        for alpha in HIGH_ALPHAS:
            base = base_of[(alpha, oi)]
            if base not in emp_grids:
                emp_grids[base] = _emp_grid(base, sids, ctx)
            gb = np.repeat(emp_grids[base][:, None, :], pred_len, axis=1)
            for cadence in CADENCES:
                reviews = tuple(range(0, month.days_in_month, cadence))
                blocks[(oi, cadence, alpha)] = {
                    "sids": sids, "review_days": reviews,
                    "Sz": _base_stock_paths(gz, reviews, cadence, alpha),
                    "Sb": _base_stock_paths(gb, reviews, cadence, alpha),
                    "inv0": inv0, "unit_cost": unit_cost, "demand": demand,
                }
    return blocks


def _cost(result, h: np.ndarray, p: np.ndarray,
          a: int = 0, b: int | None = None) -> np.ndarray:
    b = result.n_days if b is None else b
    return (h * result.i_end[:, a:b].mean(axis=1)
            + p * result.lost[:, a:b].sum(axis=1))


def _switch_rows(blocks: dict) -> pd.DataFrame:
    rows = []
    for (oi, cadence, alpha), x in blocks.items():
        cfg = ReplayConfig(n_days=x["demand"].shape[1], lead_time_days=LEAD_DAYS,
                           review_days=x["review_days"])
        rz = replay(x["demand"], x["Sz"], x["inv0"], cfg)
        rb = replay(x["demand"], x["Sb"], x["inv0"], cfg)
        h, p = costs_from_alpha(x["unit_cost"], alpha, KAPPA_H, 12)
        same_order = np.all(np.isclose(rz.order, rb.order), axis=1)
        all_zero = ((rz.order.sum(axis=1) <= 1e-9)
                    & (rb.order.sum(axis=1) <= 1e-9))
        rows.append(pd.DataFrame({
            "series_id": x["sids"], "origin_idx": oi, "alpha": alpha,
            "cadence": cadence, "dynz": _cost(rz, h, p),
            "dynb": _cost(rb, h, p), "same_order": same_order,
            "all_zero": all_zero,
        }))
    return pd.concat(rows, ignore_index=True)


def _carry_rows(blocks: dict, daily: pd.DataFrame, panel: pd.DataFrame,
                raw_orders: pd.DataFrame) -> pd.DataFrame:
    common = set.intersection(*(set(x["sids"]) for x in blocks.values()))
    sids = np.asarray(sorted(common), dtype=str)
    start = ORIGINS[0]
    stop_exclusive = ORIGINS[-1] + pd.DateOffset(months=1) + pd.Timedelta(days=LEAD_DAYS)
    n_days = int((stop_exclusive - start).days)
    demand = _demand_matrix(daily, sids, start, n_days)
    snap0 = panel[panel.month == start].set_index("series_id")
    initial_pipeline = _logged_pipeline(
        raw_orders, sids, start, n_days,
        snap0.loc[sids, "on_order_inventory"].to_numpy(float))
    inv0 = snap0.loc[sids, "beginning_inventory"].to_numpy(float)
    rows = []

    for cadence in CADENCES:
        for alpha in HIGH_ALPHAS:
            abs_reviews, zcols, bcols = [], [], []
            for oi, month in enumerate(ORIGINS):
                x = blocks[(oi, cadence, alpha)]
                pos = pd.Series(np.arange(len(x["sids"])), index=x["sids"])
                take = pos.loc[sids].to_numpy(int)
                offset = int((month - start).days)
                abs_reviews.extend(offset + t for t in x["review_days"])
                zcols.append(x["Sz"][take])
                bcols.append(x["Sb"][take])
            review_days = tuple(abs_reviews)
            cfg = ReplayConfig(n_days=n_days, lead_time_days=LEAD_DAYS,
                               review_days=review_days)
            rz = replay(demand, np.column_stack(zcols), inv0, cfg,
                        initial_pipeline_arrivals=initial_pipeline)
            rb = replay(demand, np.column_stack(bcols), inv0, cfg,
                        initial_pipeline_arrivals=initial_pipeline)
            for oi, month in enumerate(ORIGINS):
                a = int((month - start).days)
                b = int((month + pd.DateOffset(months=1) - start).days)
                snap = panel[panel.month == month].set_index("series_id")
                h, p = costs_from_alpha(
                    snap.loc[sids, "unit_cost_hist"].to_numpy(float),
                    alpha, KAPPA_H, 12)
                review_mask = np.asarray([(a <= t < b) for t in review_days])
                oz = rz.order[:, np.asarray(review_days)[review_mask]]
                ob = rb.order[:, np.asarray(review_days)[review_mask]]
                rows.append(pd.DataFrame({
                    "series_id": sids, "origin_idx": oi, "alpha": alpha,
                    "cadence": cadence, "dynz": _cost(rz, h, p, a, b),
                    "dynb": _cost(rb, h, p, a, b),
                    "same_order": np.all(np.isclose(oz, ob), axis=1),
                    "all_zero": ((oz.sum(axis=1) <= 1e-9)
                                 & (ob.sum(axis=1) <= 1e-9)),
                }))
    return pd.concat(rows, ignore_index=True)


def _effect(d: pd.DataFrame) -> float:
    return float((d.dynz.sum() - d.dynb.sum()) / d.dynb.sum() * 100)


def _cluster_ci(d: pd.DataFrame, b: int, seed: int = SEED_BASE):
    d = d.reset_index(drop=True)
    clusters = [g.index.to_numpy() for _, g in d.groupby("series_id", sort=False)]
    rng = np.random.default_rng(seed)
    draws = np.empty(b)
    for j in range(b):
        picks = rng.integers(0, len(clusters), len(clusters))
        rows = np.concatenate([clusters[k] for k in picks])
        draws[j] = _effect(d.loc[rows])
    return _effect(d), *np.quantile(draws, [.025, .975])


def _summarize(label: str, d: pd.DataFrame, b: int) -> list[dict]:
    out = []
    print(f"\n{label}")
    for cadence in CADENCES:
        x = d[d.cadence == cadence]
        point, lo, hi = _cluster_ci(x, b)
        row = {
            "estimand": label, "cadence": cadence, "effect_pct": point,
            "ci_lo": lo, "ci_hi": hi, "same_order_share": x.same_order.mean(),
            "all_zero_share": x.all_zero.mean(), "n": len(x),
            "n_series": x.series_id.nunique(),
        }
        out.append(row)
        print(f"  R={cadence:>2}d  effect={point:+7.2f}% "
              f"[{lo:+7.2f}, {hi:+7.2f}]  "
              f"same-order={row['same_order_share']:.1%}  "
              f"all-zero={row['all_zero_share']:.1%}")
    return out


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=2000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--b", type=int, default=B)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)
    args._t0 = time.time()

    ART.mkdir(parents=True, exist_ok=True)
    raw = zhao.load_raw()
    daily, _ = zhao.build_daily_panel(raw)
    panel, _ = zhao.build_panel(raw)
    panel["series_id"] = panel.series_id.astype(str)

    blocks = _monthly_policy_blocks(daily, panel, args)
    switching = _switch_rows(blocks)
    carry = _carry_rows(blocks, daily, panel, raw["orders"])
    switching.to_csv(ART / "cadence_switch.csv", index=False)
    carry.to_csv(ART / "cadence_carry.csv", index=False)

    summary = []
    summary += _summarize("monthly_reset_empty_pipeline_mar_oct", switching, args.b)
    summary += _summarize("carry_logged_pipeline_jul_oct",
                          carry[carry.origin_idx >= 4], args.b)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(ART / "cadence_summary.csv", index=False)

    chk = CheckResult(step_id="zhao_audit_review_cadence", dataset="zhao",
                      seed=SEED_BASE)
    chk.n_rows = len(switching) + len(carry)
    chk.note("forecast_update_frequency", "monthly_frozen_path")
    chk.note("summary", summary_df.round(6).to_dict("records"))
    chk.finish(ART / "checks_review_cadence")
    return chk.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
