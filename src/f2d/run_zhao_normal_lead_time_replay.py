"""Zhao closed-loop robustness check with decision-time promised lead times.

Forecast-induced stock targets remain fixed at the main-analysis values.  At
each review date, however, a simulated new order receives a promised lead time
drawn from that SKU's orders placed strictly before the review date.  The
actual-arrival field is deliberately never used.  Common lead-time schedules
are applied to every policy arm within a draw.

Usage:
    PYTHONPATH=src python -m f2d.run_zhao_normal_lead_time_replay
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfgmod
from .conventions import SEED_BASE
from .datasets import zhao
from .decision import costs_from_alpha
from .run_halfsynthetic import (HIGH_ALPHAS, KAPPA_H, ORIGINS, R,
                                SCORED_ORIGINS)
from .simulation import ReplayConfig, replay


ART = cfgmod.ARTIFACT_DIR / "zhao_normal_lead_time_replay"
HALF_ART = cfgmod.ARTIFACT_DIR / "zhao_halfsynthetic"


def _seed(base_seed: int, series_id: str, review_day: int) -> int:
    payload = f"{base_seed}|{series_id}|{review_day}".encode()
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def _logged_sales_matrix(daily: pd.DataFrame, sids: np.ndarray,
                         start: pd.Timestamp, n_days: int) -> np.ndarray:
    index = {sid: i for i, sid in enumerate(sids)}
    out = np.zeros((len(sids), n_days))
    end = start + pd.Timedelta(days=n_days - 1)
    rows = daily[(daily.d >= start) & (daily.d <= end)].copy()
    rows = rows[rows.series_id.isin(index)]
    i = rows.series_id.map(index).to_numpy()
    t = ((rows.d - start).dt.days).to_numpy()
    out[i, t] = rows.y.to_numpy(float)
    return out


def _lead_time_draws(orders: pd.DataFrame, sids: np.ndarray,
                     skus: np.ndarray, start: pd.Timestamp,
                     review_days: tuple[int, ...], n_draws: int,
                     seed: int) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Sample only from normal_lead_time values known before each review day."""
    orders = orders[["sku_ID", "order_date", "normal_lead_time"]].copy()
    orders["sku_ID"] = orders["sku_ID"].astype(str)
    orders["normal_lead_time"] = orders["normal_lead_time"].astype(int)
    n_ser, n_reviews = len(sids), len(review_days)
    draws = np.empty((n_draws, n_ser, n_reviews), dtype=np.int16)
    rows: list[dict] = []
    own_history_count = np.zeros((n_ser, n_reviews), dtype=int)
    own_zero_count = np.zeros((n_ser, n_reviews), dtype=int)

    for j, day in enumerate(review_days):
        date = start + pd.Timedelta(days=day)
        # `normal_lead_time` is recorded when an order is placed.  Strictly
        # earlier dates exclude same-day records and any post-decision data.
        history = orders[orders.order_date < date]
        global_support = history.normal_lead_time.to_numpy(int)
        if not len(global_support):
            raise RuntimeError(f"No promised-lead-time history before {date:%F}")
        by_sku = {
            sku: values.to_numpy(int)
            for sku, values in history.groupby("sku_ID").normal_lead_time
        }
        fallback = 0
        for i, (sid, sku) in enumerate(zip(sids, skus)):
            support = by_sku.get(str(sku))
            if support is None or not len(support):
                support = global_support
                fallback += 1
            else:
                own_history_count[i, j] = len(support)
                own_zero_count[i, j] = int(np.count_nonzero(support == 0))

            # The replay receives arrivals before placing new orders.  A zero
            # promised lead time therefore becomes next-day arrival, the
            # shortest feasible lead time under this event ordering.
            support = np.maximum(support, 1)
            rng = np.random.default_rng(_seed(seed, str(sid), day))
            draws[:, i, j] = rng.choice(support, size=n_draws, replace=True)

        sampled = draws[:, :, j]
        rows.append({
            "review_date": str(date.date()),
            "review_day": day,
            "sku_specific_history_share": float((own_history_count[:, j] > 0).mean()),
            "global_fallback_skus": fallback,
            "median_sku_history_orders": float(np.median(own_history_count[:, j])),
            "raw_zero_share_in_sku_history": float(
                own_zero_count[:, j].sum() / max(own_history_count[:, j].sum(), 1)),
            "sampled_mean_days": float(sampled.mean()),
            "sampled_p50_days": float(np.quantile(sampled, .50)),
            "sampled_p90_days": float(np.quantile(sampled, .90)),
            "sampled_max_days": int(sampled.max()),
        })

    metadata = {
        "lead_time_source": "normal_lead_time",
        "history_rule": "order_date < review_date",
        "arrival_date_used": False,
        "zero_day_mapping": "0-day promised lead time is mapped to 1 day",
        "common_random_numbers_across_arms": True,
    }
    return draws, pd.DataFrame(rows), metadata


def _load_policies() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Timestamp,
                              tuple[int, ...], dict, dict]:
    path = HALF_ART / "fixed_policy_inputs.npz"
    if not path.exists():
        raise RuntimeError("Missing fixed policies; run f2d.run_halfsynthetic first")
    saved = np.load(path, allow_pickle=False)
    sids = saved["sids"].astype(str)
    inv0 = saved["inv0"].astype(float)
    cost_i = saved["cost_i"].astype(float)
    start = pd.Timestamp(saved["start_date"].item())

    selection = pd.read_csv(
        cfgmod.ARTIFACT_DIR / "zhao_rolling_origins" / "baseline_selection.csv")
    baseline = {(float(row.alpha), int(row.origin_idx)): row.retuned
                for _, row in selection.iterrows()}
    arms = {"chronos2-zs"}
    arms.update(baseline[(alpha, oi)]
                for alpha in HIGH_ALPHAS for oi in SCORED_ORIGINS)

    policies: dict[tuple[str, float], np.ndarray] = {}
    review_days: tuple[int, ...] | None = None
    for arm in arms:
        for alpha in HIGH_ALPHAS:
            prefix = f"{arm.replace('-', '_')}_a{int(round(alpha * 100))}"
            days = tuple(saved[f"{prefix}_review_days"].astype(int))
            if review_days is None:
                review_days = days
            elif days != review_days:
                raise RuntimeError("Policy inputs use inconsistent review calendars")
            policies[(arm, alpha)] = saved[f"{prefix}_S_arr"].astype(float)
    assert review_days is not None
    return sids, inv0, cost_i, start, review_days, policies, baseline


def _policy_pair_costs(demand: np.ndarray, inv0: np.ndarray, cost_i: np.ndarray,
                       start: pd.Timestamp, review_days: tuple[int, ...],
                       policies: dict, baseline: dict,
                       order_lead_times: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Return per-SKU numerator and denominator across predeclared cells."""
    n_ser, n_days = demand.shape
    outcomes: dict[tuple[str, float, int], np.ndarray] = {}
    for (arm, alpha), targets in policies.items():
        h, p = costs_from_alpha(cost_i, alpha, KAPPA_H, 12)
        result = replay(
            demand, targets, inv0,
            ReplayConfig(n_days=n_days, lead_time_days=1,
                         review_cadence_days=R, review_days=review_days),
            order_lead_times=order_lead_times,
        )
        if result.conservation_violations:
            raise RuntimeError("; ".join(result.conservation_violations))
        for oi in SCORED_ORIGINS:
            month = ORIGINS[oi]
            begin = (month - start).days
            end = begin + month.days_in_month
            outcomes[(arm, alpha, oi)] = (
                h * result.i_end[:, begin:end].mean(axis=1)
                + p * result.lost[:, begin:end].sum(axis=1))

    numerator = np.zeros(n_ser)
    denominator = np.zeros(n_ser)
    for alpha in HIGH_ALPHAS:
        for oi in SCORED_ORIGINS:
            empirical = baseline[(alpha, oi)]
            c2 = outcomes[("chronos2-zs", alpha, oi)]
            emp = outcomes[(empirical, alpha, oi)]
            numerator += c2 - emp
            denominator += emp
    if np.any(denominator <= 0):
        raise RuntimeError("Nonpositive empirical-policy cost denominator")
    return numerator, denominator


def _effect(numerator: np.ndarray, denominator: np.ndarray) -> float:
    return float(numerator.sum() / denominator.sum() * 100)


def _bootstrap(fixed_d: np.ndarray, fixed_b: np.ndarray,
               variable_d: np.ndarray, variable_b: np.ndarray,
               n_boot: int, seed: int) -> np.ndarray:
    """Jointly resample SKU clusters and lead-time schedules."""
    n_draws, n_ser = variable_d.shape
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, 3))
    for b in range(n_boot):
        sku_weight = rng.multinomial(n_ser, np.full(n_ser, 1 / n_ser))
        draw_weight = rng.multinomial(n_draws, np.full(n_draws, 1 / n_draws))
        def weighted_effect(d, base):
            num = (d @ sku_weight * draw_weight).sum()
            den = (base @ sku_weight * draw_weight).sum()
            return num / den * 100
        fixed = weighted_effect(fixed_d, fixed_b)
        variable = weighted_effect(variable_d, variable_b)
        out[b] = fixed, variable, variable - fixed
    return out


def _p_two_sided(draws: np.ndarray) -> float:
    p_hi = (1 + np.count_nonzero(draws >= 0)) / (len(draws) + 1)
    p_lo = (1 + np.count_nonzero(draws <= 0)) / (len(draws) + 1)
    return min(1.0, 2 * min(p_hi, p_lo))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-draws", type=int, default=50)
    parser.add_argument("--b", type=int, default=10_000)
    args = parser.parse_args(argv)
    if args.n_draws < 1 or args.b < 1:
        parser.error("--n-draws and --b must be positive")

    ART.mkdir(parents=True, exist_ok=True)
    sids, inv0, cost_i, start, review_days, policies, baseline = _load_policies()
    raw = zhao.load_raw()
    daily, _ = zhao.build_daily_panel(raw)
    daily["series_id"] = daily.series_id.astype(str)
    sku_of = (daily.drop_duplicates("series_id")
                   .set_index("series_id")["sku_ID"].astype(str).to_dict())
    skus = np.asarray([sku_of[sid] for sid in sids])
    n_days = int(np.load(HALF_ART / "fixed_policy_inputs.npz",
                         allow_pickle=False)["total_days"].item())
    demand = _logged_sales_matrix(daily, sids, start, n_days)

    lead_draws, schedule_summary, metadata = _lead_time_draws(
        raw["orders"], sids, skus, start, review_days, args.n_draws, SEED_BASE)
    np.savez_compressed(
        ART / "normal_lead_time_draws.npz",
        sids=sids.astype("U"), review_days=np.asarray(review_days),
        lead_times=lead_draws,
    )
    schedule_summary.to_csv(ART / "lead_time_schedule_summary.csv", index=False)

    fixed_num, fixed_den = _policy_pair_costs(
        demand, inv0, cost_i, start, review_days, policies, baseline, None)
    fixed_d = np.repeat(fixed_num[None, :], args.n_draws, axis=0)
    fixed_b = np.repeat(fixed_den[None, :], args.n_draws, axis=0)
    variable_d = np.empty_like(fixed_d)
    variable_b = np.empty_like(fixed_b)
    for draw_id, schedule in enumerate(lead_draws):
        variable_d[draw_id], variable_b[draw_id] = _policy_pair_costs(
            demand, inv0, cost_i, start, review_days, policies, baseline, schedule)

    fixed_effect = _effect(fixed_num, fixed_den)
    variable_effect = _effect(variable_d.mean(axis=0), variable_b.mean(axis=0))
    boot = _bootstrap(fixed_d, fixed_b, variable_d, variable_b, args.b,
                      SEED_BASE + 909)
    names = ["fixed_one_day", "sku_history_normal_lead_time", "difference"]
    point = [fixed_effect, variable_effect, variable_effect - fixed_effect]
    units = ["percent", "percent", "percentage_points"]
    summary = pd.DataFrame({
        "estimand": names,
        "effect": point,
        "ci_low": np.quantile(boot, .025, axis=0),
        "ci_high": np.quantile(boot, .975, axis=0),
        "p_two_sided": [_p_two_sided(boot[:, k]) for k in range(3)],
        "unit": units,
    })
    summary.to_csv(ART / "summary.csv", index=False)
    np.savez_compressed(ART / "policy_cost_draws.npz",
                        fixed_numerator=fixed_d, fixed_denominator=fixed_b,
                        variable_numerator=variable_d,
                        variable_denominator=variable_b,
                        bootstrap=boot)

    metadata.update({
        "n_series": int(len(sids)),
        "n_review_dates": int(len(review_days)),
        "n_lead_time_draws": args.n_draws,
        "bootstrap_replicates": args.b,
        "fixed_policy_targets": True,
        "demand_input": "logged sales",
        "seed": SEED_BASE,
    })
    (ART / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print("Normal-lead-time closed-loop replay")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(schedule_summary[["sku_specific_history_share", "sampled_mean_days",
                            "sampled_p90_days", "sampled_max_days"]]
          .describe().to_string())
    print(f"Saved results to {ART}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
