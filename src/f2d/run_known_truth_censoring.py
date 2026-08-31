"""Known-truth validation of policy-dependent censoring bias.

The static experiment evaluates two stock targets under latent demand D and
logged sales Y=min(D, c).  The closed-loop experiment first generates logged
sales from a legacy base-stock policy and then replays both candidate policies
on latent demand and on those logged sales.

Usage:
    PYTHONPATH=src python -m f2d.run_known_truth_censoring
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR
from .conventions import SEED_BASE
from .simulation import ReplayConfig, replay


STATIC_CAPS = (99, 15, 14, 13, 12, 11)
LOGGING_TARGETS = (22, 20, 19, 18, 17, 16)


def _poisson_pmf(mean: float, max_demand: int = 100) -> tuple[np.ndarray, np.ndarray]:
    support = np.arange(max_demand + 1)
    pmf = np.empty(max_demand + 1)
    pmf[0] = math.exp(-mean)
    for k in range(1, max_demand + 1):
        pmf[k] = pmf[k - 1] * mean / k
    if 1.0 - pmf.sum() > 1e-12:
        raise ValueError("Poisson support is too short")
    return support, pmf / pmf.sum()


def _newsvendor_cost(stock: float, demand: np.ndarray,
                      holding_cost: float, shortage_cost: float) -> np.ndarray:
    return (holding_cost * np.maximum(stock - demand, 0)
            + shortage_cost * np.maximum(demand - stock, 0))


def static_table() -> pd.DataFrame:
    """Exact one-period comparison and closed-form wedge verification."""
    demand, probability = _poisson_pmf(10.0)
    low, high = 12.0, 15.0
    h, p = 1.0, 19.0
    true_delta = float(np.sum(
        probability * (_newsvendor_cost(high, demand, h, p)
                       - _newsvendor_cost(low, demand, h, p))))
    rows = []
    for cap in STATIC_CAPS:
        sales = np.minimum(demand, cap)
        logged_delta = float(np.sum(
            probability * (_newsvendor_cost(high, sales, h, p)
                           - _newsvendor_cost(low, sales, h, p))))
        wedge = logged_delta - true_delta
        # Pathwise identity for Y=min(D,c):
        # [C_H(Y)-C_L(Y)]-[C_H(D)-C_L(D)]
        # = (h+p)[min(S_H,D)-max(S_L,c)]^+.
        closed_form = float((h + p) * np.sum(
            probability
            * np.maximum(np.minimum(high, demand) - max(low, cap), 0)))
        censoring_probability = float(probability[demand > cap].sum())
        hidden_units = float(np.sum(probability * np.maximum(demand - cap, 0)))
        rows.append({
            "logging_cap": cap,
            "censoring_probability": censoring_probability,
            "hidden_demand_share": hidden_units / 10.0,
            "true_high_minus_low_cost": true_delta,
            "logged_high_minus_low_cost": logged_delta,
            "differential_bias": wedge,
            "closed_form_wedge": closed_form,
            "favors_lower_stock": wedge > 1e-12,
            "ranking_reversal": true_delta < 0 < logged_delta,
        })
    out = pd.DataFrame(rows)
    if not np.allclose(out.differential_bias, out.closed_form_wedge,
                       rtol=0.0, atol=1e-10):
        raise AssertionError("Static censoring-wedge identity failed")
    return out


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    half_width = 1.96 * float(values.std(ddof=1)) / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def _closed_loop_cost(demand: np.ndarray, target: float,
                      cfg: ReplayConfig, burn_in_days: int,
                      month_days: int, holding_cost: float,
                      shortage_cost: float) -> np.ndarray:
    n_paths, n_days = demand.shape
    n_reviews = len(range(0, n_days, cfg.review_cadence_days))
    result = replay(
        demand,
        np.full((n_paths, n_reviews), target),
        np.zeros(n_paths),
        cfg,
    )
    if result.conservation_violations:
        raise AssertionError(result.conservation_violations)
    scored_days = n_days - burn_in_days
    if scored_days % month_days:
        raise ValueError("Scored window must contain complete months")
    n_months = scored_days / month_days
    return (holding_cost * result.i_end[:, burn_in_days:].mean(axis=1)
            + shortage_cost
            * result.lost[:, burn_in_days:].sum(axis=1) / n_months)


def closed_loop_table(n_paths: int = 10_000,
                      seed: int = SEED_BASE) -> pd.DataFrame:
    """Known-truth multi-cycle stress test using the canonical replay engine."""
    n_days, burn_in_days, month_days = 360, 60, 30
    low, high = 16.0, 20.0
    h, p = 1.0, 19.0
    cfg = ReplayConfig(
        n_days=n_days,
        lead_time_days=1,
        review_cadence_days=7,
        shortage_mechanism="lost_sales",
    )
    rng = np.random.default_rng(seed)
    latent = rng.poisson(1.5, size=(n_paths, n_days))

    true_low = _closed_loop_cost(
        latent, low, cfg, burn_in_days, month_days, h, p)
    true_high = _closed_loop_cost(
        latent, high, cfg, burn_in_days, month_days, h, p)
    true_diff = true_high - true_low
    true_mean, true_lo, true_hi = _mean_ci(true_diff)

    rows = []
    n_reviews = len(range(0, n_days, cfg.review_cadence_days))
    for logging_target in LOGGING_TARGETS:
        logger = replay(
            latent,
            np.full((n_paths, n_reviews), logging_target),
            np.zeros(n_paths),
            cfg,
        )
        if logger.conservation_violations:
            raise AssertionError(logger.conservation_violations)
        sales = logger.fulfilled.copy()
        scored_lost = logger.lost[:, burn_in_days:]
        scored_latent = latent[:, burn_in_days:]
        censored_day_share = float(np.mean(scored_lost > 0))
        hidden_demand_share = float(scored_lost.sum() / scored_latent.sum())
        del logger
        logged_low = _closed_loop_cost(
            sales, low, cfg, burn_in_days, month_days, h, p)
        logged_high = _closed_loop_cost(
            sales, high, cfg, burn_in_days, month_days, h, p)
        logged_diff = logged_high - logged_low
        wedge = logged_diff - true_diff
        logged_mean, logged_lo, logged_hi = _mean_ci(logged_diff)
        wedge_mean, wedge_lo, wedge_hi = _mean_ci(wedge)
        rows.append({
            "logging_target": logging_target,
            "censored_day_share": censored_day_share,
            "hidden_demand_share": hidden_demand_share,
            "true_high_minus_low_cost": true_mean,
            "true_ci_low": true_lo,
            "true_ci_high": true_hi,
            "logged_high_minus_low_cost": logged_mean,
            "logged_ci_low": logged_lo,
            "logged_ci_high": logged_hi,
            "differential_bias": wedge_mean,
            "bias_ci_low": wedge_lo,
            "bias_ci_high": wedge_hi,
            "favors_lower_stock": wedge_mean > 1e-10,
            "ranking_reversal": true_mean < 0 < logged_mean,
            "n_paths": n_paths,
            "seed": seed,
        })
    return pd.DataFrame(rows)


def _validate(static: pd.DataFrame, dynamic: pd.DataFrame) -> None:
    no_static_bias = static[static.logging_cap >= 15].differential_bias
    if not np.allclose(no_static_bias, 0.0, atol=1e-10):
        raise AssertionError("Censoring below neither target should not change ranking")
    if not static.ranking_reversal.any():
        raise AssertionError("Static design did not produce a ranking reversal")
    dominating_logger = dynamic.loc[dynamic.logging_target >= 20]
    if not np.allclose(dominating_logger.differential_bias, 0.0, atol=1e-12):
        raise AssertionError("A logging policy at least as high as both candidates must have zero wedge")
    if not dynamic.ranking_reversal.any():
        raise AssertionError("Closed-loop design did not produce a ranking reversal")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-paths", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=SEED_BASE)
    parser.add_argument("--output-dir", type=Path,
                        default=ARTIFACT_DIR / "known_truth_censoring")
    args = parser.parse_args(argv)
    if args.n_paths < 2:
        parser.error("--n-paths must be at least 2")

    static = static_table()
    dynamic = closed_loop_table(args.n_paths, args.seed)
    _validate(static, dynamic)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    static.to_csv(args.output_dir / "static_known_truth.csv", index=False)
    dynamic.to_csv(args.output_dir / "closed_loop_known_truth.csv", index=False)

    print("Static known-truth comparison")
    print(static[["logging_cap", "true_high_minus_low_cost",
                  "logged_high_minus_low_cost", "differential_bias",
                  "ranking_reversal"]].to_string(index=False))
    print("\nClosed-loop known-truth comparison")
    print(dynamic[["logging_target", "censored_day_share",
                   "true_high_minus_low_cost", "logged_high_minus_low_cost",
                   "differential_bias", "ranking_reversal"]].to_string(index=False))
    print(f"\nSaved results to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
