"""Reviewer-facing robustness analyses for the forecast-to-decision pipeline.

This module leaves every frozen production estimand unchanged and adds four
auditable sensitivity analyses:

1. semi-synthetic latent-demand generators (empirical conditional, truncated
   Poisson, and moment-matched truncated negative binomial);
2. protection-interval aggregation under Gaussian-copula AR(1) dependence;
3. alternative quantile-to-PMF reconstructions; and
4. coverage and exceedance diagnostics at the operational quantiles.

The script deliberately reuses the common-series manifest, origin-specific
retuned empirical baseline, fixed base-stock policies, and carry-state replay
from ``run_halfsynthetic``.  It writes only new robustness artifacts and does
not overwrite the frozen half-synthetic or grid outputs.

Usage
-----
PYTHONPATH=src python -m f2d.run_review_robustness \
  --device mps --batch-size 256 --n-draws 50 --copula-draws 16384 --b 10000
"""

from __future__ import annotations

import argparse
import hashlib
import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import qmc

from . import config as cfgmod
from .aggregation import (convolve_varying_pmf, pmf_quantile,
                          quantile_grid_to_pmf)
from .conventions import SEED_BASE
from .datasets import zhao
from .decision import costs_from_alpha
from .models.chronos import (BASE_CHECKPOINT, BASE_REVISION, NATIVE_LEVELS,
                             QuantileRepair, to_grid)
from .models.gbdt_grid import make_lean_features
from .run_halfsynthetic import (_construct_latent_demand, _series_draw_seed)
from .run_rolling_origins import (EMP_SCHEMES, HIGH_ALPHAS, ORIGINS,
                                  _emp_grid)
from .simulation import ReplayConfig, replay


ART = cfgmod.ARTIFACT_DIR / "zhao_review_robustness"
CONT_ART = cfgmod.ARTIFACT_DIR / "zhao_continuous"
ROLL_ART = cfgmod.ARTIFACT_DIR / "zhao_rolling_origins"
VMAX = 60
LEAD_DAYS = 1
KAPPA_H = 0.20
R = 30
PI = R + LEAD_DAYS
BURN_IN_DAYS = 2 * PI
SCORED_ORIGINS = tuple(range(3, 8))

COARSE_LEVELS = np.array([
    .01, .10, .20, .30, .40, .50, .60, .70, .80, .90, .95, .99
])


@dataclass(frozen=True)
class PMFSpec:
    name: str
    levels: np.ndarray
    estimator: str = "midpoint"
    copula_rho: float | None = None


def _hash_seed(*parts) -> int:
    blob = "|".join(map(str, parts)).encode()
    return int(hashlib.sha256(blob).hexdigest()[:16], 16)


def _p2(draws: np.ndarray) -> float:
    x = np.asarray(draws, float)
    b = len(x)
    p_hi = (1 + np.count_nonzero(x >= 0)) / (b + 1)
    p_lo = (1 + np.count_nonzero(x <= 0)) / (b + 1)
    return min(2 * min(p_hi, p_lo), 1.0)


def _summary_row(specification: str, metric: str, point: float,
                 draws: np.ndarray, alpha: str | float = "pooled") -> dict:
    lo, hi = np.quantile(draws, [.025, .975])
    return dict(specification=specification, alpha=alpha, metric=metric,
                point=float(point), ci_lo=float(lo), ci_hi=float(hi),
                p_two_sided=float(_p2(draws)))


def _bootstrap_counts(b: int, n_series: int, n_draws: int,
                      seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Multinomial count weights for a two-way cluster × draw bootstrap."""
    rng = np.random.default_rng(seed)
    cw = np.empty((b, n_series), dtype=np.int16)
    for j in range(b):
        cw[j] = np.bincount(
            rng.integers(0, n_series, size=n_series), minlength=n_series)
    dw = np.empty((b, n_draws), dtype=np.int16)
    for j in range(b):
        dw[j] = np.bincount(
            rng.integers(0, n_draws, size=n_draws), minlength=n_draws)
    return cw, dw


def _weighted_two_way(x: np.ndarray, cw: np.ndarray,
                      dw: np.ndarray) -> np.ndarray:
    """Bootstrap totals for x with shape (draw, series)."""
    return np.einsum("bk,ki,bi->b", dw, x, cw, optimize=True)


def _weighted_cluster(x: np.ndarray, cw: np.ndarray) -> np.ndarray:
    """Cluster-bootstrap totals for x with shape (series,)."""
    return cw @ np.asarray(x, float)


def _truncated_parametric_draw(kind: str, lower: int, donor: np.ndarray,
                               rng: np.random.Generator) -> int:
    """Sample an integer X conditional on X >= lower from a fitted family."""
    donor = np.asarray(donor, float)
    mu = max(float(np.mean(donor)), 1e-6)
    var = max(float(np.var(donor, ddof=1)) if donor.size > 1 else mu, mu)

    if kind == "poisson" or var <= mu + 1e-9:
        dist = stats.poisson(mu)
    elif kind == "negative_binomial":
        size = max(mu * mu / (var - mu), 1e-6)
        prob = size / (size + mu)
        dist = stats.nbinom(size, prob)
    else:
        raise ValueError(kind)

    cdf_below = float(dist.cdf(max(lower - 1, -1)))
    if not np.isfinite(cdf_below) or cdf_below >= 1 - 1e-12:
        positive_mean = (float(donor[donor > 0].mean())
                         if np.any(donor > 0) else 1.0)
        return int(lower + rng.geometric(1.0 / max(positive_mean, 1.0)))
    u = cdf_below + (1.0 - cdf_below) * rng.random()
    x = float(dist.ppf(min(u, np.nextafter(1.0, 0.0))))
    if not np.isfinite(x):
        return int(lower)
    return max(int(math.floor(x)), int(lower))


def _construct_parametric_demand(
    dm_y: np.ndarray,
    sids: np.ndarray,
    start: pd.Timestamp,
    total_days: int,
    censor_months: dict,
    n_draws: int,
    kind: str,
    base_seed: int,
) -> tuple[np.ndarray, dict]:
    """Construct a parametric latent-demand stress DGP on the same flags.

    Parameters are fitted series by series to daily observations from months
    not classified as censored.  Only positive-sales days inside classified
    months are imputed; every other day remains exactly equal to logged sales.
    """
    n_series = len(sids)
    out = np.tile(dm_y, (n_draws, 1, 1))
    changed = []

    for i, sid in enumerate(sids):
        donor_values = []
        target_days = []
        for oi, month in enumerate(ORIGINS):
            mo = (month - start).days
            me = mo + month.days_in_month
            if mo < 0 or me > total_days:
                continue
            if censor_months.get((i, oi), False):
                target_days.extend(
                    t for t in range(mo, me) if dm_y[i, t] > 0)
            else:
                donor_values.extend(dm_y[i, mo:me].tolist())
        donor = np.asarray(donor_values, float)
        if donor.size == 0 or donor.max(initial=0) <= 0 or not target_days:
            continue

        for k in range(n_draws):
            rng = np.random.default_rng(
                _hash_seed(base_seed, "review-dgp", kind, sid, k))
            for t in target_days:
                y = int(dm_y[i, t])
                x = _truncated_parametric_draw(kind, y, donor, rng)
                out[k, i, t] = x
                changed.append(x - y)

    # Match the empirical-generator diagnostic: summarize actual positive
    # imputations, rather than diluting them with draws that equal the lower
    # truncation point.  This makes severity comparable across generators.
    excess = np.asarray(changed, float)
    excess = excess[excess > 0]
    diag = dict(
        generator=kind,
        n_imputed_draw_day_series=int(excess.size),
        mean_excess=float(excess.mean()) if excess.size else 0.0,
        median_excess=float(np.median(excess)) if excess.size else 0.0,
        p95_excess=float(np.quantile(excess, .95)) if excess.size else 0.0,
    )
    return out, diag


def _empirical_diagnostics(dm_y: np.ndarray, dm_d: np.ndarray) -> dict:
    excess = (dm_d - dm_y[None, :, :]).ravel()
    excess = excess[excess > 0]
    return dict(
        generator="empirical_conditional",
        n_imputed_draw_day_series=int(excess.size),
        mean_excess=float(excess.mean()) if excess.size else 0.0,
        median_excess=float(np.median(excess)) if excess.size else 0.0,
        p95_excess=float(np.quantile(excess, .95)) if excess.size else 0.0,
    )


def _values_for_levels(values_per_step: list[np.ndarray],
                       levels: np.ndarray) -> list[np.ndarray]:
    idx = [int(np.flatnonzero(np.isclose(NATIVE_LEVELS, x))[0]) for x in levels]
    return [np.asarray(v)[:, idx] for v in values_per_step]


def _copula_uniforms(m: int, rho: float, n_draws: int) -> np.ndarray:
    if n_draws < 2 or n_draws & (n_draws - 1):
        raise ValueError("--copula-draws must be a power of two")
    eng = qmc.Sobol(d=m, scramble=True,
                    seed=_hash_seed(SEED_BASE, "copula", m, rho) % (2**32))
    u0 = eng.random_base2(int(math.log2(n_draws)))
    z0 = stats.norm.ppf(np.clip(u0, 1e-12, 1 - 1e-12))
    z = np.empty_like(z0)
    z[:, 0] = z0[:, 0]
    scale = math.sqrt(1 - rho * rho)
    for t in range(1, m):
        z[:, t] = rho * z[:, t - 1] + scale * z0[:, t]
    return stats.norm.cdf(z)


def _copula_sum_quantiles(levels: np.ndarray,
                          values_per_step: list[np.ndarray],
                          alphas: tuple[float, ...], rho: float,
                          n_draws: int, estimator: str,
                          chunk_size: int = 32) -> dict[float, np.ndarray]:
    """Quantiles of a sum under a Gaussian-copula AR(1) dependence stress."""
    m = len(values_per_step)
    n_series = values_per_step[0].shape[0]
    u = _copula_uniforms(m, rho, n_draws)
    out = {a: np.empty(n_series) for a in alphas}

    for left in range(0, n_series, chunk_size):
        right = min(left + chunk_size, n_series)
        cdfs = [
            np.cumsum(quantile_grid_to_pmf(
                levels, v[left:right], VMAX, cdf_estimator=estimator), axis=1)
            for v in values_per_step
        ]
        sums = np.zeros((n_draws, right - left), dtype=np.int32)
        for t, cdf in enumerate(cdfs):
            for j in range(right - left):
                sums[:, j] += np.searchsorted(cdf[j], u[:, t], side="left")
        for a in alphas:
            out[a][left:right] = np.quantile(
                sums, a, axis=0, method="inverted_cdf")
    return out


def _sum_quantiles(spec: PMFSpec, values_per_step: list[np.ndarray],
                   alphas: tuple[float, ...], copula_draws: int
                   ) -> dict[float, np.ndarray]:
    vals = _values_for_levels(values_per_step, spec.levels)
    if spec.copula_rho is None:
        pmf = convolve_varying_pmf(
            spec.levels, vals, vmax=VMAX, cdf_estimator=spec.estimator)
        return pmf_quantile(pmf, alphas)
    return _copula_sum_quantiles(
        spec.levels, vals, alphas, spec.copula_rho, copula_draws,
        spec.estimator)


def _precompute_policies(monthly_grids: dict, arms: list[str],
                         sids: np.ndarray, start: pd.Timestamp,
                         total_days: int, spec: PMFSpec,
                         copula_draws: int) -> dict:
    """Precompute fixed P3 policies for one aggregation specification."""
    precomp = {}
    alphas = tuple(HIGH_ALPHAS)
    for arm in arms:
        review_days = []
        s_lists = {a: [] for a in alphas}
        for oi, month in enumerate(ORIGINS):
            gr = monthly_grids[oi][arm]
            mo = (month - start).days
            for dim in range(0, month.days_in_month, R):
                abs_day = mo + dim
                if abs_day >= total_days:
                    break
                review_days.append(abs_day)
                vals = [gr[min(dim + k, len(gr) - 1)] for k in range(PI)]
                qs = _sum_quantiles(spec, vals, alphas, copula_draws)
                for a in alphas:
                    s_lists[a].append(qs[a])

        static = {a: {} for a in alphas}
        for oi in SCORED_ORIGINS:
            month = ORIGINS[oi]
            gr = monthly_grids[oi][arm]
            vals = gr[:month.days_in_month + LEAD_DAYS]
            qs = _sum_quantiles(spec, vals, alphas, copula_draws)
            for a in alphas:
                static[a][oi] = qs[a]

        for a in alphas:
            precomp[(arm, a)] = (
                tuple(review_days), np.column_stack(s_lists[a]), static[a])
    return precomp


def _score(precomp: dict, arms: list[str], dm: np.ndarray,
           start: pd.Timestamp, total_days: int, inv0: np.ndarray,
           cost_i: np.ndarray) -> dict:
    """Return draw-level static and carry-state costs on the fixed support."""
    if dm.ndim == 2:
        dm = dm[None, :, :]
    n_draws, n_series, _ = dm.shape
    score = {}

    for arm in arms:
        for alpha in HIGH_ALPHAS:
            review_days, s_arr, s_static = precomp[(arm, alpha)]
            h, p = costs_from_alpha(cost_i, alpha, KAPPA_H, 12)
            l0 = {oi: np.empty((n_draws, n_series)) for oi in SCORED_ORIGINS}
            l3 = {oi: np.empty((n_draws, n_series)) for oi in SCORED_ORIGINS}
            for k in range(n_draws):
                rc = ReplayConfig(
                    n_days=total_days, lead_time_days=LEAD_DAYS,
                    review_cadence_days=R, shortage_mechanism="lost_sales",
                    review_days=review_days)
                res = replay(dm[k], s_arr, inv0, rc)
                for oi in SCORED_ORIGINS:
                    month = ORIGINS[oi]
                    mo = (month - start).days
                    me = mo + month.days_in_month
                    if mo < BURN_IN_DAYS:
                        raise RuntimeError("Scored origin violates burn-in")
                    l3[oi][k] = (
                        h * res.i_end[:, mo:me].mean(axis=1)
                        + p * res.lost[:, mo:me].sum(axis=1))
                    pi_end = min(me + LEAD_DAYS, total_days)
                    realized = dm[k, :, mo:pi_end].sum(axis=1)
                    s = s_static[oi]
                    l0[oi][k] = (h * np.maximum(s - realized, 0)
                                  + p * np.maximum(realized - s, 0))
            for oi in SCORED_ORIGINS:
                score[(arm, alpha, oi, "L0")] = l0[oi]
                score[(arm, alpha, oi, "L3")] = l3[oi]
    return score


def _paired(score: dict, base_of: dict, metric: str,
            alphas: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Return Chronos and Emp costs as (draw, series, cells-per-series)."""
    z, e = [], []
    for alpha in alphas:
        for oi in SCORED_ORIGINS:
            emp = base_of[(alpha, oi)]
            z.append(score[("chronos2-zs", alpha, oi, metric)])
            e.append(score[(emp, alpha, oi, metric)])
    return np.stack(z, axis=2), np.stack(e, axis=2)


def _ratio_point(z: np.ndarray, e: np.ndarray) -> float:
    zm = z.mean(axis=0)
    em = e.mean(axis=0)
    den = em.sum()
    if not np.isfinite(den) or den <= 0:
        raise RuntimeError("Nonpositive ratio-of-sums denominator")
    return float((zm - em).sum() / den)


def _cluster_effects(score: dict, base_of: dict, cw: np.ndarray,
                     specification: str) -> list[dict]:
    rows = []
    for alpha_label, alphas in [("pooled", tuple(HIGH_ALPHAS)),
                                *[(a, (a,)) for a in HIGH_ALPHAS]]:
        boot = {}
        point = {}
        for metric in ("L0", "L3"):
            z, e = _paired(score, base_of, metric, alphas)
            if z.shape[0] != 1:
                raise ValueError("Cluster-only effect expects one demand path")
            num = (z[0] - e[0]).sum(axis=1)
            den = e[0].sum(axis=1)
            bden = _weighted_cluster(den, cw)
            if np.any(bden <= 0):
                raise RuntimeError("Nonpositive bootstrap denominator")
            boot[metric] = _weighted_cluster(num, cw) / bden
            point[metric] = float(num.sum() / den.sum())
        boot["A"] = boot["L3"] - boot["L0"]
        point["A"] = point["L3"] - point["L0"]
        for metric, label in (("L0", "R_static"),
                              ("L3", "R_dynamic"), ("A", "attenuation")):
            rows.append(_summary_row(
                specification, label, point[metric], boot[metric], alpha_label))
    return rows


def _two_way_factorial(y_score: dict, d_score: dict, base_of: dict,
                       cw: np.ndarray, dw: np.ndarray,
                       specification: str) -> list[dict]:
    point, boot = {}, {}
    for metric in ("L0", "L3"):
        yz, ye = _paired(y_score, base_of, metric, tuple(HIGH_ALPHAS))
        dz, de = _paired(d_score, base_of, metric, tuple(HIGH_ALPHAS))
        yn, yd = (yz[0] - ye[0]).sum(axis=1), ye[0].sum(axis=1)
        dn, dd = (dz - de).sum(axis=2), de.sum(axis=2)
        by_den = _weighted_cluster(yd, cw)
        bd_den = _weighted_two_way(dd, cw, dw)
        if np.any(by_den <= 0) or np.any(bd_den <= 0):
            raise RuntimeError("Nonpositive factorial bootstrap denominator")
        boot[("Y", metric)] = _weighted_cluster(yn, cw) / by_den
        boot[("D", metric)] = _weighted_two_way(dn, cw, dw) / bd_den
        point[("Y", metric)] = float(yn.sum() / yd.sum())
        point[("D", metric)] = _ratio_point(dz, de)

    for demand in ("Y", "D"):
        boot[(demand, "A")] = (boot[(demand, "L3")]
                                - boot[(demand, "L0")])
        point[(demand, "A")] = (point[(demand, "L3")]
                                  - point[(demand, "L0")])
    boot[("D", "DiD")] = boot[("D", "A")] - boot[("Y", "A")]
    point[("D", "DiD")] = point[("D", "A")] - point[("Y", "A")]

    out = []
    for key, label in [
        (("D", "L0"), "R_D_static"),
        (("D", "L3"), "R_D_dynamic"),
        (("D", "A"), "A_D"),
        (("D", "DiD"), "DiD"),
    ]:
        out.append(_summary_row(
            specification, label, point[key], boot[key], "pooled"))
    return out


def _static_realized(dm: np.ndarray, start: pd.Timestamp,
                     total_days: int) -> dict[int, np.ndarray]:
    if dm.ndim == 2:
        dm = dm[None, :, :]
    out = {}
    for oi in SCORED_ORIGINS:
        month = ORIGINS[oi]
        mo = (month - start).days
        me = min(mo + month.days_in_month + LEAD_DAYS, total_days)
        out[oi] = dm[:, :, mo:me].sum(axis=2)
    return out


def _calibration_rows(precomp: dict, base_of: dict, dm: np.ndarray,
                      demand_label: str, start: pd.Timestamp,
                      total_days: int, cw: np.ndarray,
                      dw: np.ndarray | None) -> list[dict]:
    realized = _static_realized(dm, start, total_days)
    rows = []
    for alpha in HIGH_ALPHAS:
        for policy in ("Chronos-2", "Emp-retuned"):
            cover_blocks, excess_blocks = [], []
            for oi in SCORED_ORIGINS:
                arm = ("chronos2-zs" if policy == "Chronos-2"
                       else base_of[(alpha, oi)])
                s = precomp[(arm, alpha)][2][oi]
                y = realized[oi]
                cover_blocks.append((y <= s[None, :]).astype(float))
                excess_blocks.append(np.maximum(y - s[None, :], 0.0))
            cover = np.stack(cover_blocks, axis=2).sum(axis=2)
            excess = np.stack(excess_blocks, axis=2).sum(axis=2)
            n_cells = len(SCORED_ORIGINS) * cover.shape[1]

            if cover.shape[0] == 1:
                bcover = _weighted_cluster(cover[0], cw) / n_cells
                bexcess = _weighted_cluster(excess[0], cw) / n_cells
            else:
                if dw is None or dw.shape[1] != cover.shape[0]:
                    raise ValueError("Draw weights do not match calibration draws")
                scale = cover.shape[0] * n_cells
                bcover = _weighted_two_way(cover, cw, dw) / scale
                bexcess = _weighted_two_way(excess, cw, dw) / scale

            point_cover = float(cover.mean() / len(SCORED_ORIGINS))
            point_excess = float(excess.mean() / len(SCORED_ORIGINS))
            lo, hi = np.quantile(bcover - alpha, [.025, .975])
            elo, ehi = np.quantile(bexcess, [.025, .975])
            rows.append(dict(
                demand=demand_label, policy=policy, alpha=alpha,
                coverage=point_cover, coverage_gap=point_cover - alpha,
                gap_ci_lo=float(lo), gap_ci_hi=float(hi),
                exceedance_rate=1 - point_cover,
                mean_exceedance=point_excess,
                exceed_ci_lo=float(elo), exceed_ci_hi=float(ehi)))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-draws", type=int, default=50)
    ap.add_argument("--copula-draws", type=int, default=16384)
    ap.add_argument("--b", type=int, default=10_000)
    args = ap.parse_args(argv)

    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)

    common_path = CONT_ART / "common_series.csv"
    selection_path = ROLL_ART / "baseline_selection.csv"
    if not common_path.exists() or not selection_path.exists():
        raise FileNotFoundError(
            "Run continuous replay and rolling origins before robustness")
    common_sids = pd.read_csv(common_path).series_id.to_numpy()
    sel = pd.read_csv(selection_path)
    base_of = {(float(r.alpha), int(r.origin_idx)): str(r.retuned)
               for _, r in sel.iterrows()}
    for a in HIGH_ALPHAS:
        for oi in SCORED_ORIGINS:
            if (a, oi) not in base_of:
                raise KeyError(f"Missing retuned baseline for alpha={a}, origin={oi}")

    raw = zhao.load_raw()
    daily, _ = zhao.build_daily_panel(raw)
    panel, _ = zhao.build_panel(raw)
    daily = daily[daily.series_id.isin(common_sids)]
    sku_of = dict(zip(daily.series_id, daily.sku_ID))
    make_lean_features(daily)

    hist_all = {}
    for oi, month in enumerate(ORIGINS):
        h = daily[daily.d < month]
        hist_all[oi] = {s: g.y.to_numpy() for s, g in h.groupby("series_id")
                        if len(g) >= 30}

    snap = panel[panel.month == ORIGINS[0]].set_index("sku_ID")
    common_sk = np.array([sku_of[s] for s in common_sids])
    common_inv = snap.loc[common_sk, "beginning_inventory"].to_numpy(float)
    common_cost = snap.loc[common_sk, "unit_cost_hist"].to_numpy(float)
    eligible = np.isfinite(common_cost)
    sids = common_sids[eligible]
    inv0 = common_inv[eligible]
    cost_i = common_cost[eligible]
    daily = daily[daily.series_id.isin(sids)]
    n_series = len(sids)
    if n_series != 1135:
        raise RuntimeError(f"Expected 1135 eligible series, found {n_series}")

    start = ORIGINS[0]
    end = ORIGINS[-1] + pd.DateOffset(months=1) + pd.Timedelta(days=LEAD_DAYS - 1)
    total_days = (end - start).days + 1
    dm_y, dm_emp, cstats = _construct_latent_demand(
        daily, panel, sids, sku_of, start, total_days, args.n_draws,
        SEED_BASE)

    from chronos import BaseChronosPipeline
    import torch
    pipe = BaseChronosPipeline.from_pretrained(
        BASE_CHECKPOINT, revision=BASE_REVISION, device_map=args.device)
    monthly_grids = {}
    for oi, month in enumerate(ORIGINS):
        ctx = hist_all[oi]
        ctx_sids = {s: ctx.get(s, np.zeros(30)) for s in sids}
        n_days = month.days_in_month + LEAD_DAYS
        grids = {}
        for scheme in EMP_SCHEMES:
            grids[scheme] = [_emp_grid(scheme, sids, ctx_sids)] * n_days
        q, _ = pipe.predict_quantiles(
            [torch.tensor(ctx_sids[s], dtype=torch.float32) for s in sids],
            prediction_length=n_days,
            quantile_levels=list(NATIVE_LEVELS), batch_size=args.batch_size)
        g, _ = QuantileRepair()(to_grid(q))
        g = g.reshape(n_series, n_days, -1)
        grids["chronos2-zs"] = [g[:, d, :] for d in range(n_days)]
        monthly_grids[oi] = grids
        print(f"Origin {oi} grids complete ({time.time()-t0:.0f}s)")

    arms = {"chronos2-zs"}
    arms.update(base_of.values())
    arms = sorted(arms)
    specs = [
        PMFSpec("midpoint-21 (baseline)", NATIVE_LEVELS, "midpoint"),
        PMFSpec("quantile-linear-21", NATIVE_LEVELS, "quantile_linear"),
        PMFSpec("midpoint-12", COARSE_LEVELS, "midpoint"),
        PMFSpec("Gaussian-copula AR(1), rho=0.25", NATIVE_LEVELS,
                "midpoint", .25),
        PMFSpec("Gaussian-copula AR(1), rho=0.50", NATIVE_LEVELS,
                "midpoint", .50),
    ]

    cw, dw = _bootstrap_counts(
        args.b, n_series, args.n_draws,
        _hash_seed(SEED_BASE, "review-bootstrap"))

    # PMF and temporal-dependence sensitivity, evaluated on the arm-common Y.
    transform_rows = []
    baseline_precomp = None
    baseline_y_score = None
    for spec in specs:
        print(f"Precomputing {spec.name} ...")
        precomp = _precompute_policies(
            monthly_grids, arms, sids, start, total_days, spec,
            args.copula_draws)
        score_y = _score(precomp, arms, dm_y, start, total_days, inv0, cost_i)
        transform_rows.extend(_cluster_effects(score_y, base_of, cw, spec.name))
        if spec.name.startswith("midpoint-21"):
            baseline_precomp = precomp
            baseline_y_score = score_y
    pd.DataFrame(transform_rows).to_csv(
        ART / "aggregation_sensitivity.csv", index=False)

    if baseline_precomp is None or baseline_y_score is None:
        raise RuntimeError("Baseline policy construction missing")

    # Latent-demand generator sensitivity with fixed baseline policies.
    dgp_rows, diagnostic_rows = [], []
    dgp_inputs = [("empirical_conditional", dm_emp)]
    diagnostic_rows.append(_empirical_diagnostics(dm_y, dm_emp))
    for kind in ("poisson", "negative_binomial"):
        d, diag = _construct_parametric_demand(
            dm_y, sids, start, total_days, cstats["censor_months"],
            args.n_draws, kind, SEED_BASE)
        dgp_inputs.append((kind, d))
        diagnostic_rows.append(diag)

    baseline_d_score = None
    for name, d in dgp_inputs:
        print(f"Scoring latent-demand generator {name} ...")
        score_d = _score(
            baseline_precomp, arms, d, start, total_days, inv0, cost_i)
        dgp_rows.extend(_two_way_factorial(
            baseline_y_score, score_d, base_of, cw, dw, name))
        if name == "empirical_conditional":
            baseline_d_score = score_d
    pd.DataFrame(dgp_rows).to_csv(ART / "latent_dgp_sensitivity.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(
        ART / "latent_dgp_diagnostics.csv", index=False)

    # Operational tail calibration for the frozen policy specification.
    if baseline_d_score is None:
        raise RuntimeError("Empirical latent-demand score missing")
    cal = []
    cal.extend(_calibration_rows(
        baseline_precomp, base_of, dm_y, "logged_sales_Y", start,
        total_days, cw, None))
    cal.extend(_calibration_rows(
        baseline_precomp, base_of, dm_emp, "latent_demand_D", start,
        total_days, cw, dw))
    pd.DataFrame(cal).to_csv(ART / "tail_calibration.csv", index=False)

    # Frozen-estimand audit against the stored half-synthetic summary.  The
    # exact comparison is meaningful only for the production number of latent
    # draws; reduced-draw smoke tests retain the structural/finite checks but
    # intentionally allow Monte Carlo variation in D-dependent point estimates.
    frozen = cfgmod.ARTIFACT_DIR / "zhao_halfsynthetic" / "factorial_summary.csv"
    if frozen.exists():
        base = pd.DataFrame(dgp_rows)
        base = base[base.specification == "empirical_conditional"]
        ref = pd.read_csv(frozen)
        if not np.isfinite(base["point"].to_numpy(float)).all():
            raise RuntimeError("Nonfinite empirical-conditional point estimate")
        if args.n_draws == 50:
            ref_map = {
                "R_D_static": "R(D,Static)",
                "R_D_dynamic": "R(D,Dynamic)",
                "A_D": "A_D = Dyn-Stat|D",
                "DiD": "DiD = A_D - A_Y",
            }
            for metric, ref_metric in ref_map.items():
                got = float(base.loc[base.metric == metric, "point"].iloc[0])
                target = float(ref.loc[ref.metric == ref_metric, "point"].iloc[0]) / 100.0
                if abs(got - target) > 5e-4:
                    raise RuntimeError(
                        f"Frozen audit failed for {metric}: {got} vs {target}")
        else:
            print("Reduced-draw smoke test: exact frozen D audit skipped")

    meta = pd.DataFrame([dict(
        n_series=n_series, n_origins=len(SCORED_ORIGINS),
        n_alphas=len(HIGH_ALPHAS), n_latent_draws=args.n_draws,
        copula_draws=args.copula_draws, bootstrap_reps=args.b,
        censoring_rate=cstats["censoring_rate"],
        elapsed_seconds=time.time() - t0)])
    meta.to_csv(ART / "run_metadata.csv", index=False)
    print(f"Robustness analyses complete in {time.time()-t0:.0f}s: {ART}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
