"""Censoring-rate × service-level grid experiment.

For each (α, λ) on a 5×5 grid, compute R₀ (static), R₃ (dynamic),
A = R₃ − R₀ (attenuation), and DiD(α,λ) = A(α,0) − A(α,λ).

α is the full service level: each α gets its own policy quantile and
cost weights.  Policies are fixed across the censoring dimension.

λ is the recensoring fraction.  Starting from latent demand D, a
fraction λ of naturally-censored (series, origin) months have their
demand replaced with observed sales Y.  The censoring sets are nested
across λ via a SHA-256 deterministic ordering.

Simultaneous confidence bands via max-|t| across the grid.

用法:  PYTHONPATH=src python -m f2d.run_grid_censoring_alpha [--device cpu]
"""

from __future__ import annotations

import argparse
import hashlib
import time

import numpy as np
import pandas as pd

from . import config as cfgmod
from .aggregation import convolve_varying_pmf
from .conventions import SEED_BASE
from .datasets import zhao
from .decision import costs_from_alpha, order_up_to
from .models.chronos import BASE_CHECKPOINT, NATIVE_LEVELS, QuantileRepair, to_grid
from .models.gbdt_grid import make_lean_features
from .simulation import ReplayConfig, replay
from .run_rolling_origins import EMP_SCHEMES, ORIGINS, _emp_grid
from .run_halfsynthetic import (
    _construct_latent_demand, _series_draw_seed, _bootstrap_p2,
    VMAX, LEAD_DAYS, KAPPA_H, R, BURN_IN_DAYS, SCORED_ORIGINS, B,
    CONT_ART,
)

ART = cfgmod.ARTIFACT_DIR / "zhao_grid"

GRID_ALPHAS = (0.80, 0.85, 0.90, 0.95, 0.98)
LAMBDAS = (0.0, 0.25, 0.50, 0.75, 1.0)
N_DRAWS = 50


def _sha256_rank_censored_months(
    censor_months: dict[tuple[int, int], bool],
    sids: np.ndarray,
    draw_id: int | None = None,
) -> list[tuple[int, int]]:
    """Return naturally-censored (series_idx, origin_idx) pairs in a
    deterministic SHA-256 order for nested recensoring.

    When draw_id is not None, the ordering is keyed per-draw so that
    each latent-demand draw gets its own recensoring allocation.
    """
    censored = [(i, oi) for (i, oi), v in censor_months.items() if v]
    if draw_id is None:
        prefix = "recensor"
    else:
        prefix = f"recensor|{draw_id}"
    def _key(pair):
        blob = f"{prefix}|{sids[pair[0]]}|{pair[1]}".encode()
        return hashlib.sha256(blob).hexdigest()
    censored.sort(key=_key)
    return censored


def _build_recensored_demand(
    dm_y: np.ndarray,
    dm_d: np.ndarray,
    lam: float,
    censor_months: dict[tuple[int, int], bool],
    sids: np.ndarray,
    start_date: pd.Timestamp,
    total_days: int,
) -> np.ndarray:
    """Build X^(k, λ): start from D, replace top ⌊λ·Nc⌋ censored months
    with Y.  Each draw k gets its own SHA-256 recensoring ordering.
    Returns (n_draws, n_ser, total_days)."""
    n_draws = dm_d.shape[0]
    censored_pairs = [(i, oi) for (i, oi), v in censor_months.items() if v]
    n_c = len(censored_pairs)
    n_to_recensor = int(np.floor(lam * n_c))

    if n_to_recensor == 0:
        return dm_d.copy()

    X = dm_d.copy()
    for k in range(n_draws):
        ordered_k = _sha256_rank_censored_months(
            censor_months, sids, draw_id=k)
        for idx in range(n_to_recensor):
            i, oi = ordered_k[idx]
            month = ORIGINS[oi]
            mo = (month - start_date).days
            me = mo + month.days_in_month
            if mo < 0 or me > total_days:
                continue
            X[k, i, mo:me] = dm_y[i, mo:me]
    return X


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--b", type=int, default=B)
    args = ap.parse_args(argv)

    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)

    # ---- Load data (identical to halfsynthetic) ----
    series_file = CONT_ART / "common_series.csv"
    if not series_file.exists():
        print("ERROR: Run run_continuous_replay first")
        return 1
    common_sids = pd.read_csv(series_file).series_id.to_numpy()
    n_common = len(common_sids)
    print(f"Loaded {n_common} canonical common-support series")

    raw = zhao.load_raw()
    daily, _ = zhao.build_daily_panel(raw)
    panel, _ = zhao.build_panel(raw)
    daily = daily[daily.series_id.isin(common_sids)]
    sku_of = dict(zip(daily.series_id, daily.sku_ID))
    make_lean_features(daily)

    hist_all = {}
    for oi, month in enumerate(ORIGINS):
        h = daily[daily.d < month]
        hist_all[oi] = {s: g.y.to_numpy()
                        for s, g in h.groupby("series_id")
                        if len(g) >= 30}

    snap_march = panel[panel.month == ORIGINS[0]].set_index("sku_ID")
    common_sk = np.array([sku_of[s] for s in common_sids])
    common_cost = snap_march.loc[common_sk, "unit_cost_hist"].to_numpy(float)
    common_inv0 = snap_march.loc[
        common_sk, "beginning_inventory"].to_numpy(float)

    cost_finite = np.isfinite(common_cost)
    sids = common_sids[cost_finite]
    sk = common_sk[cost_finite]
    inv0 = common_inv0[cost_finite]
    cost_i = common_cost[cost_finite]
    n_ser = len(sids)
    n_clu_check = n_ser
    daily = daily[daily.series_id.isin(sids)]
    print(f"Cost-eligible series: {n_ser}/{n_common}")

    # ---- Construct latent demand ----
    start_date = ORIGINS[0]
    end_date = (ORIGINS[-1] + pd.DateOffset(months=1)
                + pd.Timedelta(days=LEAD_DAYS - 1))
    total_days = (end_date - start_date).days + 1

    dm_y, dm_d, cstats = _construct_latent_demand(
        daily, panel, sids, sku_of, start_date, total_days,
        N_DRAWS, SEED_BASE)
    censor_months = cstats["censor_months"]
    print(f"  Natural censoring rate: {cstats['censoring_rate']:.1%}")
    print(f"  Demand built  ({time.time()-t0:.0f}s)")

    # ---- Count censored pairs ----
    censored_pairs = [(i, oi) for (i, oi), v in censor_months.items() if v]
    n_c = len(censored_pairs)
    print(f"  Naturally-censored (series, origin) pairs: {n_c}")

    # ---- Audit: nested censoring sets (per-draw ordering) ----
    # With draw-specific SHA keys, nesting holds per-draw because
    # ⌊λ_lo·Nc⌋ ≤ ⌊λ_hi·Nc⌋ for any ordering when λ_lo ≤ λ_hi.
    for i in range(len(LAMBDAS) - 1):
        n_lo = int(np.floor(LAMBDAS[i] * n_c))
        n_hi = int(np.floor(LAMBDAS[i + 1] * n_c))
        assert n_lo <= n_hi, (
            f"Censoring sets not nested: λ={LAMBDAS[i]} → {n_lo}, "
            f"λ={LAMBDAS[i+1]} → {n_hi}")
    print("  Audit: censoring sets nested (count-level) ✓")

    # ---- Audit: all N_DRAWS orderings are unique ----
    order_hashes = set()
    for k in range(N_DRAWS):
        ord_k = _sha256_rank_censored_months(censor_months, sids, draw_id=k)
        h = hashlib.sha256(str(ord_k).encode()).hexdigest()
        order_hashes.add(h)
    assert len(order_hashes) == N_DRAWS, (
        f"Only {len(order_hashes)}/{N_DRAWS} unique recensoring orderings")
    print(f"  Audit: {N_DRAWS}/{N_DRAWS} unique recensoring orderings ✓")

    # ---- Audit: X(k,0) = D(k) bitwise ----
    X_lam0 = _build_recensored_demand(
        dm_y, dm_d, 0.0, censor_months, sids, start_date, total_days)
    assert np.array_equal(X_lam0, dm_d), "X(k,0) ≠ D(k)"
    del X_lam0
    print("  Audit: X(k,0) = D(k) bitwise ✓")

    # ---- Audit: X(k,1) = Y for all k ----
    X_lam1 = _build_recensored_demand(
        dm_y, dm_d, 1.0, censor_months, sids, start_date, total_days)
    for k in range(N_DRAWS):
        assert np.array_equal(X_lam1[k], dm_y), (
            f"X({k},1) ≠ Y: max diff={np.max(np.abs(X_lam1[k] - dm_y))}")
    del X_lam1
    print("  Audit: X(k,1) = Y for all draws ✓")

    # Compute per-draw scored recensoring rates for each λ
    recensor_scored_rates: dict[float, dict] = {}
    n_total_scored_cells = n_ser * len(SCORED_ORIGINS)
    for lam in LAMBDAS:
        n_rec = int(np.floor(lam * n_c))
        per_draw_rates = np.empty(N_DRAWS)
        for k in range(N_DRAWS):
            ord_k = _sha256_rank_censored_months(
                censor_months, sids, draw_id=k)
            n_scored_k = sum(1 for idx in range(n_rec)
                             if ord_k[idx][1] in SCORED_ORIGINS)
            per_draw_rates[k] = n_scored_k / n_total_scored_cells
        recensor_scored_rates[lam] = dict(
            mean=float(per_draw_rates.mean()),
            sd=float(per_draw_rates.std(ddof=1)) if N_DRAWS > 1 else 0.0)
        print(f"    λ={lam:.2f}: recensor {n_rec}/{n_c} total, "
              f"scored rate={per_draw_rates.mean():.3f} "
              f"± {per_draw_rates.std(ddof=1):.4f} (across {N_DRAWS} draws)")

    # ---- Forecast grids for all α levels ----
    from chronos import BaseChronosPipeline
    import torch
    print("Loading Chronos-2 …")
    from .models.chronos import BASE_REVISION
    pipe = BaseChronosPipeline.from_pretrained(
        BASE_CHECKPOINT, revision=BASE_REVISION, device_map=args.device)

    PI = R + LEAD_DAYS
    monthly_grids = {}
    for oi, month in enumerate(ORIGINS):
        ctx = hist_all[oi]
        ctx_sids = {s: ctx.get(s, np.zeros(30)) for s in sids}
        n_days_m = month.days_in_month + LEAD_DAYS
        grids = {}
        for sch in EMP_SCHEMES:
            ctx_valid = {s: v for s, v in ctx_sids.items() if len(v) >= 30}
            if len(ctx_valid) < n_ser:
                for s in sids:
                    if s not in ctx_valid:
                        ctx_valid[s] = np.zeros(30)
            g_emp = _emp_grid(sch, sids, ctx_valid)
            grids[sch] = [g_emp] * n_days_m

        q, _ = pipe.predict_quantiles(
            [torch.tensor(ctx_sids[s], dtype=torch.float32) for s in sids],
            prediction_length=n_days_m,
            quantile_levels=list(NATIVE_LEVELS),
            batch_size=args.batch_size)
        g, _ = QuantileRepair()(to_grid(q))
        g = g.reshape(n_ser, n_days_m, -1)
        grids["chronos2-zs"] = [g[:, d, :] for d in range(n_days_m)]
        monthly_grids[oi] = grids
        print(f"  Origin {oi} ({month:%Y-%m}) grids done  ({time.time()-t0:.0f}s)")

    # ---- Select best emp arm per (α, origin) using Y demand ----
    # For each (α, origin), compute L0_static for both emp arms and pick lower.
    base_of: dict[tuple[float, int], str] = {}

    sel_file = cfgmod.ARTIFACT_DIR / "zhao_rolling_origins" / "baseline_selection.csv"
    if not sel_file.exists():
        raise FileNotFoundError(
            f"baseline_selection.csv required: {sel_file}")
    sel = pd.read_csv(sel_file)
    for _, r in sel.iterrows():
        if r.alpha in GRID_ALPHAS:
            base_of[(r.alpha, r.origin_idx)] = r.retuned

    # Verify all (α, origin) combos are covered
    missing = [(a, oi) for a in GRID_ALPHAS for oi in SCORED_ORIGINS
               if (a, oi) not in base_of]
    if missing:
        raise KeyError(
            f"baseline_selection.csv missing {len(missing)} (α, origin) "
            f"combos: {missing[:5]}...")
    print(f"Baseline selection for {len(base_of)} (α, origin) combos "
          f"(all from frozen file)")

    # ---- Precompute policies S for all needed (arm, α) combos ----
    needed_arms = {"chronos2-zs"}
    for alpha in GRID_ALPHAS:
        for oi in SCORED_ORIGINS:
            needed_arms.add(base_of[(alpha, oi)])
    arms = sorted(needed_arms)

    precomp = {}
    for arm in arms:
        for alpha in GRID_ALPHAS:
            review_days_list = []
            S_list = []
            for oi in range(len(ORIGINS)):
                if oi not in monthly_grids:
                    continue
                month = ORIGINS[oi]
                mo = (month - start_date).days
                gr = monthly_grids[oi][arm]
                for dim in range(0, month.days_in_month, R):
                    abs_day = mo + dim
                    if abs_day >= total_days:
                        break
                    review_days_list.append(abs_day)
                    pi_grids = [gr[min(dim + k, len(gr) - 1)]
                                for k in range(PI)]
                    pmf_pi = convolve_varying_pmf(
                        NATIVE_LEVELS, pi_grids, vmax=VMAX)
                    pmf_r = convolve_varying_pmf(
                        NATIVE_LEVELS, pi_grids[:R], vmax=VMAX)
                    S_val = order_up_to(
                        pmf_r, pmf_pi, alpha, PI / R)["P3"]
                    S_list.append(S_val)
            if not review_days_list:
                continue
            S_arr = np.column_stack(S_list)
            S_static_oi = {}
            for oi in SCORED_ORIGINS:
                if oi not in monthly_grids:
                    continue
                month = ORIGINS[oi]
                n_dm = month.days_in_month + LEAD_DAYS
                gr = monthly_grids[oi][arm]
                pmf_pi_s = convolve_varying_pmf(
                    NATIVE_LEVELS, gr[:n_dm], vmax=VMAX)
                pmf_r_s = convolve_varying_pmf(
                    NATIVE_LEVELS, gr[:month.days_in_month], vmax=VMAX)
                m_r = n_dm / month.days_in_month
                S_static_oi[oi] = order_up_to(
                    pmf_r_s, pmf_pi_s, alpha, m_r)["P3"]
            precomp[(arm, alpha)] = (review_days_list, S_arr, S_static_oi)
    print(f"Precomputed S for {len(precomp)} (arm, α) combos  "
          f"({time.time()-t0:.0f}s)")

    # ---- Audit: policy consistency with baseline for α∈{0.95,0.98} ----
    from .run_halfsynthetic import HIGH_ALPHAS as BASELINE_ALPHAS
    baseline_file = (cfgmod.ARTIFACT_DIR / "zhao_halfsynthetic"
                     / "halfsynthetic_long.csv")
    if not baseline_file.exists():
        raise FileNotFoundError(
            f"halfsynthetic_long.csv required for cross-checks: {baseline_file}")
    bdf = pd.read_csv(baseline_file)
    print(f"  Audit: baseline file loaded for post-replay cross-check")

    # ---- Run replays on the (α, λ) grid ----
    # For each grid point, store per-draw cell-level costs for two-way bootstrap.
    # Cell index: (series, origin, α) but here α is a single value per grid point,
    # so cells are (series, origin) — n_ser × len(SCORED_ORIGINS) per point.
    n_oi = len(SCORED_ORIGINS)
    n_cells = n_ser * n_oi

    # Per-draw storage: for each (α, λ), store per-draw arrays of shape
    # (n_draws, n_cells) for zs and emp L0 and L3.
    # For λ=1 (pure Y), all draws are identical — we tile from 1 draw.
    grid_draw_data: dict[tuple[float, float], dict] = {}

    for alpha in GRID_ALPHAS:
        h_c, p_c = costs_from_alpha(cost_i, alpha, KAPPA_H, 12)
        ba_for_oi = {oi: base_of[(alpha, oi)] for oi in SCORED_ORIGINS}

        for lam in LAMBDAS:
            print(f"\n  α={alpha}, λ={lam:.2f}  ", end="", flush=True)

            if lam == 1.0:
                dm_list = [dm_y]
                n_draws_eff = 1
            else:
                X = _build_recensored_demand(
                    dm_y, dm_d, lam, censor_months, sids,
                    start_date, total_days)
                dm_list = list(X)
                n_draws_eff = N_DRAWS

            # Accumulators: per-draw, per-cell
            zs_L0 = np.zeros((n_draws_eff, n_cells))
            emp_L0 = np.zeros((n_draws_eff, n_cells))
            zs_H3 = np.zeros((n_draws_eff, n_cells))
            emp_H3 = np.zeros((n_draws_eff, n_cells))
            zs_P3 = np.zeros((n_draws_eff, n_cells))
            emp_P3 = np.zeros((n_draws_eff, n_cells))

            # Determine which emp arms are needed for this α
            emp_arms_needed = set(ba_for_oi.values())

            for di, dm in enumerate(dm_list):
                # Replay each needed arm once (full horizon)
                replay_results = {}
                for arm in ["chronos2-zs"] + sorted(emp_arms_needed):
                    if (arm, alpha) not in precomp:
                        continue
                    rd_list, S_arr, _ = precomp[(arm, alpha)]
                    rc = ReplayConfig(
                        n_days=total_days, lead_time_days=LEAD_DAYS,
                        review_cadence_days=R,
                        shortage_mechanism="lost_sales",
                        review_days=tuple(rd_list))
                    replay_results[arm] = replay(dm, S_arr, inv0, rc)

                # Extract per-origin costs
                _, _, S_static_oi_zs = precomp[("chronos2-zs", alpha)]
                for oi_local, oi in enumerate(SCORED_ORIGINS):
                    month = ORIGINS[oi]
                    mo = (month - start_date).days
                    me = mo + month.days_in_month
                    if me > total_days or mo < BURN_IN_DAYS:
                        continue
                    cs = oi_local * n_ser
                    ce = cs + n_ser

                    # Chronos
                    res_zs = replay_results["chronos2-zs"]
                    zs_H3[di, cs:ce] = h_c * res_zs.i_end[:, mo:me].mean(axis=1)
                    zs_P3[di, cs:ce] = p_c * res_zs.lost[:, mo:me].sum(axis=1)
                    if oi in S_static_oi_zs:
                        S_st = S_static_oi_zs[oi]
                        pi_end = min(mo + month.days_in_month + LEAD_DAYS,
                                     total_days)
                        y_pi = dm[:, mo:pi_end].sum(axis=1)
                        zs_L0[di, cs:ce] = (
                            h_c * np.maximum(S_st - y_pi, 0)
                            + p_c * np.maximum(y_pi - S_st, 0))

                    # Emp (origin-specific arm)
                    ba = ba_for_oi[oi]
                    _, _, S_static_oi_e = precomp[(ba, alpha)]
                    res_emp = replay_results[ba]
                    emp_H3[di, cs:ce] = h_c * res_emp.i_end[:, mo:me].mean(axis=1)
                    emp_P3[di, cs:ce] = p_c * res_emp.lost[:, mo:me].sum(axis=1)
                    if oi in S_static_oi_e:
                        S_st = S_static_oi_e[oi]
                        pi_end = min(mo + month.days_in_month + LEAD_DAYS,
                                     total_days)
                        y_pi = dm[:, mo:pi_end].sum(axis=1)
                        emp_L0[di, cs:ce] = (
                            h_c * np.maximum(S_st - y_pi, 0)
                            + p_c * np.maximum(y_pi - S_st, 0))

            # For λ=1, tile the single-draw result to N_DRAWS
            if lam == 1.0:
                zs_L0 = np.tile(zs_L0, (N_DRAWS, 1))
                emp_L0 = np.tile(emp_L0, (N_DRAWS, 1))
                zs_H3 = np.tile(zs_H3, (N_DRAWS, 1))
                emp_H3 = np.tile(emp_H3, (N_DRAWS, 1))
                zs_P3 = np.tile(zs_P3, (N_DRAWS, 1))
                emp_P3 = np.tile(emp_P3, (N_DRAWS, 1))

            grid_draw_data[(alpha, lam)] = dict(
                zs_L0=zs_L0, emp_L0=emp_L0,
                zs_H3=zs_H3, emp_H3=emp_H3,
                zs_P3=zs_P3, emp_P3=emp_P3)

            print(f"({time.time()-t0:.0f}s)", flush=True)

    print(f"\nAll replays done  ({time.time()-t0:.0f}s)")

    # ---- Compute point estimates ----
    # For each (α, λ): ratio-of-sums R₀, R₃, A, and DiD(α,λ) = A(α,0) − A(α,λ)
    all_rows = np.arange(n_cells)

    def _ros(numer, denom):
        """Ratio-of-sums × 100."""
        d = denom.sum()
        return numer.sum() / d * 100

    grid_results = []
    # Also store per-draw numerator/denominator arrays for bootstrap
    # d_s = zs_L0 - emp_L0 (static numer), b_s = emp_L0 (static denom)
    # d_d = zs_L3 - emp_L3 (dynamic numer), b_d = emp_L3 (dynamic denom)
    grid_arrays: dict[tuple[float, float], dict] = {}

    for alpha in GRID_ALPHAS:
        for lam in LAMBDAS:
            gd = grid_draw_data[(alpha, lam)]
            zs_L3 = gd["zs_H3"] + gd["zs_P3"]
            emp_L3 = gd["emp_H3"] + gd["emp_P3"]

            # Draw-averaged arrays (n_cells,)
            d_s = (gd["zs_L0"] - gd["emp_L0"]).mean(axis=0)
            b_s = gd["emp_L0"].mean(axis=0)
            d_d = (zs_L3 - emp_L3).mean(axis=0)
            b_d = emp_L3.mean(axis=0)

            R0 = _ros(d_s, b_s)
            R3 = _ros(d_d, b_d)
            A = R3 - R0

            # Per-draw arrays for two-way bootstrap
            grid_arrays[(alpha, lam)] = dict(
                d_s_draw=gd["zs_L0"] - gd["emp_L0"],  # (N_DRAWS, n_cells)
                b_s_draw=gd["emp_L0"],
                d_d_draw=zs_L3 - emp_L3,
                b_d_draw=emp_L3,
                d_s=d_s, b_s=b_s, d_d=d_d, b_d=b_d)

            grid_results.append(dict(
                alpha=alpha, lam=lam, R0=R0, R3=R3, A=A))

    # Compute DiD = A(α,0) − A(α,λ)
    a_at_lam0 = {}
    for r in grid_results:
        if r["lam"] == 0.0:
            a_at_lam0[r["alpha"]] = r["A"]
    for r in grid_results:
        r["DiD"] = a_at_lam0[r["alpha"]] - r["A"]

    # ---- Audit: support ----
    assert n_clu_check == 1135, f"Expected 1135 clusters, got {n_clu_check}"
    expected_cells_per_alpha = n_ser * n_oi
    assert expected_cells_per_alpha == 1135 * 5, (
        f"Expected 5675 cells per α, got {expected_cells_per_alpha}")
    total_grid_cells = len(GRID_ALPHAS) * expected_cells_per_alpha
    assert total_grid_cells == 28375, (
        f"Expected 28375 total cells, got {total_grid_cells}")
    for alpha in GRID_ALPHAS:
        for lam in LAMBDAS:
            ga = grid_arrays[(alpha, lam)]
            assert ga["d_s_draw"].shape == (N_DRAWS, n_cells), (
                f"Wrong shape at α={alpha}, λ={lam}")
    print(f"\n  Audit: support = {n_ser} clusters × {n_oi} origins × "
          f"{len(GRID_ALPHAS)} αs = {total_grid_cells} cells ✓")

    # ---- Audit: re-aggregate α=0.95,0.98 to reproduce baseline DiD ----
    # Pool cell-level sufficient statistics from both αs at λ=0 (D) and λ=1 (Y)
    baseline_alphas = [0.95, 0.98]
    # Y evaluation: λ=1
    d_s_y_pooled = np.concatenate([
        grid_arrays[(a, 1.0)]["d_s"].reshape(1, -1) for a in baseline_alphas
    ], axis=1).ravel()
    b_s_y_pooled = np.concatenate([
        grid_arrays[(a, 1.0)]["b_s"].reshape(1, -1) for a in baseline_alphas
    ], axis=1).ravel()
    d_d_y_pooled = np.concatenate([
        grid_arrays[(a, 1.0)]["d_d"].reshape(1, -1) for a in baseline_alphas
    ], axis=1).ravel()
    b_d_y_pooled = np.concatenate([
        grid_arrays[(a, 1.0)]["b_d"].reshape(1, -1) for a in baseline_alphas
    ], axis=1).ravel()
    # D evaluation: λ=0
    d_s_d_pooled = np.concatenate([
        grid_arrays[(a, 0.0)]["d_s"].reshape(1, -1) for a in baseline_alphas
    ], axis=1).ravel()
    b_s_d_pooled = np.concatenate([
        grid_arrays[(a, 0.0)]["b_s"].reshape(1, -1) for a in baseline_alphas
    ], axis=1).ravel()
    d_d_d_pooled = np.concatenate([
        grid_arrays[(a, 0.0)]["d_d"].reshape(1, -1) for a in baseline_alphas
    ], axis=1).ravel()
    b_d_d_pooled = np.concatenate([
        grid_arrays[(a, 0.0)]["b_d"].reshape(1, -1) for a in baseline_alphas
    ], axis=1).ravel()

    ry_s = _ros(d_s_y_pooled, b_s_y_pooled)
    ry_d = _ros(d_d_y_pooled, b_d_y_pooled)
    rd_s = _ros(d_s_d_pooled, b_s_d_pooled)
    rd_d = _ros(d_d_d_pooled, b_d_d_pooled)
    a_y_pooled = ry_d - ry_s
    a_d_pooled = rd_d - rd_s
    did_pooled = a_d_pooled - a_y_pooled

    print(f"  Audit: re-aggregated α={{0.95,0.98}} baseline reproduction:")
    print(f"    R(Y,Static)={ry_s:+.2f}%  R(Y,Dynamic)={ry_d:+.2f}%  "
          f"A_Y={a_y_pooled:+.2f} pp")
    print(f"    R(D,Static)={rd_s:+.2f}%  R(D,Dynamic)={rd_d:+.2f}%  "
          f"A_D={a_d_pooled:+.2f} pp")
    print(f"    DiD={did_pooled:+.2f} pp  (baseline: -8.28 pp)")

    # Frozen baseline values — recompute unrounded from baseline CSV
    def _ros_from_bdf(demand_tag, metric_col):
        """Ratio-of-sums from halfsynthetic_long for baseline αs."""
        numer, denom = 0.0, 0.0
        for alpha in baseline_alphas:
            for oi in SCORED_ORIGINS:
                emp_arm = base_of[(alpha, oi)]
                sel_zs = bdf[(bdf.demand == demand_tag)
                             & (bdf.alpha == alpha)
                             & (bdf.arm == "chronos2-zs")
                             & (bdf.origin_idx == oi)]
                sel_emp = bdf[(bdf.demand == demand_tag)
                              & (bdf.alpha == alpha)
                              & (bdf.arm == emp_arm)
                              & (bdf.origin_idx == oi)]
                zs_vals = sel_zs.set_index("series_id")[metric_col]
                emp_vals = sel_emp.set_index("series_id")[metric_col]
                common = zs_vals.index.intersection(emp_vals.index)
                numer += (zs_vals.loc[common] - emp_vals.loc[common]).sum()
                denom += emp_vals.loc[common].sum()
        return numer / denom * 100

    bl_ry_s = _ros_from_bdf("Y", "L0_static")
    bl_ry_d = _ros_from_bdf("Y", "L3_longrun")
    bl_rd_s = _ros_from_bdf("D", "L0_static")
    bl_rd_d = _ros_from_bdf("D", "L3_longrun")
    bl_a_y = bl_ry_d - bl_ry_s
    bl_a_d = bl_rd_d - bl_rd_s
    bl_did = bl_a_d - bl_a_y

    grid_vals = np.array([ry_s, ry_d, rd_s, rd_d, a_y_pooled,
                          a_d_pooled, did_pooled])
    bl_vals = np.array([bl_ry_s, bl_ry_d, bl_rd_s, bl_rd_d,
                        bl_a_y, bl_a_d, bl_did])
    np.testing.assert_allclose(
        grid_vals, bl_vals, rtol=0.0, atol=1e-8,
        err_msg="Grid re-aggregation ≠ baseline CSV (unrounded)")
    print(f"  Audit: all 7 re-aggregated values match baseline CSV "
          f"(atol=1e-8) ✓")

    # Cross-check cell values against baseline halfsynthetic_long.csv
    # Check: Chronos Y-arm (λ=1), Emp Y-arm (λ=1), D endpoints (λ=0)
    max_diff = 0.0
    n_checked = 0
    for alpha in BASELINE_ALPHAS:
        # --- Chronos arm on Y (λ=1) and D (λ=0) ---
        for demand_tag, lam_key in [("Y", 1.0), ("D", 0.0)]:
            ga = grid_arrays[(alpha, lam_key)]
            grid_zs_L0 = ga["d_s"] + ga["b_s"]
            grid_zs_L3 = ga["d_d"] + ga["b_d"]
            sel_zs = bdf[(bdf.demand == demand_tag)
                         & (bdf.alpha == alpha)
                         & (bdf.arm == "chronos2-zs")].set_index(
                             ["series_id", "origin_idx"])
            for oi_local, oi in enumerate(SCORED_ORIGINS):
                cs = oi_local * n_ser
                for s_idx, sid in enumerate(sids):
                    if (sid, oi) not in sel_zs.index:
                        continue
                    row = sel_zs.loc[(sid, oi)]
                    for col, arr in [("L0_static", grid_zs_L0),
                                     ("L3_longrun", grid_zs_L3)]:
                        d = abs(float(row[col]) - arr[cs + s_idx])
                        max_diff = max(max_diff, d)
                        n_checked += 1

        # --- Emp arm on Y (λ=1) and D (λ=0): per-origin arm ---
        for demand_tag, lam_key in [("Y", 1.0), ("D", 0.0)]:
            ga_emp = grid_arrays[(alpha, lam_key)]
            grid_emp_L0 = ga_emp["b_s"]
            grid_emp_L3 = ga_emp["b_d"]
            for oi_local, oi in enumerate(SCORED_ORIGINS):
                emp_arm = base_of[(alpha, oi)]
                sel_emp = bdf[(bdf.demand == demand_tag)
                              & (bdf.alpha == alpha)
                              & (bdf.arm == emp_arm)].set_index(
                                  ["series_id", "origin_idx"])
                cs = oi_local * n_ser
                for s_idx, sid in enumerate(sids):
                    if (sid, oi) not in sel_emp.index:
                        continue
                    row = sel_emp.loc[(sid, oi)]
                    for col, arr in [("L0_static", grid_emp_L0),
                                     ("L3_longrun", grid_emp_L3)]:
                        d = abs(float(row[col]) - arr[cs + s_idx])
                        max_diff = max(max_diff, d)
                        n_checked += 1

    assert n_checked == 90_800, f"Expected 90800 cross-checks, got {n_checked}"
    if max_diff > 1e-10:
        raise RuntimeError(
            f"Cell mismatch vs baseline: max_diff={max_diff:.3g}")
    print(f"  Audit: {n_checked} cell values (Chronos+Emp, Y+D) match "
          f"baseline (max_diff={max_diff:.3g}) ✓")

    print(f"\n{'='*78}")
    print("GRID POINT ESTIMATES")
    print(f"{'='*78}")
    print(f"  {'α':>5} {'λ':>5} {'R₀':>8} {'R₃':>8} {'A':>8} {'DiD':>8}")
    print(f"  {'-'*44}")
    for r in grid_results:
        print(f"  {r['alpha']:>5.2f} {r['lam']:>5.2f} "
              f"{r['R0']:>+7.2f}% {r['R3']:>+7.2f}% "
              f"{r['A']:>+7.2f} {r['DiD']:>+7.2f}")

    # ---- Two-way bootstrap with simultaneous max-|t| bands ----
    # Cluster structure: same as halfsynthetic
    series_ids_rep = list(np.tile(sids, n_oi))
    codes, uniq = pd.factorize(series_ids_rep)
    n_clu = len(uniq)
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(n_clu))
    ends = np.searchsorted(codes[order], np.arange(n_clu), side="right")

    n_grid = len(grid_results)
    # 4 estimands per grid point: R0, R3, A, DiD
    n_estimands = n_grid * 4

    rng_b = np.random.default_rng(SEED_BASE + 7777)
    boot_matrix = np.empty((args.b, n_grid, 4))

    for b_idx in range(args.b):
        clu_picks = rng_b.integers(0, n_clu, size=n_clu)
        rows = np.concatenate([order[starts[c]:ends[c]] for c in clu_picks])
        draw_picks = rng_b.integers(0, N_DRAWS, size=N_DRAWS)

        for g_idx, r in enumerate(grid_results):
            ga = grid_arrays[(r["alpha"], r["lam"])]

            avg_d_s = ga["d_s_draw"][draw_picks].mean(axis=0)
            avg_b_s = ga["b_s_draw"][draw_picks].mean(axis=0)
            avg_d_d = ga["d_d_draw"][draw_picks].mean(axis=0)
            avg_b_d = ga["b_d_draw"][draw_picks].mean(axis=0)

            R0_b = avg_d_s[rows].sum() / avg_b_s[rows].sum() * 100
            R3_b = avg_d_d[rows].sum() / avg_b_d[rows].sum() * 100
            A_b = R3_b - R0_b

            boot_matrix[b_idx, g_idx, :3] = [R0_b, R3_b, A_b]

        # DiD: A(α,0) − A(α,λ) — need to reference A at λ=0 from this replicate
        for g_idx, r in enumerate(grid_results):
            # Find the λ=0 grid index for this α
            lam0_idx = next(
                j for j, r2 in enumerate(grid_results)
                if r2["alpha"] == r["alpha"] and r2["lam"] == 0.0)
            boot_matrix[b_idx, g_idx, 3] = (
                boot_matrix[b_idx, lam0_idx, 2]
                - boot_matrix[b_idx, g_idx, 2])

    # Point estimate vector
    pt_matrix = np.array([[r["R0"], r["R3"], r["A"], r["DiD"]]
                          for r in grid_results])

    # ---- Pointwise CIs ----
    ci_lo = np.quantile(boot_matrix, 0.025, axis=0)
    ci_hi = np.quantile(boot_matrix, 0.975, axis=0)

    # ---- Simultaneous max-|t| bands ----
    # Standard errors from bootstrap
    boot_se = boot_matrix.std(axis=0, ddof=1)

    # Verify zero SEs are exactly the 5 mechanical-zero DiD(α,0) entries
    zero_se = ~(np.isfinite(boot_se) & (boot_se > 0))
    expected_zero = np.zeros_like(zero_se, dtype=bool)
    for g_idx, r in enumerate(grid_results):
        if r["lam"] == 0.0:
            expected_zero[g_idx, 3] = True
    assert np.array_equal(zero_se, expected_zero), (
        "Zero/nonfinite bootstrap SE outside mechanical DiD(alpha,0)")
    print(f"  Audit: 5 zero-SE entries are exactly DiD(α,0) ✓")

    # Studentized deviates: t_b = (θ̂*_b − θ̂) / se
    t_matrix = np.zeros_like(boot_matrix)
    np.divide(
        boot_matrix - pt_matrix[None, :, :],
        boot_se[None, :, :],
        out=t_matrix,
        where=(~zero_se)[None, :, :],
    )

    # Family 1: R₀, R₃ (for conversion region)
    # Exclude mechanically-zero entries: none for R₀, R₃
    r0r3_cols = [0, 1]  # R0, R3
    max_abs_t_r0r3 = np.max(
        np.abs(t_matrix[:, :, r0r3_cols].reshape(args.b, -1)), axis=1)
    crit_r0r3 = float(np.quantile(max_abs_t_r0r3, 0.95))

    # Family 2: A, DiD (for mechanism)
    # Exclude DiD(α,0) which is mechanically zero
    a_did_entries = []
    for g_idx, r in enumerate(grid_results):
        a_did_entries.append((g_idx, 2))  # A
        if r["lam"] != 0.0:  # exclude mechanical zeros
            a_did_entries.append((g_idx, 3))  # DiD
    a_did_indices = np.array(a_did_entries)
    max_abs_t_adid = np.max(np.abs(
        np.array([t_matrix[:, gi, ci] for gi, ci in a_did_indices]).T
    ), axis=1)
    crit_adid = float(np.quantile(max_abs_t_adid, 0.95))

    # Simultaneous CIs
    sim_lo_r0r3 = pt_matrix[:, :2] - crit_r0r3 * boot_se[:, :2]
    sim_hi_r0r3 = pt_matrix[:, :2] + crit_r0r3 * boot_se[:, :2]

    sim_lo_adid = pt_matrix[:, 2:] - crit_adid * boot_se[:, 2:]
    sim_hi_adid = pt_matrix[:, 2:] + crit_adid * boot_se[:, 2:]

    # ---- Audit: family dimensions ----
    n_r0r3_dim = n_grid * 2  # 25 R₀ + 25 R₃
    n_adid_dim = len(a_did_entries)
    n_mech_zero = sum(1 for r in grid_results if r["lam"] == 0.0)
    expected_adid = n_grid + (n_grid - n_mech_zero)  # 25 A + 20 DiD
    assert n_r0r3_dim == 50, f"Conversion family: expected 50, got {n_r0r3_dim}"
    assert n_adid_dim == expected_adid == 45, (
        f"Mechanism family: expected 45, got {n_adid_dim}")
    print(f"\n  Audit: conversion family = {n_r0r3_dim} dims, "
          f"mechanism family = {n_adid_dim} dims ✓")

    # Defense: all SEs used in simultaneous families must be finite and >0
    assert np.all(np.isfinite(boot_se[:, :2])), \
        "Non-finite SE in R₀/R₃ family"
    assert np.all(boot_se[:, :2] > 0), \
        f"Zero SE in R₀/R₃ family: {np.argwhere(boot_se[:, :2] == 0)}"
    for gi, ci in a_did_entries:
        assert np.isfinite(boot_se[gi, ci]) and boot_se[gi, ci] > 0, \
            f"SE not positive at grid {gi}, estimand {ci}: {boot_se[gi, ci]}"
    print(f"  Audit: all {n_r0r3_dim + n_adid_dim} family SEs finite and >0 ✓")

    print(f"\n{'='*78}")
    print(f"TWO-WAY BOOTSTRAP (clusters={n_clu}, draws={N_DRAWS}, B={args.b})")
    print(f"{'='*78}")
    print(f"  max-|t| critical values:  R₀,R₃ family: {crit_r0r3:.3f}  "
          f"  A,DiD family: {crit_adid:.3f}")
    print(f"  (pointwise 1.96 for comparison)")

    est_names = ["R₀", "R₃", "A", "DiD"]
    print(f"\n  {'α':>5} {'λ':>5} {'Est':>4} {'Point':>8} "
          f"{'Pointwise 95%':>20} {'Simultaneous 95%':>20} "
          f"{'p':>7}")
    print(f"  {'-'*78}")

    out_rows = []
    for g_idx, r in enumerate(grid_results):
        for e_idx, e_name in enumerate(est_names):
            pt_val = pt_matrix[g_idx, e_idx]
            pw_lo = ci_lo[g_idx, e_idx]
            pw_hi = ci_hi[g_idx, e_idx]
            p2 = _bootstrap_p2(boot_matrix[:, g_idx, e_idx])

            if e_idx < 2:
                s_lo = sim_lo_r0r3[g_idx, e_idx]
                s_hi = sim_hi_r0r3[g_idx, e_idx]
            else:
                s_lo = sim_lo_adid[g_idx, e_idx - 2]
                s_hi = sim_hi_adid[g_idx, e_idx - 2]

            # Skip printing DiD at λ=0 (mechanically zero)
            if e_name == "DiD" and r["lam"] == 0.0:
                continue

            print(f"  {r['alpha']:>5.2f} {r['lam']:>5.2f} {e_name:>4} "
                  f"{pt_val:>+7.2f}% [{pw_lo:>+7.2f},{pw_hi:>+7.2f}] "
                  f"[{s_lo:>+7.2f},{s_hi:>+7.2f}] "
                  f"{p2:>7.4f}")

            out_rows.append(dict(
                alpha=r["alpha"], lam=r["lam"], estimand=e_name,
                point=pt_val, ci_lo=pw_lo, ci_hi=pw_hi,
                sim_lo=s_lo, sim_hi=s_hi,
                p_two_sided_pointwise=p2, n_cells=n_cells, n_clusters=n_clu,
                n_draws=N_DRAWS, bootstrap_draws=args.b,
                natural_censoring_rate=cstats["censoring_rate"],
                recensor_fraction=r["lam"],
                recensor_rate_total=int(np.floor(r["lam"] * n_c)) / (n_ser * len(ORIGINS)),
                recensor_rate_scored_mean=recensor_scored_rates[r["lam"]]["mean"],
                recensor_rate_scored_sd=recensor_scored_rates[r["lam"]]["sd"],
                crit_r0r3=crit_r0r3 if e_idx < 2 else np.nan,
                crit_adid=crit_adid if e_idx >= 2 else np.nan))

    pd.DataFrame(out_rows).to_csv(ART / "grid_results.csv", index=False)

    # ---- Conversion region ----
    print(f"\n{'='*78}")
    print("CONVERSION REGION (simultaneous R₀ < 0 AND R₃ < 0)")
    print(f"{'='*78}")
    for g_idx, r in enumerate(grid_results):
        r0_hi = sim_hi_r0r3[g_idx, 0]
        r3_hi = sim_hi_r0r3[g_idx, 1]
        converts = r0_hi < 0 and r3_hi < 0
        marker = "  ✓" if converts else ""
        print(f"  α={r['alpha']:.2f}, λ={r['lam']:.2f}: "
              f"R₀ upper={r0_hi:>+7.2f}, R₃ upper={r3_hi:>+7.2f}"
              f"{marker}")

    print(f"\nAll results saved to {ART}")
    print(f"Total time: {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
