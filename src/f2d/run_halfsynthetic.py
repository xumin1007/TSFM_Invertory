"""D/Y × static/dynamic 半合成实验。

2×2 factorial:
  - Demand input: Y (observed/censored sales) vs D (latent demand)
  - Evaluation: Static newsvendor loss vs Continuous carry-state replay

D construction: for each series, identify censored days (I_avail was
binding in the logged data).  On those days, draw D from the empirical
distribution of uncensored days, conditional on D ≥ Y.  On uncensored
days, D = Y.

Uses the canonical manifest from run_continuous_replay (1836 common
series before cost eligibility, origins 3–7, SEED_BASE=42).  Analysis
eligibility is fixed before replay by finite unit_cost_hist.

用法:  PYTHONPATH=src python -m f2d.run_halfsynthetic [--n-draws 50]
"""

from __future__ import annotations

import argparse
import hashlib
import time

import numpy as np
import pandas as pd

from . import config as cfgmod
from .aggregation import convolve_varying_pmf
from .checks import CheckResult
from .conventions import SEED_BASE
from .datasets import zhao
from .decision import costs_from_alpha, order_up_to
from .models.chronos import BASE_CHECKPOINT, NATIVE_LEVELS, QuantileRepair, to_grid
from .models.gbdt_grid import make_lean_features
from .simulation import ReplayConfig, replay
from .run_rolling_origins import (ALPHAS, EMP_SCHEMES, HIGH_ALPHAS, ORIGINS,
                                  _emp_grid)

ART = cfgmod.ARTIFACT_DIR / "zhao_halfsynthetic"
CONT_ART = cfgmod.ARTIFACT_DIR / "zhao_continuous"
VMAX = 60
LEAD_DAYS = 1
KAPPA_H = 0.20
R = 30
BURN_IN_DAYS = 2 * (R + LEAD_DAYS)  # 62
SCORED_ORIGINS = list(range(3, 8))
B = 10_000


def _bootstrap_p2(draws: np.ndarray) -> float:
    """Plus-one corrected, two-sided bootstrap sign p-value."""
    x = np.asarray(draws)
    b = len(x)
    p_hi = (1 + np.count_nonzero(x >= 0)) / (b + 1)
    p_lo = (1 + np.count_nonzero(x <= 0)) / (b + 1)
    return min(2.0 * min(p_hi, p_lo), 1.0)


def _series_draw_seed(base_seed: int, series_id, draw_id: int) -> int:
    """Deterministic seed keyed by (base, series, draw) via SHA-256."""
    blob = f"{base_seed}|{series_id}|{draw_id}".encode()
    return int(hashlib.sha256(blob).hexdigest()[:16], 16)


def _construct_latent_demand(daily: pd.DataFrame, panel: pd.DataFrame,
                             sids: np.ndarray, sku_of: dict,
                             start: pd.Timestamp, total_days: int,
                             n_draws: int, base_seed: int
                             ) -> tuple[np.ndarray, np.ndarray, dict]:
    """Construct Y (observed) and D (latent) demand matrices.

    Each (series_id, draw_id) pair gets an independent RNG keyed via
    SHA-256, so the draws for any given series are invariant to which
    other series are in the set.

    Returns:
        dm_y: (n_ser, total_days) observed sales
        dm_d: (n_draws, n_ser, total_days) latent demand draws
        stats: censoring statistics
    """
    n_ser = len(sids)
    sid_idx = {s: i for i, s in enumerate(sids)}
    end = start + pd.Timedelta(days=total_days - 1)

    # Build Y matrix
    dm_y = np.zeros((n_ser, total_days))
    sub = daily[(daily.d >= start) & (daily.d <= end)].copy()
    sub = sub[sub.series_id.isin(sid_idx)]
    rows_i = sub.series_id.map(sid_idx).to_numpy()
    cols_i = ((sub.d - start).dt.days).to_numpy()
    vals = sub.y.to_numpy(float)
    valid = (cols_i >= 0) & (cols_i < total_days)
    dm_y[rows_i[valid], cols_i[valid]] = vals[valid]

    # Monthly censoring flags
    censor_months = {}  # (series_idx, origin_idx) -> bool
    n_censored_months = 0
    n_total_months = 0
    for oi in range(len(ORIGINS)):
        month = ORIGINS[oi]
        snap = panel[panel.month == month].set_index("sku_ID")
        for i, s in enumerate(sids):
            sk = sku_of.get(s)
            if sk not in snap.index:
                continue
            row = snap.loc[sk]
            inv = float(row.beginning_inventory)
            sales = float(row.observed_sales_next_month)
            censored = sales >= inv and inv > 0
            censor_months[(i, oi)] = censored
            n_total_months += 1
            if censored:
                n_censored_months += 1

    print(f"  Monthly censoring: {n_censored_months}/{n_total_months} "
          f"= {n_censored_months/max(n_total_months,1)*100:.1f}%")

    # Per-series empirical distribution of daily demand from uncensored months
    uncensored_demand = {}
    for i in range(n_ser):
        vals_i = []
        for oi in range(len(ORIGINS)):
            if censor_months.get((i, oi), False):
                continue
            month = ORIGINS[oi]
            mo = (month - start).days
            me = mo + month.days_in_month
            if mo >= 0 and me <= total_days:
                vals_i.extend(dm_y[i, mo:me].tolist())
        uncensored_demand[i] = np.array(vals_i) if vals_i else dm_y[i]

    # Construct D draws with per-(series, draw) keyed RNG
    dm_d = np.tile(dm_y, (n_draws, 1, 1))  # (n_draws, n_ser, total_days)

    n_inflated = 0
    for i in range(n_ser):
        ud = uncensored_demand[i]
        if len(ud) == 0 or ud.max() == 0:
            continue
        sid = sids[i]
        censored_days = []
        for oi in range(len(ORIGINS)):
            if not censor_months.get((i, oi), False):
                continue
            month = ORIGINS[oi]
            mo = (month - start).days
            me = mo + month.days_in_month
            if mo < 0 or me > total_days:
                continue
            for t in range(mo, me):
                if dm_y[i, t] > 0:
                    censored_days.append(t)
        if not censored_days:
            continue
        n_inflated += len(censored_days)
        for k in range(n_draws):
            rng_sk = np.random.default_rng(
                _series_draw_seed(base_seed, sid, k))
            for t in censored_days:
                y_t = dm_y[i, t]
                candidates = ud[ud >= y_t]
                if len(candidates) == 0:
                    mean_unc = ud[ud > 0].mean() if (ud > 0).any() else 1.0
                    excess = rng_sk.geometric(1.0 / max(mean_unc, 1.0))
                    dm_d[k, i, t] = y_t + excess
                else:
                    dm_d[k, i, t] = rng_sk.choice(candidates)

    avg_inflate = 0.0
    if n_inflated > 0:
        mask_c = dm_d.mean(axis=0) != dm_y
        if mask_c.any():
            avg_inflate = (dm_d.mean(axis=0)[mask_c] / np.clip(dm_y[mask_c], 1, None)).mean()

    stats = dict(
        n_censored_months=n_censored_months,
        n_total_months=n_total_months,
        censoring_rate=n_censored_months / max(n_total_months, 1),
        n_inflated_day_series=n_inflated,
        avg_inflate_ratio=float(avg_inflate),
        censor_months=censor_months)
    return dm_y, dm_d, stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-draws", type=int, default=50,
                    help="Number of latent-demand draws")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--b", type=int, default=B)
    args = ap.parse_args(argv)

    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    chk = CheckResult(step_id="zhao_halfsynthetic", dataset="zhao",
                      seed=SEED_BASE)

    # Load canonical series from continuous replay
    series_file = CONT_ART / "common_series.csv"
    if not series_file.exists():
        print("ERROR: Run run_continuous_replay first to generate common_series.csv")
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

    # Build history contexts
    hist_all = {}
    for oi, month in enumerate(ORIGINS):
        h = daily[daily.d < month]
        hist_all[oi] = {s: g.y.to_numpy()
                        for s, g in h.groupby("series_id")
                        if len(g) >= 30}

    # Cost vectors from March panel
    snap_march = panel[panel.month == ORIGINS[0]].set_index("sku_ID")
    common_sk = np.array([sku_of[s] for s in common_sids])
    common_inv0 = snap_march.loc[
        common_sk, "beginning_inventory"].to_numpy(float)
    common_cost = snap_march.loc[
        common_sk, "unit_cost_hist"].to_numpy(float)

    # Predeclared eligibility: common-series manifest plus finite unit cost.
    # This is fixed before any Y/D outcome is replayed.
    cost_finite = np.isfinite(common_cost)
    eligibility = pd.DataFrame({
        "series_id": common_sids,
        "finite_unit_cost": cost_finite,
        "eligible": cost_finite})
    eligibility.to_csv(ART / "analysis_eligibility.csv", index=False)

    sids = common_sids[cost_finite]
    sk = common_sk[cost_finite]
    inv0 = common_inv0[cost_finite]
    cost_i = common_cost[cost_finite]
    n_ser = len(sids)
    daily = daily[daily.series_id.isin(sids)]
    eligible_hash = hashlib.sha256(
        ",".join(str(s) for s in sids).encode()).hexdigest()[:12]
    print(f"Cost-eligible series: {n_ser}/{n_common} "
          f"(excluded nonfinite unit_cost={n_common - n_ser}, "
          f"hash={eligible_hash})")
    if not np.isfinite(inv0).all():
        raise RuntimeError("Nonfinite initial inventory on the fixed eligible set")

    # Demand matrices
    start_date = ORIGINS[0]
    end_date = (ORIGINS[-1] + pd.DateOffset(months=1)
                + pd.Timedelta(days=LEAD_DAYS - 1))
    total_days = (end_date - start_date).days + 1

    dm_y, dm_d, cstats = _construct_latent_demand(
        daily, panel, sids, sku_of, start_date, total_days,
        args.n_draws, SEED_BASE)
    print(f"  Censoring rate: {cstats['censoring_rate']:.1%}")
    print(f"  Avg inflate ratio: {cstats['avg_inflate_ratio']:.2f}")
    print(f"  Demand matrices built  ({time.time()-t0:.0f}s)")

    # Load baseline selection
    sel = pd.read_csv(cfgmod.ARTIFACT_DIR / "zhao_rolling_origins"
                      / "baseline_selection.csv")
    base_of = {(r.alpha, r.origin_idx): r.retuned for _, r in sel.iterrows()}

    # Load Chronos
    from chronos import BaseChronosPipeline
    import torch
    print("Loading Chronos-2 …")
    from .models.chronos import BASE_REVISION
    pipe = BaseChronosPipeline.from_pretrained(
        BASE_CHECKPOINT, revision=BASE_REVISION, device_map=args.device)

    # Precompute grids (same for Y and D — forecasts are trained on Y)
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

    # ---- Main 2×2 experiment ----
    # Only run chronos2-zs + per-origin retuned baseline, HIGH_ALPHAS only
    PI = R + LEAD_DAYS
    needed_arms = {"chronos2-zs"}
    for alpha in HIGH_ALPHAS:
        for oi in SCORED_ORIGINS:
            ba = base_of.get((alpha, oi))
            if ba:
                needed_arms.add(ba)
    arms = sorted(needed_arms)
    print(f"Arms to run: {arms}")
    long_rows = []

    # Precompute S values and static S per (arm, alpha, origin) — independent of demand
    precomp = {}  # (arm, alpha) -> (review_days_list, S_arr, {oi: S_static})
    for arm in arms:
        for alpha in HIGH_ALPHAS:
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
    print(f"Precomputed S for {len(precomp)} (arm, alpha) combos  "
          f"({time.time()-t0:.0f}s)")

    # Per-draw D arrays for hierarchical bootstrap.
    # Key: (arm, alpha, oi, metric) -> (n_draws, n_ser)
    d_draw_arrays: dict[tuple, np.ndarray] = {}
    n_d_draws = len(dm_d)

    for demand_label, dm_list in [("Y", [dm_y]), ("D", list(dm_d))]:
        n_draws_eff = len(dm_list)
        print(f"\n{'='*74}")
        print(f"Demand = {demand_label}  ({n_draws_eff} draws)")
        print(f"{'='*74}")

        for arm in arms:
            for alpha in HIGH_ALPHAS:
                if (arm, alpha) not in precomp:
                    continue
                review_days_list, S_arr, S_static_oi = precomp[(arm, alpha)]
                h_c, p_c = costs_from_alpha(cost_i, alpha, KAPPA_H, 12)

                z = lambda: {oi: np.zeros(n_ser) for oi in SCORED_ORIGINS}
                l0_by_oi = z()
                lr_by_oi, h3_by_oi, p3_by_oi = z(), z(), z()
                inv_by_oi, lost_by_oi = z(), z()

                # Per-draw accumulators for D
                if demand_label == "D":
                    draw_h3 = {oi: np.zeros((n_draws_eff, n_ser))
                               for oi in SCORED_ORIGINS}
                    draw_p3 = {oi: np.zeros((n_draws_eff, n_ser))
                               for oi in SCORED_ORIGINS}
                    draw_l0 = {oi: np.zeros((n_draws_eff, n_ser))
                               for oi in SCORED_ORIGINS}

                for di, dm in enumerate(dm_list):
                    rc = ReplayConfig(
                        n_days=total_days, lead_time_days=LEAD_DAYS,
                        review_cadence_days=R,
                        shortage_mechanism="lost_sales",
                        review_days=tuple(review_days_list))
                    res_lr = replay(dm, S_arr, inv0, rc)

                    for oi in SCORED_ORIGINS:
                        month = ORIGINS[oi]
                        mo = (month - start_date).days
                        me = mo + month.days_in_month
                        if me > total_days or mo < BURN_IN_DAYS:
                            continue
                        h_part = h_c * res_lr.i_end[:, mo:me].mean(axis=1)
                        p_part = p_c * res_lr.lost[:, mo:me].sum(axis=1)
                        lr_by_oi[oi] += (h_part + p_part) / n_draws_eff
                        h3_by_oi[oi] += h_part / n_draws_eff
                        p3_by_oi[oi] += p_part / n_draws_eff
                        inv_by_oi[oi] += res_lr.i_end[:, mo:me].mean(axis=1) / n_draws_eff
                        lost_by_oi[oi] += res_lr.lost[:, mo:me].sum(axis=1) / n_draws_eff

                        if demand_label == "D":
                            draw_h3[oi][di] = h_part
                            draw_p3[oi][di] = p_part

                        if oi in S_static_oi:
                            S_st = S_static_oi[oi]
                            pi_end = min(mo + month.days_in_month + LEAD_DAYS,
                                         total_days)
                            y_pi = dm[:, mo:pi_end].sum(axis=1)
                            l0 = (h_c * np.maximum(S_st - y_pi, 0)
                                  + p_c * np.maximum(y_pi - S_st, 0))
                            l0_by_oi[oi] += l0 / n_draws_eff
                            if demand_label == "D":
                                draw_l0[oi][di] = l0

                if demand_label == "D":
                    for oi in SCORED_ORIGINS:
                        d_draw_arrays[(arm, alpha, oi, "H3")] = draw_h3[oi]
                        d_draw_arrays[(arm, alpha, oi, "P3")] = draw_p3[oi]
                        d_draw_arrays[(arm, alpha, oi, "L0")] = draw_l0[oi]

                for oi in SCORED_ORIGINS:
                    rec = pd.DataFrame(dict(
                        series_id=sids, origin_idx=oi, alpha=alpha,
                        arm=arm, demand=demand_label,
                        L0_static=l0_by_oi[oi],
                        L3_longrun=lr_by_oi[oi],
                        H3=h3_by_oi[oi], P3=p3_by_oi[oi],
                        avg_inv=inv_by_oi[oi],
                        lost_units=lost_by_oi[oi]))
                    metric_cols = ["L0_static", "L3_longrun", "H3", "P3",
                                   "avg_inv", "lost_units"]
                    bad = {
                        col: int(np.count_nonzero(~np.isfinite(rec[col])))
                        for col in metric_cols
                        if not np.isfinite(rec[col]).all()
                    }
                    if bad:
                        raise RuntimeError(
                            "Nonfinite outcomes on the predeclared eligible "
                            f"support (demand={demand_label}, arm={arm}, "
                            f"alpha={alpha}, origin={oi}): {bad}")
                    long_rows.append(rec)

            print(f"  {arm}  ({time.time()-t0:.0f}s)")

    df = pd.concat(long_rows, ignore_index=True)
    df.to_csv(ART / "halfsynthetic_long.csv", index=False)
    print(f"\nSaved {len(df)} rows  ({time.time()-t0:.0f}s)")

    # ======== ANALYSIS (manifest-based eligibility) ========
    print(f"\n{'='*78}")
    print("2×2 FACTORIAL ANALYSIS (manifest-based eligibility)")
    print(f"{'='*78}")

    metric_cols = ["L0_static", "L3_longrun", "H3", "P3",
                   "avg_inv", "lost_units"]
    storage_key = ["series_id", "origin_idx", "alpha", "arm", "demand"]
    expected_rows = (n_ser * len(SCORED_ORIGINS) * len(HIGH_ALPHAS)
                     * len(arms) * 2)
    if len(df) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} stored eligible rows, got {len(df)}")
    if df.duplicated(storage_key).any():
        raise RuntimeError("Duplicate rows on the stored analysis key")
    nonfinite = {
        col: int(np.count_nonzero(~np.isfinite(df[col])))
        for col in metric_cols if not np.isfinite(df[col]).all()
    }
    if nonfinite:
        raise RuntimeError(
            f"Nonfinite stored outcomes on fixed eligibility: {nonfinite}")
    hp_error = np.max(np.abs(df.L3_longrun - df.H3 - df.P3))
    if hp_error > 1e-8:
        raise RuntimeError(
            f"L3 != H3 + P3 (maximum absolute error={hp_error:.3g})")

    expected_index = pd.MultiIndex.from_product(
        [sids, SCORED_ORIGINS, HIGH_ALPHAS],
        names=["series_id", "origin_idx", "alpha"])
    expected_cells = len(expected_index)
    key_blob = "\n".join(
        "|".join(map(str, key)) for key in expected_index.to_list())
    cell_hash = hashlib.sha256(key_blob.encode()).hexdigest()[:16]
    n_zero_l0 = int(np.count_nonzero(df.L0_static.to_numpy() == 0))
    print(f"  Common-series manifest: {n_common} series")
    print(f"  Cost-eligible support: {n_ser} series; "
          f"excluded nonfinite unit_cost={n_common - n_ser}")
    print(f"  Predeclared policy-pair cells: {expected_cells} "
          f"(hash={cell_hash})")
    print(f"  Stored rows: {len(df)}; nonfinite outcomes: 0; "
          f"zero L0 rows retained: {n_zero_l0}")
    print(f"  H/P identity max error: {hp_error:.3g}")

    # Build joint table: merge Y and D for each (series, origin, alpha)
    # using the correct origin-specific baseline
    def _extract(sub_df, cols=metric_cols):
        observed_index = pd.MultiIndex.from_frame(
            sub_df[["series_id", "origin_idx", "alpha"]].drop_duplicates())
        missing = expected_index.difference(observed_index)
        extra = observed_index.difference(expected_index)
        if len(missing) or len(extra):
            raise RuntimeError(
                "Stored policy-pair support differs from the predeclared "
                f"support (missing={len(missing)}, extra={len(extra)})")
        recs = []
        for (sid, oi, alpha), grp in sub_df.groupby(
                ["series_id", "origin_idx", "alpha"]):
            ba = base_of.get((alpha, oi))
            if ba is None:
                raise RuntimeError(
                    f"Missing frozen baseline for alpha={alpha}, origin={oi}")
            zs = grp[grp.arm == "chronos2-zs"]
            emp = grp[grp.arm == ba]
            if len(zs) != 1 or len(emp) != 1:
                raise RuntimeError(
                    "Expected exactly one row per selected policy "
                    f"(series={sid}, origin={oi}, alpha={alpha}, "
                    f"n_zs={len(zs)}, n_emp={len(emp)})")
            r = dict(series_id=sid, origin_idx=oi, alpha=alpha)
            for pf, row in [("zs_", zs.iloc[0]), ("emp_", emp.iloc[0])]:
                for col in cols:
                    r[pf + col] = float(row[col])
            recs.append(r)
        out = pd.DataFrame(recs).set_index(
            ["series_id", "origin_idx", "alpha"]).reindex(expected_index)
        value_cols = [f"{pf}{col}" for pf in ("zs_", "emp_") for col in cols]
        if not np.isfinite(out[value_cols].to_numpy()).all():
            raise RuntimeError(
                "Missing/nonfinite selected-policy values after exact reindex")
        return out

    ey = _extract(df[df.demand == "Y"])
    ed = _extract(df[df.demand == "D"])
    if not ey.index.equals(expected_index) or not ed.index.equals(expected_index):
        raise RuntimeError("Y/D analysis indexes do not equal fixed eligibility")
    jy, jd = ey.copy(), ed.copy()
    common_idx = expected_index
    n_j = expected_cells
    n_clu_j = len(sids)
    print(f"  Eligible cells: Y={len(ey)}, D={len(ed)}; missing: 0/0")
    print(f"  Joint deletions: 0; analysis cells: {n_j}; "
          f"clusters: {n_clu_j}")

    if n_j == 0:
        chk.n_rows = len(df)
        chk.finish(ART / "checks")
        return 0

    # Cross-script Y audit.  The R=30 continuous replay and this Y arm use
    # the same demand, forecasts, policies, cost vectors, and scoring windows.
    # Any discrepancy must therefore be support/aggregation related; cell
    # values are required to agree on the fixed eligible support.
    continuous_file = CONT_ART / "continuous_long_v2.csv"
    if continuous_file.exists():
        cdf = pd.read_csv(continuous_file)
        cdf = cdf[(cdf.R == R)
                  & cdf.series_id.isin(sids)
                  & cdf.origin_idx.isin(SCORED_ORIGINS)
                  & cdf.alpha.isin(HIGH_ALPHAS)]
        cy = _extract(cdf, cols=["L0_static", "L3_longrun"])
        audit_cols = [
            "zs_L0_static", "emp_L0_static",
            "zs_L3_longrun", "emp_L3_longrun"]
        y_vals = jy[audit_cols].to_numpy()
        c_vals = cy[audit_cols].to_numpy()
        exact = np.array_equal(y_vals, c_vals)
        max_abs = float(np.max(np.abs(y_vals - c_vals)))
        n_diff = int(np.count_nonzero(y_vals != c_vals))
        if not np.allclose(y_vals, c_vals, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                "Continuous and half-synthetic Y cell values disagree "
                f"(max_abs={max_abs:.3g}, n_diff={n_diff})")
        print(f"  Cross-script Y audit: exact={exact}; "
              f"max_abs={max_abs:.3g}; differing values={n_diff}")
    else:
        print("  Cross-script Y audit skipped: continuous_long_v2.csv not found")

    # Ratio-of-sums arrays
    for pf, jx in [("y_", jy), ("d_", jd)]:
        jx["d_s"] = jx.zs_L0_static - jx.emp_L0_static
        jx["b_s"] = jx.emp_L0_static
        jx["d_d"] = jx.zs_L3_longrun - jx.emp_L3_longrun
        jx["b_d"] = jx.emp_L3_longrun

    d_ys, b_ys = jy.d_s.to_numpy(), jy.b_s.to_numpy()
    d_yd, b_yd = jy.d_d.to_numpy(), jy.b_d.to_numpy()
    d_ds, b_ds = jd.d_s.to_numpy(), jd.b_s.to_numpy()
    d_dd, b_dd = jd.d_d.to_numpy(), jd.b_d.to_numpy()

    # Per-draw D cell arrays for hierarchical bootstrap.
    # Shape: (n_d_draws, n_cells) aligned with expected_index.
    zs_L0_draw = np.zeros((n_d_draws, n_j))
    emp_L0_draw = np.zeros((n_d_draws, n_j))
    zs_H3_draw = np.zeros((n_d_draws, n_j))
    emp_H3_draw = np.zeros((n_d_draws, n_j))
    zs_P3_draw = np.zeros((n_d_draws, n_j))
    emp_P3_draw = np.zeros((n_d_draws, n_j))

    cell = 0
    for s_idx in range(n_ser):
        for oi in SCORED_ORIGINS:
            for alpha in HIGH_ALPHAS:
                ba = base_of[(alpha, oi)]
                zs_L0_draw[:, cell] = d_draw_arrays[
                    ("chronos2-zs", alpha, oi, "L0")][:, s_idx]
                emp_L0_draw[:, cell] = d_draw_arrays[
                    (ba, alpha, oi, "L0")][:, s_idx]
                zs_H3_draw[:, cell] = d_draw_arrays[
                    ("chronos2-zs", alpha, oi, "H3")][:, s_idx]
                emp_H3_draw[:, cell] = d_draw_arrays[
                    (ba, alpha, oi, "H3")][:, s_idx]
                zs_P3_draw[:, cell] = d_draw_arrays[
                    ("chronos2-zs", alpha, oi, "P3")][:, s_idx]
                emp_P3_draw[:, cell] = d_draw_arrays[
                    (ba, alpha, oi, "P3")][:, s_idx]
                cell += 1
    assert cell == n_j

    zs_L3_draw = zs_H3_draw + zs_P3_draw
    emp_L3_draw = emp_H3_draw + emp_P3_draw
    d_ds_draw = zs_L0_draw - emp_L0_draw   # (n_d_draws, n_cells)
    b_ds_draw = emp_L0_draw
    d_dd_draw = zs_L3_draw - emp_L3_draw
    b_dd_draw = emp_L3_draw

    # Verify per-draw means match the stored draw-averaged values
    assert np.allclose(d_ds_draw.mean(axis=0), d_ds, atol=1e-10)
    assert np.allclose(b_ds_draw.mean(axis=0), b_ds, atol=1e-10)
    assert np.allclose(d_dd_draw.mean(axis=0), d_dd, atol=1e-10)
    assert np.allclose(b_dd_draw.mean(axis=0), b_dd, atol=1e-10)

    def _ratio_of_sums(d, b, rows):
        denom = b[rows].sum()
        if not np.isfinite(denom) or denom <= 0:
            raise RuntimeError(
                f"Nonpositive/nonfinite aggregate denominator: {denom}")
        return d[rows].sum() / denom * 100

    def _compute(rows):
        ys = _ratio_of_sums(d_ys, b_ys, rows)
        yd = _ratio_of_sums(d_yd, b_yd, rows)
        ds = _ratio_of_sums(d_ds, b_ds, rows)
        dd = _ratio_of_sums(d_dd, b_dd, rows)
        a_y = yd - ys
        a_d = dd - ds
        did = a_d - a_y
        return (ys, yd, ds, dd, a_y, a_d, did)

    pt = _compute(np.arange(n_j))

    # ---- 2×2 summary ----
    print(f"\n  {'':20} {'Static':>12} {'Dynamic':>12} {'Attenuation':>14}")
    print(f"  {'-'*60}")
    for i, dl in enumerate(["Y", "D"]):
        s_i, d_i = i * 2, i * 2 + 1
        att = pt[d_i] - pt[s_i]
        print(f"  Demand={dl:3}          {pt[s_i]:>+10.2f}%  "
              f"{pt[d_i]:>+10.2f}%  {att:>+10.2f} pp")

    # ---- Hierarchical bootstrap ----
    # Y quantities: cluster-only (1 draw).
    # D quantities: jointly resample clusters AND D-draw IDs.
    # All estimands share the same cluster picks per replicate.
    series_ids = [idx[0] for idx in common_idx]
    codes_j, uniq_j = pd.factorize(series_ids)
    n_clu = len(uniq_j)
    order_j = np.argsort(codes_j, kind="stable")
    starts_j = np.searchsorted(codes_j[order_j], np.arange(n_clu))
    ends_j = np.searchsorted(codes_j[order_j], np.arange(n_clu), side="right")

    rng_b = np.random.default_rng(SEED_BASE + 5555)
    draws = np.empty((args.b, 7))
    for k in range(args.b):
        picks = rng_b.integers(0, n_clu, size=n_clu)
        rows = np.concatenate([order_j[starts_j[c]:ends_j[c]]
                               for c in picks])
        draw_picks = rng_b.integers(0, n_d_draws, size=n_d_draws)

        ys = _ratio_of_sums(d_ys, b_ys, rows)
        yd = _ratio_of_sums(d_yd, b_yd, rows)

        avg_d_ds = d_ds_draw[draw_picks].mean(axis=0)
        avg_b_ds = b_ds_draw[draw_picks].mean(axis=0)
        avg_d_dd = d_dd_draw[draw_picks].mean(axis=0)
        avg_b_dd = b_dd_draw[draw_picks].mean(axis=0)
        ds = avg_d_ds[rows].sum() / avg_b_ds[rows].sum() * 100
        dd = avg_d_dd[rows].sum() / avg_b_dd[rows].sum() * 100

        a_y = yd - ys
        a_d = dd - ds
        did = a_d - a_y
        draws[k] = (ys, yd, ds, dd, a_y, a_d, did)

    labels = ["R(Y,Static)", "R(Y,Dynamic)", "R(D,Static)", "R(D,Dynamic)",
              "A_Y = Dyn-Stat|Y", "A_D = Dyn-Stat|D",
              "DiD = A_D - A_Y"]
    units = ["%", "%", "%", "%", " pp", " pp", " pp"]

    print(f"\n  Two-way bootstrap (clusters={n_clu}, draws={n_d_draws}, "
          f"B={args.b}):")
    print(f"  {'Metric':<24} {'Point':>8} {'95% CI':>22} "
          f"{'p (2-sided)':>12}")
    print(f"  {'-'*70}")
    factorial_rows = []
    for i, (nm, u) in enumerate(zip(labels, units)):
        lo = float(np.quantile(draws[:, i], .025))
        hi = float(np.quantile(draws[:, i], .975))
        p2 = _bootstrap_p2(draws[:, i])
        print(f"  {nm:<24} {pt[i]:>+7.2f}{u}  "
              f"[{lo:>+7.2f}, {hi:>+7.2f}]  p={p2:.4f}")
        factorial_rows.append(dict(
            metric=nm, unit=u.strip(), point=pt[i], ci_low=lo,
            ci_high=hi, p_two_sided=p2, n_cells=n_j,
            n_clusters=n_clu, n_d_draws=n_d_draws,
            bootstrap_type="two-way",
            bootstrap_draws=args.b))
        chk.note(nm.split("=")[0].strip().replace(" ", "_").lower(),
                 round(float(pt[i]), 3))
    pd.DataFrame(factorial_rows).to_csv(
        ART / "factorial_summary.csv", index=False)

    # ---- ARM-LEVEL COST BRIDGE with H/P decomposition ----
    print(f"\n{'='*78}")
    print("ARM-LEVEL COST BRIDGE (H/P decomposition)")
    print(f"{'='*78}")

    print(f"\n  Per-arm absolute costs (mean per cell):")
    print(f"  {'Arm':<10} {'Demand':>6} {'L3':>8} {'H3':>8} {'P3':>8} "
          f"{'AvgInv':>8} {'Lost':>8}")
    print(f"  {'-'*58}")
    arm_rows = []
    for dl, jx in [("Y", jy), ("D", jd)]:
        for pf, lab in [("zs_", "Chronos"), ("emp_", "Emp")]:
            l3 = jx[pf + "L3_longrun"].mean()
            h3 = jx[pf + "H3"].mean()
            p3 = jx[pf + "P3"].mean()
            inv = jx[pf + "avg_inv"].mean()
            lost = jx[pf + "lost_units"].mean()
            print(f"  {lab:<10} {dl:>6} {l3:>8.2f} {h3:>8.2f} {p3:>8.2f} "
                  f"{inv:>8.2f} {lost:>8.3f}")
            arm_rows.append(dict(
                arm=lab, demand=dl, total_cost=l3, holding_cost=h3,
                shortage_cost=p3, average_inventory=inv,
                lost_units=lost, n_cells=n_j))
    pd.DataFrame(arm_rows).to_csv(
        ART / "arm_level_costs.csv", index=False)

    # Censoring-bias bridge:
    # B_m = C_m(Y)-C_m(D) = B_m^H+B_m^P and
    # ΔB = B_ZS-B_Emp = ΔB^H+ΔB^P.
    bridge_arrays = {}
    for pf, lab in [("zs_", "Chronos"), ("emp_", "Emp")]:
        bridge_arrays[(lab, "Total")] = (
            jy[pf + "L3_longrun"] - jd[pf + "L3_longrun"]).to_numpy()
        bridge_arrays[(lab, "Holding")] = (
            jy[pf + "H3"] - jd[pf + "H3"]).to_numpy()
        bridge_arrays[(lab, "Shortage")] = (
            jy[pf + "P3"] - jd[pf + "P3"]).to_numpy()
    for component in ["Total", "Holding", "Shortage"]:
        bridge_arrays[("Delta", component)] = (
            bridge_arrays[("Chronos", component)]
            - bridge_arrays[("Emp", component)])

    for lab in ["Chronos", "Emp", "Delta"]:
        err = np.max(np.abs(
            bridge_arrays[(lab, "Total")]
            - bridge_arrays[(lab, "Holding")]
            - bridge_arrays[(lab, "Shortage")]))
        if err > 1e-8:
            raise RuntimeError(
                f"Bridge identity failed for {lab} (max error={err:.3g})")

    # Per-draw bridge arrays for hierarchical bootstrap.
    # B_m(draw k) = C_m(Y) - C_m(D, draw k); Y is fixed.
    y_zs_L3 = jy.zs_L3_longrun.to_numpy()
    y_emp_L3 = jy.emp_L3_longrun.to_numpy()
    y_zs_H3 = jy.zs_H3.to_numpy()
    y_emp_H3 = jy.emp_H3.to_numpy()
    y_zs_P3 = jy.zs_P3.to_numpy()
    y_emp_P3 = jy.emp_P3.to_numpy()

    bridge_draw_arrays = {}
    bridge_draw_arrays[("Chronos", "Total")] = (
        y_zs_L3[None, :] - zs_L3_draw)
    bridge_draw_arrays[("Chronos", "Holding")] = (
        y_zs_H3[None, :] - zs_H3_draw)
    bridge_draw_arrays[("Chronos", "Shortage")] = (
        y_zs_P3[None, :] - zs_P3_draw)
    bridge_draw_arrays[("Emp", "Total")] = (
        y_emp_L3[None, :] - emp_L3_draw)
    bridge_draw_arrays[("Emp", "Holding")] = (
        y_emp_H3[None, :] - emp_H3_draw)
    bridge_draw_arrays[("Emp", "Shortage")] = (
        y_emp_P3[None, :] - emp_P3_draw)
    for component in ["Total", "Holding", "Shortage"]:
        bridge_draw_arrays[("Delta", component)] = (
            bridge_draw_arrays[("Chronos", component)]
            - bridge_draw_arrays[("Emp", component)])

    bridge_keys = [
        (lab, component)
        for lab in ["Chronos", "Emp", "Delta"]
        for component in ["Total", "Holding", "Shortage"]]
    bridge_matrix = np.column_stack([bridge_arrays[k] for k in bridge_keys])
    bridge_draws = np.empty((args.b, len(bridge_keys)))
    rng_b2 = np.random.default_rng(SEED_BASE + 6666)
    for k in range(args.b):
        picks = rng_b2.integers(0, n_clu, size=n_clu)
        rows = np.concatenate([order_j[starts_j[c]:ends_j[c]]
                               for c in picks])
        draw_picks = rng_b2.integers(0, n_d_draws, size=n_d_draws)
        for i, bk in enumerate(bridge_keys):
            avg_bridge = bridge_draw_arrays[bk][draw_picks].mean(axis=0)
            bridge_draws[k, i] = avg_bridge[rows].mean()
        # Per-replicate H/P identity: Total = Holding + Shortage
        for j, lab in enumerate(["Chronos", "Emp", "Delta"]):
            t_idx, h_idx, s_idx = j * 3, j * 3 + 1, j * 3 + 2
            err = abs(bridge_draws[k, t_idx]
                      - bridge_draws[k, h_idx] - bridge_draws[k, s_idx])
            if err > 1e-8:
                raise RuntimeError(
                    f"Bridge identity failed in replicate {k} for {lab} "
                    f"(error={err:.3g})")

    print(f"\n  H/P-decomposed bridge (cell means; two-way bootstrap):")
    print(f"  {'Contrast':<12} {'Component':<10} {'Point':>10} "
          f"{'95% CI':>24} {'p (2-sided)':>13}")
    print(f"  {'-'*75}")
    bridge_rows = []
    for i, (lab, component) in enumerate(bridge_keys):
        point = float(bridge_matrix[:, i].mean())
        lo, hi = np.quantile(bridge_draws[:, i], [.025, .975])
        p2 = _bootstrap_p2(bridge_draws[:, i])
        print(f"  {lab:<12} {component:<10} {point:>+10.4f} "
              f"[{lo:>+9.4f}, {hi:>+9.4f}]  {p2:>12.4f}")
        bridge_rows.append(dict(
            contrast=lab, component=component, point=point,
            ci_low=float(lo), ci_high=float(hi), p_two_sided=p2,
            n_cells=n_j, n_clusters=n_clu, n_d_draws=n_d_draws,
            bootstrap_type="two-way",
            bootstrap_draws=args.b))
    pd.DataFrame(bridge_rows).to_csv(
        ART / "cost_bridge_hp.csv", index=False)

    dt = bridge_arrays[("Delta", "Total")].mean()
    dh = bridge_arrays[("Delta", "Holding")].mean()
    dp = bridge_arrays[("Delta", "Shortage")].mean()
    print(f"\n  ΔB identity: {dt:+.4f} = {dh:+.4f} + {dp:+.4f}")
    print(f"  Component shares: holding={dh/dt:.1%}, "
          f"shortage={dp/dt:.1%}")

    # Interpretation
    print(f"\n  Interpretation:")
    lo_did = float(np.quantile(draws[:, 6], .025))
    hi_did = float(np.quantile(draws[:, 6], .975))
    a_d_pt = pt[5]
    if hi_did < 0:
        print(f"    DiD < 0: attenuation is smaller under latent D")
        print(f"    → censored-sales evaluation distorts the dynamic")
        print(f"      comparison in this calibrated semi-synthetic design")
    elif lo_did > 0:
        print(f"    DiD > 0: attenuation is larger under latent D")
    else:
        print(f"    DiD CI includes zero: censoring effect not significant")

    lo_ad, hi_ad = np.quantile(draws[:, 5], [.025, .975])
    if lo_ad > 0:
        print(f"    A_D significant: attenuation persists under latent D")
    elif lo_ad <= 0 <= hi_ad:
        print(f"    A_D CI includes zero: no statistically detectable "
              f"attenuation under D (point={a_d_pt:+.2f} pp)")
    else:
        print(f"    A_D is significantly negative")

    chk.n_rows = len(df)
    chk.finish(ART / "checks")
    print(f"\nAll results saved to {ART}")
    print(f"Total time: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
