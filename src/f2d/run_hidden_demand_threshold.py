"""Hidden-demand thresholds for the Zhao fixed-policy comparison.

Layer 1 computes the sharp lower bound on decision-relevant hidden overlap
units implied by the static censoring identity.  Layer 2 scales the existing
semi-synthetic excess demand while holding both policies fixed, then replays
the inventory system over multiple replenishment cycles.
"""

from __future__ import annotations

import argparse
import hashlib
import time

import numpy as np
import pandas as pd

from . import config as cfgmod
from .conventions import SEED_BASE
from .datasets import zhao
from .decision import costs_from_alpha
from .run_halfsynthetic import (HIGH_ALPHAS, KAPPA_H, LEAD_DAYS, ORIGINS, R,
                                SCORED_ORIGINS,
                                _construct_latent_demand)
from .simulation import ReplayConfig, replay

ART = cfgmod.ARTIFACT_DIR / "zhao_hidden_demand_threshold"
HALF_ART = cfgmod.ARTIFACT_DIR / "zhao_halfsynthetic"
RHOS = (0.0, 0.10, 0.25, 0.30, 0.35, 0.50, 0.75, 1.0, 1.25, 1.50)


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npz_content_sha256(saved) -> str:
    """Hash array contents rather than ZIP metadata."""
    digest = hashlib.sha256()
    # Policy targets do not depend on latent-demand draw count or seed.
    for key in sorted(set(saved.files).difference({"n_draws", "seed"})):
        value = np.ascontiguousarray(saved[key])
        digest.update(key.encode())
        digest.update(value.dtype.str.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _minimum_overlap_units(weights: np.ndarray, capacities: np.ndarray,
                           target_cost: float) -> tuple[float, int]:
    """Continuous-knapsack lower bound for weighted hidden overlap units."""
    keep = (weights > 0) & (capacities > 0)
    order = np.flatnonzero(keep)[np.argsort(weights[keep])[::-1]]
    remaining = float(target_cost)
    units = 0.0
    used = 0
    for j in order:
        take = min(float(capacities[j]), remaining / float(weights[j]))
        units += take
        remaining -= take * float(weights[j])
        used += take > 0
        if remaining <= 1e-10:
            return units, used
    raise RuntimeError(
        f"Target wedge {target_cost:.6g} exceeds positive capacity "
        f"{np.sum(weights[keep] * capacities[keep]):.6g}")


def _positive_overlap_segments(y: np.ndarray, low: np.ndarray,
                               high: np.ndarray, sign: np.ndarray,
                               weights: np.ndarray
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Positive marginal wedge segments, coupling alphas within each cell."""
    slopes, capacities = [], []
    for i, o in np.ndindex(y.shape):
        cuts = {float(y[i, o])}
        for a in range(low.shape[2]):
            lo = max(float(y[i, o]), float(low[i, o, a]))
            hi = float(high[i, o, a])
            if hi > lo:
                cuts.update((lo, hi))
        cuts = sorted(cuts)
        for left, right in zip(cuts[:-1], cuts[1:]):
            mid = (left + right) / 2
            slope = sum(
                sign[i, o, a] * weights[i, o, a]
                for a in range(low.shape[2])
                if low[i, o, a] < mid < high[i, o, a])
            if slope > 0 and right > left:
                slopes.append(slope)
                capacities.append(right - left)
    return np.asarray(slopes), np.asarray(capacities)


def _crossing(x: np.ndarray, y: np.ndarray, threshold: float = 0.0) -> float:
    """First downward linear crossing; NaN if the grid never crosses."""
    if y[0] <= threshold:
        return float(x[0])
    for j in range(1, len(x)):
        if y[j] <= threshold < y[j - 1]:
            return float(x[j - 1] + (threshold - y[j - 1])
                         * (x[j] - x[j - 1]) / (y[j] - y[j - 1]))
    return float("nan")


def _ratio(d: np.ndarray, b: np.ndarray) -> float:
    return float(d.sum() / b.sum() * 100)


def _two_way_bootstrap_ratios(ds: np.ndarray, bs: np.ndarray,
                              dd: np.ndarray, bd: np.ndarray,
                              n_boot: int, seed: int
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jointly resample series and latent-demand draw IDs."""
    if not (ds.shape == bs.shape == dd.shape == bd.shape):
        raise ValueError("All bootstrap inputs must have the same shape")
    n_rho, n_draws, n_series = ds.shape
    rng = np.random.default_rng(seed)
    series_weights = rng.multinomial(
        n_series, np.full(n_series, 1 / n_series), size=n_boot).astype(float)
    draw_weights = rng.multinomial(
        n_draws, np.full(n_draws, 1 / n_draws), size=n_boot) / n_draws
    packed = np.concatenate((ds, bs, dd, bd), axis=1)
    aggregated = (series_weights @ packed.reshape(
        n_rho * 4 * n_draws, n_series).T).reshape(
            n_boot, n_rho, 4, n_draws)
    weighted = (aggregated * draw_weights[:, None, None, :]).sum(axis=-1)
    if np.any(weighted[:, :, (1, 3)] <= 0):
        raise RuntimeError("Bootstrap baseline cost must be positive")
    static = weighted[:, :, 0] / weighted[:, :, 1] * 100
    dynamic = weighted[:, :, 2] / weighted[:, :, 3] * 100
    return static, dynamic, dynamic - static


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-draws", type=int, default=50)
    ap.add_argument("--b", type=int, default=10_000,
                    help="Two-way bootstrap replicates")
    args = ap.parse_args(argv)
    if args.n_draws < 1 or args.b < 1:
        ap.error("--n-draws and --b must be positive")
    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)

    policy_file = HALF_ART / "fixed_policy_inputs.npz"
    long_file = HALF_ART / "halfsynthetic_long.csv"
    missing = [path.name for path in (policy_file, long_file)
               if not path.exists()]
    if missing:
        raise RuntimeError(
            "Run the full f2d.run_halfsynthetic experiment first; missing: "
            + ", ".join(missing))

    eligibility = pd.read_csv(HALF_ART / "analysis_eligibility.csv")
    sids = eligibility.loc[eligibility.eligible, "series_id"].astype(str).to_numpy()
    n_ser = len(sids)
    raw = zhao.load_raw()
    daily, _ = zhao.build_daily_panel(raw)
    panel, _ = zhao.build_panel(raw)
    daily["series_id"] = daily.series_id.astype(str)
    daily = daily[daily.series_id.isin(sids)]
    sku_of = dict(zip(daily.series_id, daily.sku_ID))

    snap = panel[panel.month == ORIGINS[0]].set_index("sku_ID")
    sk = np.array([sku_of[s] for s in sids])
    inv0 = snap.loc[sk, "beginning_inventory"].to_numpy(float)
    cost_i = snap.loc[sk, "unit_cost_hist"].to_numpy(float)
    if not np.isfinite(cost_i).all():
        raise RuntimeError("Saved eligibility and finite unit costs disagree")

    start = ORIGINS[0]
    end = ORIGINS[-1] + pd.DateOffset(months=1)
    total_days = (end - start).days + LEAD_DAYS
    dm_y, dm_d, cstats = _construct_latent_demand(
        daily, panel, sids, sku_of, start, total_days,
        args.n_draws, SEED_BASE)

    sel = pd.read_csv(cfgmod.ARTIFACT_DIR / "zhao_rolling_origins"
                      / "baseline_selection.csv")
    base_of = {(float(r.alpha), int(r.origin_idx)): r.retuned
               for _, r in sel.iterrows()}
    arms = {"chronos2-zs"}
    arms.update(base_of[(a, oi)] for a in HIGH_ALPHAS for oi in SCORED_ORIGINS)

    saved = np.load(policy_file, allow_pickle=False)
    metadata_keys = {"sids", "inv0", "cost_i", "start_date", "total_days"}
    missing_keys = sorted(metadata_keys.difference(saved.files))
    if missing_keys:
        raise RuntimeError(
            "Regenerate fixed policy inputs; missing metadata: "
            + ", ".join(missing_keys))
    if not np.array_equal(saved["sids"].astype(str), sids):
        raise RuntimeError("Fixed policy inputs use a different eligible sample")
    if ("n_draws" in saved.files
            and int(saved["n_draws"].item()) != args.n_draws):
        raise RuntimeError("Fixed policy inputs use a different draw count")
    if "seed" in saved.files and int(saved["seed"].item()) != SEED_BASE:
        raise RuntimeError("Fixed policy inputs use a different latent-demand seed")
    metadata_ok = (
        np.array_equal(saved["inv0"], inv0)
        and np.array_equal(saved["cost_i"], cost_i)
        and saved["start_date"].item() == str(start.date())
        and int(saved["total_days"].item()) == total_days)
    if not metadata_ok:
        raise RuntimeError("Fixed policy input metadata does not match this run")
    policy_hash = _npz_content_sha256(saved)
    long_hash = _file_sha256(long_file)
    precomp = {}
    for arm in sorted(arms):
        for alpha in HIGH_ALPHAS:
            prefix = f"{arm.replace('-', '_')}_a{int(round(alpha * 100))}"
            static = {oi: saved[f"{prefix}_static_o{oi}"]
                      for oi in SCORED_ORIGINS}
            precomp[(arm, alpha)] = (
                tuple(saved[f"{prefix}_review_days"].astype(int)),
                saved[f"{prefix}_S_arr"], static)
    saved.close()

    n_cells = n_ser * len(SCORED_ORIGINS) * len(HIGH_ALPHAS)
    cell_shape = (n_ser, len(SCORED_ORIGINS), len(HIGH_ALPHAS))
    target_z = np.empty(cell_shape)
    target_e = np.empty(cell_shape)
    weights = np.empty(cell_shape)
    y_pi = np.empty((n_ser, len(SCORED_ORIGINS)))
    for o, oi in enumerate(SCORED_ORIGINS):
        month = ORIGINS[oi]
        mo = (month - start).days
        y_pi[:, o] = dm_y[:, mo:mo + month.days_in_month + LEAD_DAYS].sum(1)
        for a, alpha in enumerate(HIGH_ALPHAS):
            ba = base_of[(alpha, oi)]
            target_z[:, o, a] = precomp[("chronos2-zs", alpha)][2][oi]
            target_e[:, o, a] = precomp[(ba, alpha)][2][oi]
            h, p = costs_from_alpha(cost_i, alpha, KAPPA_H, 12)
            weights[:, o, a] = h + p

    y_cell = np.repeat(y_pi[:, :, None], len(HIGH_ALPHAS), axis=2)
    low, high = np.minimum(target_z, target_e), np.maximum(target_z, target_e)
    sign = np.sign(target_z - target_e)
    segment_weights, segment_capacities = _positive_overlap_segments(
        y_pi, low, high, sign, weights)

    nr, nd = len(RHOS), len(dm_d)
    ds = np.empty((nr, nd, n_ser))
    bs = np.empty_like(ds)
    dd = np.empty_like(ds)
    bd = np.empty_like(ds)
    hidden_rows = []
    rc = ReplayConfig(n_days=total_days, lead_time_days=LEAD_DAYS,
                      review_cadence_days=R,
                      shortage_mechanism="lost_sales",
                      review_days=precomp[("chronos2-zs", HIGH_ALPHAS[0])][0])

    def evaluate(dm):
        stat_z = np.empty(cell_shape)
        stat_e = np.empty(cell_shape)
        dyn = {}
        for a, alpha in enumerate(HIGH_ALPHAS):
            h, p = costs_from_alpha(cost_i, alpha, KAPPA_H, 12)
            for arm in arms:
                _, S, _ = precomp[(arm, alpha)]
                out = replay(dm, S, inv0, rc)
                for o, oi in enumerate(SCORED_ORIGINS):
                    month = ORIGINS[oi]
                    mo = (month - start).days
                    me = mo + month.days_in_month
                    dyn[(arm, oi, alpha)] = (
                        h * out.i_end[:, mo:me].mean(1)
                        + p * out.lost[:, mo:me].sum(1))
            for o, oi in enumerate(SCORED_ORIGINS):
                month = ORIGINS[oi]
                mo = (month - start).days
                dpi = dm[:, mo:mo + month.days_in_month + LEAD_DAYS].sum(1)
                ba = base_of[(alpha, oi)]
                for dest, arm in [(stat_z, "chronos2-zs"), (stat_e, ba)]:
                    S = precomp[(arm, alpha)][2][oi]
                    dest[:, o, a] = (h * np.maximum(S - dpi, 0)
                                     + p * np.maximum(dpi - S, 0))
        dyn_z = np.empty(cell_shape)
        dyn_e = np.empty(cell_shape)
        for o, oi in enumerate(SCORED_ORIGINS):
            for a, alpha in enumerate(HIGH_ALPHAS):
                dyn_z[:, o, a] = dyn[("chronos2-zs", oi, alpha)]
                dyn_e[:, o, a] = dyn[(base_of[(alpha, oi)], oi, alpha)]
        return stat_z, stat_e, dyn_z, dyn_e

    for r, rho in enumerate(RHOS):
        draw_ids = [0] if rho == 0 else range(nd)
        for k in draw_ids:
            dm = dm_y + rho * (dm_d[k] - dm_y)
            sz, se, dz, de = evaluate(dm)
            ds[r, k] = (sz - se).sum(axis=(1, 2))
            bs[r, k] = se.sum(axis=(1, 2))
            dd[r, k] = (dz - de).sum(axis=(1, 2))
            bd[r, k] = de.sum(axis=(1, 2))
        if rho == 0:
            for arr in (ds, bs, dd, bd):
                arr[r] = arr[r, 0]
        excess = rho * (dm_d - dm_y).mean(axis=0)
        scored_excess = []
        scored_logged = []
        for oi in SCORED_ORIGINS:
            month = ORIGINS[oi]
            mo = (month - start).days
            sl = slice(mo, mo + month.days_in_month + LEAD_DAYS)
            scored_excess.append(excess[:, sl].sum())
            scored_logged.append(dm_y[:, sl].sum())
        hidden_rows.append(dict(
            rho=rho,
            mean_excess_per_scored_sku_month=sum(scored_excess)
            / (n_ser * len(SCORED_ORIGINS)),
            excess_pct_logged_protection_demand=
            sum(scored_excess) / sum(scored_logged) * 100,
            mean_excess_per_designated_inflated_day=
            excess.sum() / max(cstats["n_inflated_day_series"], 1)))
        print(f"  rho={rho:.2f} replayed ({time.time() - t0:.0f}s)")

    # Cached targets must reproduce both saved Y cells and mean D cells.
    old_all = pd.read_csv(long_file)
    old_all["series_id"] = old_all.series_id.astype(str)
    for demand_label, rho_idx in (("Y", 0), ("D", RHOS.index(1.0))):
        old = old_all[old_all.demand == demand_label].set_index(
            ["series_id", "origin_idx", "alpha", "arm"])
        expected_arrays = [[], [], [], []]
        for sid in sids:
            for oi in SCORED_ORIGINS:
                for alpha in HIGH_ALPHAS:
                    z = old.loc[(sid, oi, alpha, "chronos2-zs")]
                    e = old.loc[(sid, oi, alpha, base_of[(alpha, oi)])]
                    values = (z.L0_static - e.L0_static, e.L0_static,
                              z.L3_longrun - e.L3_longrun, e.L3_longrun)
                    for dest, value in zip(expected_arrays, values):
                        dest.append(value)
        rebuilt = [x[rho_idx].mean(axis=0) for x in (ds, bs, dd, bd)]
        for name, got, expected in zip(
                ("static difference", "static baseline", "dynamic difference",
                 "dynamic baseline"), rebuilt, expected_arrays):
            expected = np.asarray(expected).reshape(n_ser, -1).sum(1)
            err = float(np.max(np.abs(got - expected)))
            if err > 1e-8:
                raise RuntimeError(
                    f"rho={RHOS[rho_idx]:g} {name} audit failed: "
                    f"max error={err:.3g}")

    bootstrap_seed = SEED_BASE + 7777
    s_boot_all, d_boot_all, att_boot_all = _two_way_bootstrap_ratios(
        ds, bs, dd, bd, args.b, bootstrap_seed)
    curve_rows = []
    for r, rho in enumerate(RHOS):
        dsm, bsm = ds[r].mean(0), bs[r].mean(0)
        ddm, bdm = dd[r].mean(0), bd[r].mean(0)
        s_pt, d_pt = _ratio(dsm, bsm), _ratio(ddm, bdm)
        s_boot = s_boot_all[:, r]
        d_boot = d_boot_all[:, r]
        att_boot = att_boot_all[:, r]
        curve_rows.append(dict(
            **hidden_rows[r], static_effect_pct=s_pt,
            static_ci_low=np.quantile(s_boot, .025),
            static_ci_high=np.quantile(s_boot, .975),
            closed_loop_effect_pct=d_pt,
            closed_loop_ci_low=np.quantile(d_boot, .025),
            closed_loop_ci_high=np.quantile(d_boot, .975),
            attenuation_pp=d_pt - s_pt,
            attenuation_ci_low=np.quantile(att_boot, .025),
            attenuation_ci_high=np.quantile(att_boot, .975),
            absolute_static_difference=dsm.sum() / n_cells,
            absolute_closed_loop_difference=ddm.sum() / n_cells,
            n_series=n_ser, n_draws=nd, bootstrap_draws=args.b,
            bootstrap_type="two-way_series_and_latent_draw"))
    curve = pd.DataFrame(curve_rows)
    curve["did_vs_logged_pp"] = (curve.attenuation_pp
                                  - curve.attenuation_pp.iloc[0])
    curve.to_csv(ART / "closed_loop_scaling.csv", index=False)
    pd.DataFrame([dict(
        seed=SEED_BASE, bootstrap_seed=bootstrap_seed,
        n_series=n_ser, n_draws=nd, bootstrap_draws=args.b,
        bootstrap_type="two-way_series_and_latent_draw",
        start_date=str(start.date()), total_days=total_days,
        rho_grid=",".join(map(str, RHOS)),
        policy_input_content_sha256=policy_hash,
        halfsynthetic_long_sha256=long_hash,
    )]).to_csv(ART / "run_metadata.csv", index=False)

    static_bridge = ((ds[0].mean(0) - ds[RHOS.index(1.0)].mean(0)).sum()
                     / n_cells)
    target_total = static_bridge * n_cells
    min_units, used = _minimum_overlap_units(
        segment_weights, segment_capacities, target_total)
    d_pi_draw = np.empty((nd, n_ser, len(SCORED_ORIGINS)))
    for o, oi in enumerate(SCORED_ORIGINS):
        month = ORIGINS[oi]
        mo = (month - start).days
        d_pi_draw[:, :, o] = dm_d[:, :, mo:mo + month.days_in_month
                                  + LEAD_DAYS].sum(2)
    d_cell_draw = np.repeat(
        d_pi_draw[:, :, :, None], len(HIGH_ALPHAS), axis=3)
    overlap = np.maximum(
        np.minimum(d_cell_draw, high[None])
        - np.maximum(y_cell, low)[None], 0).mean(0)
    identity_bridge = np.mean(sign * weights * overlap)
    if not np.isclose(identity_bridge, static_bridge, atol=1e-8):
        raise RuntimeError(
            f"Static overlap identity failed: {identity_bridge} != {static_bridge}")

    static_row = pd.DataFrame([dict(
        target_absolute_static_wedge_per_cell=static_bridge,
        identity_absolute_static_wedge_per_cell=identity_bridge,
        minimum_decision_relevant_hidden_units=min_units,
        minimum_hidden_units_per_sku_origin=
        min_units / (n_ser * len(SCORED_ORIGINS)),
        minimum_overlap_pct_logged_protection_demand=
        min_units / y_pi.sum() * 100,
        positive_marginal_segments_used=used,
        positive_marginal_segments_available=len(segment_capacities),
        calibrated_total_excess_units=(d_pi_draw - y_pi).mean(0).sum(),
        calibrated_positive_overlap_units=np.where(sign > 0, overlap, 0).sum(),
        calibrated_signed_overlap_units=(sign * overlap).sum(),
        n_policy_cells=n_cells,
        n_sku_origin_cells=n_ser * len(SCORED_ORIGINS),
        interpretation=(
            "Continuous lower bound on physical hidden units inside target "
            "intervals after coupling service levels; "
            "not a lower bound on all latent demand when logged sales lie "
            "below both policy targets."))])
    static_row.to_csv(ART / "static_threshold.csv", index=False)

    rho = curve.rho.to_numpy()
    calibrated_did = curve.did_vs_logged_pp.iloc[RHOS.index(1.0)]
    threshold_rows = [
        ("point closed-loop advantage", _crossing(
            rho, curve.closed_loop_effect_pct.to_numpy())),
        ("95% upper CI below zero", _crossing(
            rho, curve.closed_loop_ci_high.to_numpy())),
        ("reproduce calibrated -8.28 pp DiD", _crossing(
            rho, curve.did_vs_logged_pp.to_numpy(), calibrated_did)),
    ]
    summary = pd.DataFrame([
        dict(
            threshold=label, rho=rho_star,
            mean_excess_per_scored_sku_month=np.interp(
                rho_star, rho, curve.mean_excess_per_scored_sku_month),
            excess_pct_logged_protection_demand=np.interp(
                rho_star, rho, curve.excess_pct_logged_protection_demand),
            mean_excess_per_designated_inflated_day=np.interp(
                rho_star, rho,
                curve.mean_excess_per_designated_inflated_day))
        for label, rho_star in threshold_rows])
    summary.to_csv(ART / "break_even_summary.csv", index=False)
    print(curve.to_string(index=False))
    print(static_row.to_string(index=False))
    print(summary.to_string(index=False))
    print(f"Saved results to {ART} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
