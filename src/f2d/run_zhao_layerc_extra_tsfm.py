"""Layer C for extra TSFMs (TimesFM-ZS, Moirai-ZS) on Zhao validation + test.

Produces per-series costs and appends bootstrap CIs to the existing CSV.

Usage: PYTHONPATH=src python -m f2d.run_zhao_layerc_extra_tsfm [--device mps]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import torch

from . import config as cfgmod
from .aggregation import convolve_varying_pmf
from .conventions import SEED_BASE
from .datasets import zhao
from .decision import costs_from_alpha, order_up_to
from .models.chronos import NATIVE_LEVELS, QuantileRepair
from .models.gbdt_grid import make_lean_features
from .simulation import ReplayConfig, replay
from .uncertainty import paired_bootstrap_mean
from .run_zhao_layerc_ci import _setup_window

ART = cfgmod.ARTIFACT_DIR / "zhao_layerc"

VMAX = 60
LEAD_DAYS = 1
ALPHA = 0.85
KAPPA_H = 0.20
BASELINE = "emp-daily"
BOOTSTRAP_B = 10_000

VALID_MONTHS = [pd.Timestamp("2019-07-01"), pd.Timestamp("2019-08-01")]
TEST_MONTHS = [pd.Timestamp("2019-09-01"), pd.Timestamp("2019-10-01")]

TIMESFM_LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def _replay_per_series(arm, S_val, demand, initial_inv, cost_arr,
                       replay_days_total, review_days_list, n_reviews, n_cross):
    S_matrix = np.tile(S_val[:, None], (1, n_reviews))
    rc = ReplayConfig(
        n_days=replay_days_total,
        lead_time_days=LEAD_DAYS,
        review_cadence_days=review_days_list[1] if n_reviews > 1 else replay_days_total,
        shortage_mechanism="lost_sales")
    res = replay(demand, S_matrix, initial_inv, rc)

    avg_iend = res.i_end.mean(axis=1)
    total_short = res.lost.sum(axis=1)
    h_c, p_c = costs_from_alpha(cost_arr, ALPHA, KAPPA_H, 12)
    row_cost = (h_c * avg_iend + p_c * total_short) / replay_days_total

    n_so = 0
    n_cyc = 0
    for ci, start in enumerate(review_days_list):
        end = review_days_list[ci + 1] if ci + 1 < n_reviews else replay_days_total
        cyc_lost = res.lost[:, start:end].sum(axis=1)
        n_so += int((cyc_lost > 0).sum())
        n_cyc += n_cross
    csr = 1.0 - n_so / max(n_cyc, 1)
    fr = 1.0 - float(res.lost.sum()) / max(res.demand.sum(), 1e-12)

    has_cost = np.isfinite(cost_arr)
    avg_cpsd = float(np.nanmean(row_cost[has_cost]))
    return row_cost, avg_cpsd, csr, fr


def _forecast_timesfm(sids, ctx, n_days, batch_size):
    """TimesFM 2.5 zero-shot: returns list of n_days grids, each (n_series, 9)."""
    import timesfm
    fc = timesfm.ForecastConfig(
        max_context=512, max_horizon=n_days,
        per_core_batch_size=batch_size, infer_is_positive=True,
        fix_quantile_crossing=True)
    m = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch", local_files_only=True)
    m.compile(fc)

    inputs = [ctx[s].astype(np.float32) for s in sids]
    pts, quants = m.forecast(horizon=n_days, inputs=inputs)
    # quants: (n_series, n_days, 10) — first 9 are q0.1..q0.9, index 5 is median (=point)
    # Actually the quantile dim is 10; let's take indices 0..8 for 0.1..0.9
    q = quants[:, :, :9]  # (n_series, n_days, 9)
    q = np.clip(q, 0.0, None)
    grids = []
    for d in range(n_days):
        g = q[:, d, :]  # (n_series, 9)
        g = np.maximum.accumulate(g, axis=1)  # ensure monotonicity
        grids.append(g)
    return grids, TIMESFM_LEVELS


def _forecast_moirai(sids, ctx, n_days, batch_size, device="cpu"):
    """Moirai 1.1-R-base zero-shot: sample-based → quantiles at NATIVE_LEVELS."""
    from einops import rearrange
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    module = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-base")

    n_series = len(sids)
    n_samples = 200

    # Build dataset
    from gluonts.dataset.common import ListDataset
    ds = ListDataset(
        [{"start": pd.Timestamp("2019-01-01"),  # dummy start
          "target": ctx[s].astype(np.float32)}
         for s in sids],
        freq="D")

    # Moirai uses float64 internally; MPS doesn't support it → force CPU
    predictor = MoiraiForecast(
        prediction_length=n_days,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
        context_length=512,
        module=module,
        patch_size="auto",
        num_samples=n_samples,
    ).create_predictor(batch_size=batch_size, device=torch.device("cpu"))

    forecasts = list(predictor.predict(ds))
    # Each forecast has .samples of shape (n_samples, n_days)
    all_quantiles = np.zeros((n_series, n_days, len(NATIVE_LEVELS)))
    for i, fc in enumerate(forecasts):
        samples = np.clip(fc.samples, 0.0, None)  # (n_samples, n_days)
        for d in range(n_days):
            all_quantiles[i, d, :] = np.quantile(
                samples[:, d], NATIVE_LEVELS, method="inverted_cdf")

    grids = []
    for d in range(n_days):
        g = all_quantiles[:, d, :]
        g = np.maximum.accumulate(g, axis=1)
        grids.append(g)
    return grids, NATIVE_LEVELS


def run_arm(arm_name, grids, q_levels, w, t0):
    """Run one TSFM arm through the Layer C pipeline."""
    review_cadence = w["review_days_list"][1] if w["n_reviews"] > 1 else w["replay_days_total"]
    pi_days = review_cadence + LEAD_DAYS

    pi_grids = grids[:pi_days] if pi_days <= len(grids) else grids
    if len(pi_grids) < pi_days:
        pi_grids = pi_grids + [pi_grids[-1]] * (pi_days - len(pi_grids))

    pmf_pi = convolve_varying_pmf(q_levels, pi_grids, vmax=VMAX)
    pmf_r = convolve_varying_pmf(q_levels, grids[:review_cadence], vmax=VMAX)
    m_ratio = pi_days / review_cadence
    S_dict = order_up_to(pmf_r, pmf_pi, ALPHA, m_ratio)
    S_val = S_dict["P1"]

    row_cost, avg_cpsd, csr, fr = _replay_per_series(
        arm_name, S_val, w["demand"], w["initial_inv"], w["cost_arr"],
        w["replay_days_total"], w["review_days_list"], w["n_reviews"], w["n_cross"])

    print(f"  {arm_name:<24} cost={avg_cpsd:.4f} ({time.time()-t0:.0f}s)")
    return row_cost, avg_cpsd, csr, fr


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(argv)
    t0 = time.time()

    raw = zhao.load_raw()
    daily, _ = zhao.build_daily_panel(raw)
    panel, _ = zhao.build_panel(raw)

    rng = np.random.default_rng(SEED_BASE)
    pool = np.asarray(sorted(set(daily.series_id)))
    keep = rng.choice(pool, size=min(2000, len(pool)), replace=False)
    daily = daily[daily.series_id.isin(keep)]

    windows = [
        ("validation", VALID_MONTHS),
        ("test", TEST_MONTHS),
    ]

    all_rows = []
    for window_name, months in windows:
        print(f"\n{'='*60}")
        print(f"  Window: {window_name} ({months[0].strftime('%Y-%m')} ~ {months[-1].strftime('%Y-%m')})")
        print(f"{'='*60}")

        (sids_cross, sk_cross, initial_inv, cost_arr, demand, ctx,
         replay_days_total, review_days_list, n_reviews, n_cross
         ) = _setup_window(daily, panel, months)
        w = dict(sids_cross=sids_cross, sk_cross=sk_cross, n_cross=n_cross,
                 initial_inv=initial_inv, cost_arr=cost_arr, demand=demand, ctx=ctx,
                 replay_days_total=replay_days_total,
                 review_days_list=review_days_list, n_reviews=n_reviews)
        sids = sids_cross
        n_days = replay_days_total
        print(f"  序列数: {n_cross}, 回放天数: {n_days}")

        # --- TimesFM ---
        print(f"\n--- TimesFM-ZS ({time.time()-t0:.0f}s) ---")
        tfm_grids, tfm_levels = _forecast_timesfm(sids, ctx, n_days, args.batch_size)
        tfm_cost, tfm_avg, tfm_csr, tfm_fr = run_arm(
            "timesfm-zs", tfm_grids, tfm_levels, w, t0)

        # --- Moirai ---
        print(f"\n--- Moirai-ZS ({time.time()-t0:.0f}s) ---")
        mor_grids, mor_levels = _forecast_moirai(
            sids, ctx, n_days, args.batch_size, device=args.device)
        mor_cost, mor_avg, mor_csr, mor_fr = run_arm(
            "moirai-zs", mor_grids, mor_levels, w, t0)

        # --- Bootstrap vs emp-daily ---
        # Load emp-daily per-series from existing run
        # We need to re-compute emp-daily per-series here
        from .models.chronos import QuantileRepair
        emp = np.array([np.quantile(ctx[s], NATIVE_LEVELS, method="inverted_cdf")
                        for s in sids], float)
        emp, _ = QuantileRepair()(emp)
        emp_grids = [emp] * n_days
        emp_cost, emp_avg, emp_csr, emp_fr = run_arm(
            "emp-daily", emp_grids, NATIVE_LEVELS, w, t0)

        # Bootstrap
        print(f"\n--- Bootstrap (B={BOOTSTRAP_B}) ---")
        per_series = {
            "emp-daily": emp_cost,
            "timesfm-zs": tfm_cost,
            "moirai-zs": mor_cost,
        }

        rows = []
        for arm, costs in per_series.items():
            for i, s in enumerate(sids):
                rows.append({"variant": arm, "series_id": s, "cost": costs[i], "month": 0})
        df = pd.DataFrame(rows)
        nan_series = df[df["cost"].isna()]["series_id"].unique()
        if len(nan_series):
            df = df[~df["series_id"].isin(nan_series)]

        variants = ["timesfm-zs", "moirai-zs"]
        cis = paired_bootstrap_mean(
            df, value_col="cost", baseline="emp-daily", variants=variants,
            variant_col="variant", series_col="series_id",
            b=BOOTSTRAP_B, ci=0.95, seed=SEED_BASE)

        for ci in cis:
            sig = "***" if ci.significant else "n.s."
            print(f"  {ci.variant:<24} Δ={ci.delta:+.4f}  "
                  f"[{ci.lo:+.4f}, {ci.hi:+.4f}]  {sig}")

        arm_data = {
            "timesfm-zs": (tfm_avg, tfm_csr, tfm_fr),
            "moirai-zs": (mor_avg, mor_csr, mor_fr),
        }
        for ci in cis:
            avg_cpsd, csr, fr = arm_data[ci.variant]
            all_rows.append({
                "split": window_name,
                "arm": ci.variant,
                "cost": round(avg_cpsd, 4),
                "CSR": round(csr, 4),
                "FR": round(fr, 4),
                "delta_cost": round(ci.delta, 6),
                "ci_lo": round(ci.lo, 6),
                "ci_hi": round(ci.hi, 6),
                "significant": ci.significant,
                "n_series": ci.n_series,
                "B": ci.b,
            })

    out_df = pd.DataFrame(all_rows)
    out_path = ART / "layer_c_extra_tsfm.csv"
    ART.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\n结果已保存: {out_path}")
    print(f"\n总耗时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
