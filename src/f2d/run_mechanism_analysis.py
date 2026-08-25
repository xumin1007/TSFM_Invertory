"""Mechanism analysis: TSFM advantage vs non-stationarity.

Per-SKU non-stationarity metrics vs per-SKU cost advantage of TSFM over emp-daily.
Tests the hypothesis: TSFM's edge grows with demand non-stationarity.

Metrics:
  1. drift_ratio: mean(validation sales) / mean(history sales)  — level shift
  2. cv_rolling: CV of 30-day rolling means — local trend volatility
  3. zero_share: fraction of zero-demand days — intermittency

用法:  PYTHONPATH=src python -m f2d.run_mechanism_analysis [--device mps]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy import stats as sp_stats

from . import config as cfgmod
from .aggregation import convolve_varying_pmf
from .checks import CheckResult
from .conventions import SEED_BASE
from .datasets import zhao
from .decision import costs_from_alpha, layer_b, order_up_to
from .models.chronos import BASE_CHECKPOINT, NATIVE_LEVELS, QuantileRepair, to_grid

ART = cfgmod.ARTIFACT_DIR / "zhao_mechanism"
VMAX_DEC = 60
ALPHA_PRIMARY = 0.95
KAPPA_H = 0.20
LEAD_DAYS = 1
MIN_CONTEXT = 30

VALID_MONTHS = [pd.Timestamp("2019-07-01"), pd.Timestamp("2019-08-01")]


def _nonstationarity_metrics(hist: np.ndarray, val_sales: float,
                              n_days_val: int) -> dict:
    """Compute per-series non-stationarity metrics."""
    T = len(hist)

    # 1. drift ratio: validation daily rate / history daily rate
    hist_daily_mean = hist.mean() if T > 0 else 1e-8
    val_daily_mean = val_sales / max(n_days_val, 1)
    drift = val_daily_mean / max(hist_daily_mean, 1e-8)

    # 2. CV of 30-day rolling means — trend volatility
    window = min(30, T)
    if T >= window:
        rolling_means = pd.Series(hist).rolling(window).mean().dropna().to_numpy()
        cv_rolling = float(rolling_means.std() / max(rolling_means.mean(), 1e-8))
    else:
        cv_rolling = 0.0

    # 3. zero share — intermittency
    zero_share = float((hist == 0).mean()) if T > 0 else 0.0

    # 4. ADF test p-value (stationarity test, higher p = more non-stationary)
    if T >= 20:
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_p = adfuller(hist, maxlag=min(7, T // 5), autolag=None)[1]
        except Exception:
            adf_p = np.nan
    else:
        adf_p = np.nan

    return dict(drift_ratio=drift, cv_rolling=cv_rolling,
                zero_share=zero_share, adf_p=adf_p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=2000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args(argv)

    from chronos import BaseChronosPipeline

    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    chk = CheckResult(step_id="mechanism", dataset="zhao", seed=SEED_BASE)

    raw = zhao.load_raw()
    daily, _ = zhao.build_daily_panel(raw)
    panel, _ = zhao.build_panel(raw)

    rng = np.random.default_rng(SEED_BASE)
    all_sids = sorted(daily.series_id.unique())
    keep = rng.choice(all_sids, size=min(args.n_series, len(all_sids)), replace=False)
    daily = daily[daily.series_id.isin(keep)]
    sku_of = dict(zip(daily.series_id, daily.sku_ID))

    pipe = BaseChronosPipeline.from_pretrained(
        BASE_CHECKPOINT, device_map=args.device, torch_dtype=torch.float32)

    rows = []

    for month in VALID_MONTHS:
        n_days = month.days_in_month + LEAD_DAYS
        snap = panel[panel.month == month].set_index("sku_ID")
        hist_df = daily[daily.d < month]
        ctx = {s: g.sort_values("d").y.to_numpy(float)
               for s, g in hist_df.groupby("series_id")}
        sids = np.array(sorted([s for s in ctx
                                if len(ctx[s]) >= MIN_CONTEXT
                                and sku_of.get(s) in snap.index]))
        sk = np.array([sku_of[s] for s in sids])
        cur = snap.loc[sk]
        ip = cur["beginning_inventory"].to_numpy(float) + cur["on_order_inventory"].to_numpy(float)
        y = cur["observed_sales_next_month"].to_numpy(float)
        cost_i = cur["unit_cost_hist"].to_numpy(float)
        has_c = np.isfinite(cost_i)
        h_c, p_c = costs_from_alpha(cost_i, ALPHA_PRIMARY, KAPPA_H, 12)

        # --- TSFM ---
        q, _ = pipe.predict_quantiles(
            [torch.tensor(ctx[s], dtype=torch.float32) for s in sids],
            prediction_length=n_days, quantile_levels=list(NATIVE_LEVELS),
            batch_size=args.batch_size)
        g, _ = QuantileRepair()(to_grid(q))
        g = g.reshape(len(sids), n_days, -1)
        grids = [g[:, i, :] for i in range(n_days)]
        pmf_pi = convolve_varying_pmf(NATIVE_LEVELS, grids, vmax=VMAX_DEC)
        pmf_r = convolve_varying_pmf(NATIVE_LEVELS,
                                      grids[:month.days_in_month], vmax=VMAX_DEC)
        m_ratio = n_days / month.days_in_month
        S_tsfm = order_up_to(pmf_r, pmf_pi, ALPHA_PRIMARY, m_ratio)["P3"]
        r_tsfm = layer_b(S_tsfm, ip, y, h_c, p_c)

        # --- emp-daily ---
        emp = np.array([np.quantile(ctx[s], NATIVE_LEVELS, method="inverted_cdf")
                        for s in sids], float)
        emp, _ = QuantileRepair()(emp)
        emp_grids = [emp] * n_days
        pmf_pi_e = convolve_varying_pmf(NATIVE_LEVELS, emp_grids, vmax=VMAX_DEC)
        pmf_r_e = convolve_varying_pmf(NATIVE_LEVELS,
                                        emp_grids[:month.days_in_month], vmax=VMAX_DEC)
        S_emp = order_up_to(pmf_r_e, pmf_pi_e, ALPHA_PRIMARY, m_ratio)["P3"]
        r_emp = layer_b(S_emp, ip, y, h_c, p_c)

        # Per-series cost
        pos_t = np.maximum(ip, S_tsfm)
        cost_tsfm = h_c * np.clip(pos_t - y, 0, None) + p_c * np.clip(y - pos_t, 0, None)
        pos_e = np.maximum(ip, S_emp)
        cost_emp = h_c * np.clip(pos_e - y, 0, None) + p_c * np.clip(y - pos_e, 0, None)

        for i, s in enumerate(sids):
            if not has_c[i]:
                continue
            metrics = _nonstationarity_metrics(ctx[s], y[i], month.days_in_month)
            rows.append(dict(
                series_id=s, month=month,
                cost_tsfm=float(cost_tsfm[i]),
                cost_emp=float(cost_emp[i]),
                cost_delta=float(cost_tsfm[i] - cost_emp[i]),
                S_tsfm=float(S_tsfm[i]),
                S_emp=float(S_emp[i]),
                y=float(y[i]),
                context_len=len(ctx[s]),
                **metrics))

        print(f"  {month:%Y-%m}: {len(sids)} series processed ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(ART / "mechanism_per_sku.csv", index=False)
    print(f"\n{len(df)} per-SKU-month observations")

    # ================================================================
    # Analysis 1: Binned drift ratio vs cost advantage
    # ================================================================
    print("\n" + "=" * 70)
    print("Analysis 1: Cost advantage by drift ratio quintile")
    print("=" * 70)

    df["drift_bin"] = pd.qcut(df.drift_ratio.clip(0, 3), q=5, duplicates="drop")
    drift_agg = df.groupby("drift_bin", observed=True).agg(
        n=("cost_delta", "size"),
        mean_drift=("drift_ratio", "mean"),
        cost_tsfm=("cost_tsfm", "mean"),
        cost_emp=("cost_emp", "mean"),
        delta=("cost_delta", "mean"),
        delta_pct=("cost_delta", lambda x: x.mean() / max(df.cost_emp.mean(), 1e-8) * 100),
    ).round(3)
    print(drift_agg.to_string())

    # Rank correlation
    rho_drift, p_drift = sp_stats.spearmanr(df.drift_ratio.clip(0, 3), df.cost_delta)
    print(f"\nSpearman ρ(drift_ratio, cost_delta) = {rho_drift:.4f}, p = {p_drift:.4e}")
    chk.note("rho_drift", round(rho_drift, 4))
    chk.note("p_drift", float(f"{p_drift:.4e}"))

    # ================================================================
    # Analysis 2: Binned CV of rolling means vs cost advantage
    # ================================================================
    print("\n" + "=" * 70)
    print("Analysis 2: Cost advantage by trend volatility (cv_rolling) quintile")
    print("=" * 70)

    df["cv_bin"] = pd.qcut(df.cv_rolling.clip(0, 5), q=5, duplicates="drop")
    cv_agg = df.groupby("cv_bin", observed=True).agg(
        n=("cost_delta", "size"),
        mean_cv=("cv_rolling", "mean"),
        cost_tsfm=("cost_tsfm", "mean"),
        cost_emp=("cost_emp", "mean"),
        delta=("cost_delta", "mean"),
    ).round(3)
    print(cv_agg.to_string())

    rho_cv, p_cv = sp_stats.spearmanr(df.cv_rolling.clip(0, 5), df.cost_delta)
    print(f"\nSpearman ρ(cv_rolling, cost_delta) = {rho_cv:.4f}, p = {p_cv:.4e}")
    chk.note("rho_cv", round(rho_cv, 4))
    chk.note("p_cv", float(f"{p_cv:.4e}"))

    # ================================================================
    # Analysis 3: Binned zero share vs cost advantage
    # ================================================================
    print("\n" + "=" * 70)
    print("Analysis 3: Cost advantage by intermittency (zero_share) quintile")
    print("=" * 70)

    df["zero_bin"] = pd.qcut(df.zero_share, q=5, duplicates="drop")
    zero_agg = df.groupby("zero_bin", observed=True).agg(
        n=("cost_delta", "size"),
        mean_zero=("zero_share", "mean"),
        cost_tsfm=("cost_tsfm", "mean"),
        cost_emp=("cost_emp", "mean"),
        delta=("cost_delta", "mean"),
    ).round(3)
    print(zero_agg.to_string())

    rho_zero, p_zero = sp_stats.spearmanr(df.zero_share, df.cost_delta)
    print(f"\nSpearman ρ(zero_share, cost_delta) = {rho_zero:.4f}, p = {p_zero:.4e}")
    chk.note("rho_zero", round(rho_zero, 4))

    # ================================================================
    # Analysis 4: ADF p-value (stationarity) vs cost advantage
    # ================================================================
    print("\n" + "=" * 70)
    print("Analysis 4: Cost advantage by ADF p-value (higher = less stationary)")
    print("=" * 70)

    df_adf = df.dropna(subset=["adf_p"])
    if len(df_adf) > 100:
        df_adf = df_adf.copy()
        df_adf["adf_bin"] = pd.qcut(df_adf.adf_p, q=5, duplicates="drop")
        adf_agg = df_adf.groupby("adf_bin", observed=True).agg(
            n=("cost_delta", "size"),
            mean_adf_p=("adf_p", "mean"),
            delta=("cost_delta", "mean"),
        ).round(3)
        print(adf_agg.to_string())

        rho_adf, p_adf = sp_stats.spearmanr(df_adf.adf_p, df_adf.cost_delta)
        print(f"\nSpearman ρ(adf_p, cost_delta) = {rho_adf:.4f}, p = {p_adf:.4e}")
        chk.note("rho_adf", round(rho_adf, 4))
    else:
        print("  Insufficient ADF data")

    # ================================================================
    # Analysis 5: Multivariate — which metric best explains TSFM advantage?
    # ================================================================
    print("\n" + "=" * 70)
    print("Analysis 5: OLS regression — cost_delta ~ drift + cv + zero + adf")
    print("=" * 70)

    df_reg = df.dropna(subset=["adf_p"]).copy()
    if len(df_reg) > 100:
        from sklearn.preprocessing import StandardScaler
        X_cols = ["drift_ratio", "cv_rolling", "zero_share", "adf_p"]
        X = df_reg[X_cols].to_numpy(float)
        X = np.clip(X, -10, 10)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        y_reg = df_reg.cost_delta.to_numpy(float)

        # OLS with statsmodels-free implementation
        X_s = np.column_stack([np.ones(len(X_s)), X_s])
        beta = np.linalg.lstsq(X_s, y_reg, rcond=None)[0]
        y_hat = X_s @ beta
        resid = y_reg - y_hat
        n, k = X_s.shape
        se = np.sqrt(np.diag(np.sum(resid**2) / (n - k) * np.linalg.inv(X_s.T @ X_s)))
        t_stat = beta / se

        print(f"{'variable':<16} {'beta':>8} {'t-stat':>8}")
        print("-" * 36)
        for j, name in enumerate(["intercept"] + X_cols):
            sig = "*" if abs(t_stat[j]) > 1.96 else ""
            print(f"{name:<16} {beta[j]:>8.4f} {t_stat[j]:>8.2f} {sig}")
            if j > 0:
                chk.note(f"ols_beta_{name}", round(beta[j], 4))
                chk.note(f"ols_t_{name}", round(t_stat[j], 2))

        r2 = 1 - np.sum(resid**2) / np.sum((y_reg - y_reg.mean())**2)
        print(f"\nR² = {r2:.4f}")
        chk.note("ols_r2", round(r2, 4))

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("Mechanism Summary")
    print("=" * 70)
    print(f"\n  Total observations: {len(df)}")
    print(f"  Mean cost delta (TSFM - emp): {df.cost_delta.mean():.2f}")
    print(f"  TSFM wins: {(df.cost_delta < 0).mean() * 100:.1f}% of SKU-months")
    print(f"\n  Correlations with TSFM advantage (negative delta = TSFM better):")
    print(f"    drift_ratio:  ρ = {rho_drift:+.4f} (p = {p_drift:.2e})")
    print(f"    cv_rolling:   ρ = {rho_cv:+.4f} (p = {p_cv:.2e})")
    print(f"    zero_share:   ρ = {rho_zero:+.4f} (p = {p_zero:.2e})")
    chk.note("tsfm_win_pct", round(float((df.cost_delta < 0).mean() * 100), 1))
    chk.note("mean_delta", round(float(df.cost_delta.mean()), 2))

    chk.n_rows = len(df)
    chk.note("wall_clock_sec", round(time.time() - t0, 1))
    chk.finish(ART / "checks" / "mechanism.json")
    print(f"\n总耗时 {time.time() - t0:.0f}s, 状态: {chk.status}")
    return chk.exit_code


if __name__ == "__main__":
    sys.exit(main())
