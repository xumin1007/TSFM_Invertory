"""WS-6 零/少 context 冷启动实验。

实验设计：
  1. 固定验证窗 (2019-07 ~ 2019-08)，对每个 SKU 截取最后 C 天历史
     C ∈ {7, 14, 30, 60, 90, full}
  2. 预测臂：chronos2-zs（各 context 长度）、knn-analogy（类比法）、
     emp-daily（需要 ≥30 天，短 context 不可用）
  3. 评价：预测层 (NPL) + 决策层 (Layer B cost/CSR, P3, α=0.95)
  4. 输出：性能 vs context 长度衰减曲线

用法:  PYTHONPATH=src python -m f2d.run_zhao_coldstart [--device mps]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch

from . import config as cfgmod
from .aggregation import convolve_varying, convolve_varying_pmf
from .checks import CheckResult
from .conventions import SEED_BASE
from .datasets import zhao
from .decision import costs_from_alpha, layer_b, order_up_to
from .metrics import evaluate_slice
from .models.chronos import BASE_CHECKPOINT, NATIVE_LEVELS, QuantileRepair, to_grid
from .uncertainty import paired_bootstrap, report

ART = cfgmod.ARTIFACT_DIR / "zhao_coldstart"
HORIZON = 7
VMAX_PRED = 300
VMAX_DEC = 60
ALPHA_PRIMARY = 0.95
KAPPA_H = 0.20
LEAD_DAYS = 1

VALID_ORIGINS = pd.date_range("2019-07-01", "2019-08-26", freq="7D")
VALID_MONTHS = [pd.Timestamp("2019-07-01"), pd.Timestamp("2019-08-01")]

CONTEXT_LENGTHS = [7, 14, 30, 60, 90]  # plus "full" handled separately
KNN_K = 5


def _truncate_context(ctx: dict[str, np.ndarray], max_days: int) -> dict[str, np.ndarray]:
    return {s: v[-max_days:] for s, v in ctx.items()}


def _knn_forecast(target_hist: np.ndarray, pool: dict[str, np.ndarray],
                  k: int, horizon: int, quantile_levels: np.ndarray
                  ) -> np.ndarray:
    """k-NN 类比法：找历史模式最相似的 k 条序列，用它们的未来段构建经验分位数。

    相似度：最后 min(len(target), len(neighbor)) 天的归一化 L2 距离。
    返回 shape (horizon, n_quantiles)。
    """
    t_len = len(target_hist)
    if t_len == 0:
        return np.full((horizon, len(quantile_levels)), np.nan)

    t_norm = target_hist / (np.std(target_hist) + 1e-8)
    t_mean = np.mean(target_hist)
    t_std = np.std(target_hist) + 1e-8

    dists = []
    futures = []
    for s, v in pool.items():
        if len(v) < t_len + horizon:
            continue
        # align to same length as target, ending horizon days before the end
        ref = v[-(t_len + horizon):-(horizon)]
        fut = v[-horizon:]
        ref_norm = ref / (np.std(ref) + 1e-8)
        d = np.sqrt(np.mean((t_norm - ref_norm) ** 2))
        dists.append(d)
        futures.append(fut)

    if len(futures) < k:
        return np.full((horizon, len(quantile_levels)), np.nan)

    idx = np.argsort(dists)[:k]
    fut_matrix = np.array([futures[i] for i in idx])  # (k, horizon)
    # Scale neighbors to target's level
    fut_scaled = fut_matrix * (t_std / (np.std(fut_matrix, axis=1, keepdims=True) + 1e-8))
    fut_scaled = fut_scaled - np.mean(fut_scaled, axis=1, keepdims=True) + t_mean

    result = np.quantile(fut_scaled, quantile_levels, axis=0).T  # (horizon, n_q)
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=500)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args(argv)

    from chronos import BaseChronosPipeline

    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    chk = CheckResult(step_id="ws6_coldstart", dataset="zhao", seed=SEED_BASE)
    cfg = cfgmod.load("zhao")
    y_min = float(cfg.metric["y_min"])

    raw = zhao.load_raw()
    daily, _ = zhao.build_daily_panel(raw)
    weekly = zhao.aggregate_to_period(daily, "W")
    panel, _ = zhao.build_panel(raw)
    target = weekly[weekly.origin.isin(VALID_ORIGINS)][["series_id", "origin", "y"]]

    rng = np.random.default_rng(SEED_BASE)
    pool = np.asarray(sorted(set(target.series_id)))
    keep = rng.choice(pool, size=min(args.n_series, len(pool)), replace=False)
    target = target[target.series_id.isin(keep)].reset_index(drop=True)
    daily = daily[daily.series_id.isin(keep)]
    sku_of = dict(zip(daily.series_id, daily.sku_ID))

    pipe = BaseChronosPipeline.from_pretrained(
        BASE_CHECKPOINT, device_map=args.device, torch_dtype=torch.float32)

    # Build full context pool (for knn neighbors, use ALL series not just sampled)
    daily_all, _ = zhao.build_daily_panel(raw)

    # ================================================================
    # Phase 1: Prediction layer — NPL vs context length
    # ================================================================
    print("=== Phase 1: 预测层 NPL vs context 长度 ===\n")
    pred_rows = []

    for origin in VALID_ORIGINS:
        cur = target[target.origin == origin]
        sids = cur.series_id.to_numpy()
        hist_full = daily[daily.d < origin]
        ctx_full = {s: g.sort_values("d").y.to_numpy(float)
                    for s, g in hist_full.groupby("series_id")}
        sids = np.array([s for s in sids if len(ctx_full.get(s, [])) >= 90 + HORIZON])
        if not len(sids):
            continue
        cur = cur[cur.series_id.isin(sids)].set_index("series_id")

        # knn pool: all series with enough history
        knn_pool_hist = daily_all[daily_all.d < origin]
        knn_pool = {s: g.sort_values("d").y.to_numpy(float)
                    for s, g in knn_pool_hist.groupby("series_id")
                    if len(g) >= 90 + HORIZON}

        for ctx_len in CONTEXT_LENGTHS + ["full"]:
            label = f"chronos2-C{ctx_len}" if ctx_len != "full" else "chronos2-full"
            if ctx_len == "full":
                ctx = {s: ctx_full[s] for s in sids}
            else:
                ctx = {s: ctx_full[s][-ctx_len:] for s in sids}

            q, _ = pipe.predict_quantiles(
                [torch.tensor(ctx[s], dtype=torch.float32) for s in sids],
                prediction_length=HORIZON,
                quantile_levels=list(NATIVE_LEVELS),
                batch_size=args.batch_size)
            g, _ = QuantileRepair()(to_grid(q))
            g = g.reshape(len(sids), HORIZON, -1)
            r = convolve_varying(NATIVE_LEVELS, [g[:, h, :] for h in range(HORIZON)],
                                 taus=(.5, .85), vmax=VMAX_PRED)
            pred_rows.append(pd.DataFrame({
                "variant": label, "series_id": sids, "month": origin,
                "split": "validation",
                "y": cur.loc[sids, "y"].to_numpy(),
                "q50": r[.5], "q85": r[.85], "w": 1.0}))

        # emp-daily baseline (needs >=30 days, use full context)
        emp_q50 = np.array([np.median(ctx_full[s]) * HORIZON for s in sids])
        emp_q85 = np.array([np.quantile(ctx_full[s], 0.85) * HORIZON for s in sids])
        pred_rows.append(pd.DataFrame({
            "variant": "emp-daily", "series_id": sids, "month": origin,
            "split": "validation",
            "y": cur.loc[sids, "y"].to_numpy(),
            "q50": emp_q50, "q85": emp_q85, "w": 1.0}))

        # knn analogy for each context length
        for ctx_len in CONTEXT_LENGTHS:
            label = f"knn-C{ctx_len}"
            knn_results = []
            for s in sids:
                h = ctx_full[s][-ctx_len:]
                excl_pool = {k: v for k, v in knn_pool.items() if k != s}
                qf = _knn_forecast(h, excl_pool, KNN_K, HORIZON, NATIVE_LEVELS)
                knn_results.append(qf)
            knn_arr = np.stack(knn_results)  # (n, horizon, n_q)
            r_knn = convolve_varying(NATIVE_LEVELS,
                                     [knn_arr[:, h, :] for h in range(HORIZON)],
                                     taus=(.5, .85), vmax=VMAX_PRED)
            pred_rows.append(pd.DataFrame({
                "variant": label, "series_id": sids, "month": origin,
                "split": "validation",
                "y": cur.loc[sids, "y"].to_numpy(),
                "q50": r_knn[.5], "q85": r_knn[.85], "w": 1.0}))

    pred = pd.concat(pred_rows, ignore_index=True)
    # Drop NaN rows from knn failures
    pred = pred.dropna(subset=["q50", "q85"]).reset_index(drop=True)

    # Align paired rows to common series×month
    keys_all = {v: set(map(tuple, pred[pred.variant == v][["series_id", "month"]].to_numpy()))
                for v in pred.variant.unique()}
    common = set.intersection(*keys_all.values())
    pred = pred[[k in common for k in
                 map(tuple, pred[["series_id", "month"]].to_numpy())]]
    pred = pred.sort_values(["variant", "month", "series_id"]).reset_index(drop=True)

    print("%-20s %-10s %-11s %-11s %-6s" % ("model", "NPL", "cov50", "cov85", "n"))
    print("-" * 60)
    for v in sorted(pred.variant.unique()):
        s = pred[pred.variant == v]
        r = evaluate_slice(s.y, s.q50, s.q85, s.w, y_min)
        print("%-20s %-10.5f %-11.4f %-11.4f %-6d" % (
            v, r.npl, r.cov_50_pos, r.cov_85_pos, r.n))
        chk.note(f"npl_{v}", round(r.npl, 5))

    # Bootstrap vs emp-daily
    variants_all = [v for v in sorted(pred.variant.unique()) if v != "emp-daily"]
    print("\n配对 bootstrap NPL (基准=emp-daily)")
    cis = paired_bootstrap(pred, "emp-daily", variants_all,
                           y_min=y_min, split="validation")
    print(report(cis)[["variant", "delta", "lo", "hi", "verdict"]]
          .round(5).to_string(index=False))

    # Save prediction results
    pred_summary = []
    for v in sorted(pred.variant.unique()):
        s = pred[pred.variant == v]
        r = evaluate_slice(s.y, s.q50, s.q85, s.w, y_min)
        # Parse context length
        if v.startswith("chronos2-C"):
            ctx_len = int(v.split("C")[1])
            method = "chronos2-zs"
        elif v == "chronos2-full":
            ctx_len = 9999
            method = "chronos2-zs"
        elif v.startswith("knn-C"):
            ctx_len = int(v.split("C")[1])
            method = "knn-analogy"
        elif v == "emp-daily":
            ctx_len = 9999
            method = "emp-daily"
        else:
            ctx_len = 9999
            method = v
        pred_summary.append(dict(method=method, context_days=ctx_len,
                                 npl=round(r.npl, 5),
                                 cov50=round(r.cov_50_pos, 4),
                                 cov85=round(r.cov_85_pos, 4), n=r.n))
    pd.DataFrame(pred_summary).to_csv(ART / "prediction_vs_context.csv", index=False)

    # ================================================================
    # Phase 2: Decision layer — cost/CSR vs context length
    # ================================================================
    print("\n=== Phase 2: 决策层 cost/CSR vs context 长度 ===\n")
    dec_rows = []

    for month in VALID_MONTHS:
        n_days = month.days_in_month + LEAD_DAYS
        snap = panel[panel.month == month].set_index("sku_ID")
        hist_full = daily[daily.d < month]
        ctx_full = {s: g.sort_values("d").y.to_numpy(float)
                    for s, g in hist_full.groupby("series_id")}
        # Only SKUs with >=90 days history AND cost data
        sids = np.array(sorted([s for s in ctx_full
                                if len(ctx_full[s]) >= 90 + HORIZON
                                and sku_of.get(s) in snap.index]))
        sk = np.array([sku_of[s] for s in sids])
        cur = snap.loc[sk]
        ip = cur["beginning_inventory"].to_numpy(float) + cur["on_order_inventory"].to_numpy(float)
        y = cur["observed_sales_next_month"].to_numpy(float)
        cost_i = cur["unit_cost_hist"].to_numpy(float)
        has_c = np.isfinite(cost_i)
        h_c, p_c = costs_from_alpha(cost_i, ALPHA_PRIMARY, KAPPA_H, 12)

        for ctx_len in CONTEXT_LENGTHS + ["full"]:
            label = f"chronos2-C{ctx_len}" if ctx_len != "full" else "chronos2-full"
            if ctx_len == "full":
                ctx = {s: ctx_full[s] for s in sids}
            else:
                ctx = {s: ctx_full[s][-ctx_len:] for s in sids}

            q, _ = pipe.predict_quantiles(
                [torch.tensor(ctx[s], dtype=torch.float32) for s in sids],
                prediction_length=n_days,
                quantile_levels=list(NATIVE_LEVELS),
                batch_size=args.batch_size)
            g, _ = QuantileRepair()(to_grid(q))
            g = g.reshape(len(sids), n_days, -1)
            grids = [g[:, i, :] for i in range(n_days)]
            pmf_pi = convolve_varying_pmf(NATIVE_LEVELS, grids, vmax=VMAX_DEC)
            pmf_r = convolve_varying_pmf(NATIVE_LEVELS,
                                          grids[:month.days_in_month], vmax=VMAX_DEC)
            m_ratio = n_days / month.days_in_month
            S = order_up_to(pmf_r, pmf_pi, ALPHA_PRIMARY, m_ratio)["P3"]
            r = layer_b(S, ip, y, h_c, p_c)
            dec_rows.append(dict(
                month=month, arm=label,
                CSR_ub=r.csr_upper_bound, FR_ub=r.fill_rate_upper_bound,
                cost=float(r.cost[has_c].mean()) if has_c.any() else float("nan"),
                n=len(sids), n_costed=int(has_c.sum())))
            print(f"  {month:%Y-%m} {label:<20} CSR={r.csr_upper_bound:.4f} "
                  f"cost={dec_rows[-1]['cost']:.2f}")

        # emp-daily
        emp_ctx = ctx_full
        emp_grids = []
        for d in range(n_days):
            day_q = np.array([np.quantile(emp_ctx[s], NATIVE_LEVELS) for s in sids])
            emp_grids.append(day_q)
        pmf_pi = convolve_varying_pmf(NATIVE_LEVELS, emp_grids, vmax=VMAX_DEC)
        pmf_r = convolve_varying_pmf(NATIVE_LEVELS,
                                      emp_grids[:month.days_in_month], vmax=VMAX_DEC)
        m_ratio = n_days / month.days_in_month
        S = order_up_to(pmf_r, pmf_pi, ALPHA_PRIMARY, m_ratio)["P3"]
        r = layer_b(S, ip, y, h_c, p_c)
        dec_rows.append(dict(
            month=month, arm="emp-daily",
            CSR_ub=r.csr_upper_bound, FR_ub=r.fill_rate_upper_bound,
            cost=float(r.cost[has_c].mean()) if has_c.any() else float("nan"),
            n=len(sids), n_costed=int(has_c.sum())))
        print(f"  {month:%Y-%m} {'emp-daily':<20} CSR={r.csr_upper_bound:.4f} "
              f"cost={dec_rows[-1]['cost']:.2f}")

    dec_df = pd.DataFrame(dec_rows)
    dec_df.to_csv(ART / "decision_vs_context.csv", index=False)

    print("\n--- 决策层两月平均 ---")
    dec_agg = dec_df.groupby("arm")[["CSR_ub", "FR_ub", "cost"]].mean()
    print(dec_agg.round(4).to_string())

    for arm in dec_agg.index:
        chk.note(f"decision_cost_{arm}", round(float(dec_agg.loc[arm, "cost"]), 4))
        chk.note(f"decision_csr_{arm}", round(float(dec_agg.loc[arm, "CSR_ub"]), 4))

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("WS-6 Cold-start 汇总: TSFM NPL 随 context 长度的衰减")
    print("=" * 70)
    for row in sorted(pred_summary, key=lambda x: (x["method"], x["context_days"])):
        print(f"  {row['method']:<16} C={row['context_days']:<5}  NPL={row['npl']:.5f}")

    chk.n_rows = len(pred) + len(dec_df)
    chk.note("wall_clock_sec", round(time.time() - t0, 1))
    chk.finish(ART / "checks" / "ws6.json")
    print(f"\n总耗时 {time.time() - t0:.0f}s, 状态: {chk.status}")
    return chk.exit_code


if __name__ == "__main__":
    sys.exit(main())
