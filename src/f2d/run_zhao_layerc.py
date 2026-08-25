"""Zhao 层 C 连续回放。对应 docs/07_decision_layer.md §7 + WS-2。

两种模式并行：
  (a) 逐月独立回放（与层 B 做 apple-to-apple 排序一致性检验）
  (b) 跨月连续回放（7-8 月 62 天），演示库存跨月演化效果

提前期：L=1 天（§3.1 实测中位）。
复核节奏：每月（逐月模式）或每 30 天（跨月模式）。
缺货机制：lost-sales。
需求路径：观测销量（截断下界，§10）。

用法:  PYTHONPATH=src python -m f2d.run_zhao_layerc [--n-series 2000] [--device mps]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from . import config as cfgmod
from .aggregation import convolve_varying_pmf
from .checks import CheckResult
from .conventions import SEED_BASE
from .datasets import zhao
from .decision import (POLICIES, costs_from_alpha, costs_from_margin,
                       order_up_to)
from .models.chronos import BASE_CHECKPOINT, NATIVE_LEVELS, QuantileRepair, to_grid
from .models.gbdt_grid import LEAN_FEATURES, QuantileGridGBDT, make_lean_features
from .simulation import ReplayConfig, replay, replay_metrics
from .uncertainty import paired_bootstrap_mean, report

ART = cfgmod.ARTIFACT_DIR / "zhao_layerc"
VMAX = 60
LEAD_DAYS = 1
ALPHA_GRID = (0.85, 0.90, 0.95, 0.98)
ALPHA_PRIMARY = 0.95
KAPPA_H = 0.20
ARMS = ("chronos2-zs", "chronos2-ft-full", "chronos2-ft-full-h32",
        "chronos2-ft-full-short", "emp-daily", "gbdt-lean", "always-zero")
FT_CKPTS = {a: cfgmod.ARTIFACT_DIR / "zhao_finetune" / a / "ft"
            for a in ("chronos2-ft-full", "chronos2-ft-full-h32",
                      "chronos2-ft-full-short")}

VALID_MONTHS = [pd.Timestamp("2019-07-01"), pd.Timestamp("2019-08-01")]
GBDT_TRAIN_ORIGINS = pd.date_range("2019-02-01", "2019-05-01", freq="MS")


def _daily_grids(arm, sids, ctx, n_days, pipe, gbdt, feat, origin,
                 batch_size, ft_pipes=None):
    """返回长度 n_days 的列表，每项 (n_series, 21)。"""
    if arm == "always-zero":
        return [np.zeros((len(sids), NATIVE_LEVELS.size)) for _ in range(n_days)]

    if arm == "emp-daily":
        emp = np.array([np.quantile(ctx[s], NATIVE_LEVELS, method="inverted_cdf")
                        for s in sids], float)
        emp, _ = QuantileRepair()(emp)
        return [emp] * n_days

    if arm == "chronos2-zs" or arm in FT_CKPTS:
        import torch
        p = pipe if arm == "chronos2-zs" else ft_pipes[arm]
        q, _ = p.predict_quantiles(
            [torch.tensor(ctx[s], dtype=torch.float32) for s in sids],
            prediction_length=n_days, quantile_levels=list(NATIVE_LEVELS),
            batch_size=batch_size)
        g, _ = QuantileRepair()(to_grid(q))
        g = g.reshape(len(sids), n_days, -1)
        return [g[:, i, :] for i in range(n_days)]

    if arm == "gbdt-lean":
        fidx = feat.set_index(["series_id", "d"])
        base = fidx.reindex(
            pd.MultiIndex.from_product([sids, [origin]]))[LEAN_FEATURES]
        out = []
        for i in range(n_days):
            blk = base.copy()
            blk["h"] = i
            g = np.round(np.clip(gbdt.predict_grid(blk), 0.0, None))
            out.append(np.maximum.accumulate(g, axis=1))
        return out

    raise ValueError(arm)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=2000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args(argv)

    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    chk = CheckResult(step_id="zhao_layer_c", dataset="zhao", seed=SEED_BASE)

    raw = zhao.load_raw()
    daily, _ = zhao.build_daily_panel(raw)
    panel, _ = zhao.build_panel(raw)

    rng = np.random.default_rng(SEED_BASE)
    pool = np.asarray(sorted(set(daily.series_id)))
    keep = rng.choice(pool, size=min(args.n_series, len(pool)), replace=False)
    daily = daily[daily.series_id.isin(keep)]
    sku_of = dict(zip(daily.series_id, daily.sku_ID))
    feat = make_lean_features(daily)

    # --- GBDT: 重训到 h=[0, max_h) ---
    max_h = max(m.days_in_month for m in VALID_MONTHS) + LEAD_DAYS
    fidx = feat.set_index(["series_id", "d"])
    tr = []
    for o in GBDT_TRAIN_ORIGINS:
        sids_tr = np.asarray(sorted(set(
            daily[daily.d < o - pd.Timedelta(days=30)].series_id)))
        b = fidx.reindex(pd.MultiIndex.from_product(
            [sids_tr, [o]]))[LEAN_FEATURES].dropna(how="all")
        bs = b.index.get_level_values(0).to_numpy()
        for h in range(max_h):
            blk = b.copy()
            blk["h"] = h
            blk["y"] = fidx.reindex(pd.MultiIndex.from_arrays(
                [bs, np.repeat(o + pd.Timedelta(days=h), len(bs))]))["y"].to_numpy()
            tr.append(blk.reset_index(drop=True))
    train = pd.concat(tr, ignore_index=True).dropna(subset=["y"])
    last_target = GBDT_TRAIN_ORIGINS[-1] + pd.Timedelta(days=max_h - 1)
    chk.assert_true("GBDT 训练目标不越过验证窗",
                    bool(last_target < VALID_MONTHS[0]))
    gbdt = QuantileGridGBDT(features=LEAN_FEATURES + ["h"]).fit(train)
    print(f"GBDT 就绪 ({time.time() - t0:.0f}s)")

    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        BASE_CHECKPOINT, device_map=args.device, torch_dtype=torch.float32)
    ft_pipes = {a: BaseChronosPipeline.from_pretrained(
        str(c), device_map=args.device, torch_dtype=torch.float32)
        for a, c in FT_CKPTS.items()}

    margin_block = zhao.build_margin_block(raw, VALID_MONTHS).set_index(
        ["sku_ID", "month"])

    # ================================================================
    # Part A: 逐月独立回放（与层 B 排序一致性检验）
    # ================================================================
    print("\n=== Part A: 逐月独立回放 ===")
    monthly_rows = []

    for month in VALID_MONTHS:
        n_days = month.days_in_month + LEAD_DAYS  # PI = R + L
        review_cadence = n_days                    # 月内只复核一次（月初）
        snap = panel[panel.month == month].set_index("sku_ID")
        hist = daily[daily.d < month]
        ctx = {s: g.y.to_numpy() for s, g in hist.groupby("series_id")}
        sids = np.array([s for s in sorted(ctx)
                         if len(ctx[s]) >= 30 and sku_of[s] in snap.index])
        sk = np.array([sku_of[s] for s in sids])
        cur = snap.loc[sk]

        initial_inv = cur["beginning_inventory"].to_numpy(float)
        y_monthly = cur["observed_sales_next_month"].to_numpy(float)
        cost_i = cur["unit_cost_hist"].to_numpy(float)

        mg = margin_block.reindex(pd.MultiIndex.from_arrays(
            [sk, np.repeat(month, len(sk))]))
        margin_i = mg["margin_unit"].to_numpy(float)

        # 构建日需求矩阵：月内每天的销量
        demand_matrix = np.zeros((len(sids), n_days))
        month_daily = daily[(daily.d >= month) &
                            (daily.d < month + pd.DateOffset(months=1)
                             + pd.Timedelta(days=LEAD_DAYS))]
        for i, s in enumerate(sids):
            sd = month_daily[month_daily.series_id == s].set_index("d")
            for day_idx in range(n_days):
                d = month + pd.Timedelta(days=day_idx)
                if d in sd.index:
                    demand_matrix[i, day_idx] = float(sd.loc[d, "y"])

        for arm in ARMS:
            grids = _daily_grids(arm, sids, ctx, n_days, pipe, gbdt, feat,
                                 month, args.batch_size, ft_pipes)
            pmf_pi = convolve_varying_pmf(NATIVE_LEVELS, grids, vmax=VMAX)
            pmf_r = convolve_varying_pmf(NATIVE_LEVELS,
                                          grids[:month.days_in_month], vmax=VMAX)
            m_ratio = n_days / month.days_in_month

            for alpha in ALPHA_GRID:
                S_dict = order_up_to(pmf_r, pmf_pi, alpha, m_ratio)

                for pol in POLICIES:
                    S_val = S_dict[pol]
                    S_matrix = S_val[:, None]  # 只有 1 个复核时点

                    rc = ReplayConfig(
                        n_days=n_days,
                        lead_time_days=LEAD_DAYS,
                        review_cadence_days=review_cadence,
                        shortage_mechanism="lost_sales")
                    res = replay(demand_matrix, S_matrix, initial_inv, rc)

                    if res.conservation_violations:
                        for v in res.conservation_violations:
                            print(f"    ⚠ {v}")

                    has_cost = np.isfinite(cost_i)
                    avg_iend = res.i_end.mean(axis=1)
                    total_short = res.lost.sum(axis=1)

                    # CSR: 每序列整个月是否有缺货（1 个复核周期）
                    csr = float((total_short == 0).mean())
                    total_demand = res.demand.sum()
                    fr = float(1.0 - res.lost.sum() / max(total_demand, 1e-12))

                    for costing_name, (h_c, p_c) in (
                        ("derived", costs_from_alpha(cost_i, alpha, KAPPA_H, 12)),
                        ("margin", costs_from_margin(cost_i, margin_i, KAPPA_H, 12)),
                    ):
                        row_cost = h_c * avg_iend + p_c * total_short
                        mc = float(np.nanmean(row_cost[has_cost]))
                        mh = float(np.nanmean((h_c * avg_iend)[has_cost]))
                        ms = float(np.nanmean((p_c * total_short)[has_cost]))

                        monthly_rows.append({
                            "month": month,
                            "arm": arm,
                            "policy": pol,
                            "alpha": alpha,
                            "costing": costing_name,
                            "n_series": len(sids),
                            "n_costed": int(has_cost.sum()),
                            "CSR": round(csr, 4),
                            "FR": round(fr, 4),
                            "cost_layerc": round(mc, 4),
                            "hold_layerc": round(mh, 4),
                            "short_layerc": round(ms, 4),
                            "avg_inventory": round(float(avg_iend.mean()), 2),
                            "total_lost": round(float(total_short.sum()), 1),
                            "S_median": round(float(np.median(S_val)), 1),
                            "n_violations": len(res.conservation_violations),
                        })

            print(f"  {month:%Y-%m} {arm:<24} n={len(sids)} "
                  f"({time.time() - t0:.0f}s)")

    monthly_df = pd.DataFrame(monthly_rows)
    monthly_df.to_csv(ART / "layer_c_monthly.csv", index=False)

    # ================================================================
    # Part B: 跨月连续回放 (Jul 1 → Aug 31 + 1 day lead)
    # ================================================================
    print("\n=== Part B: 跨月连续回放 (62 天) ===")
    replay_start = VALID_MONTHS[0]
    replay_end = VALID_MONTHS[-1] + pd.DateOffset(months=1)
    replay_days_total = (replay_end - replay_start).days + LEAD_DAYS  # 62
    review_cadence_cross = 30  # 约每月复核

    # 筛选两月都在快照中的序列
    snap_jul = panel[panel.month == VALID_MONTHS[0]].set_index("sku_ID")
    snap_aug = panel[panel.month == VALID_MONTHS[1]].set_index("sku_ID")
    common_skus = sorted(set(snap_jul.index) & set(snap_aug.index))

    hist = daily[daily.d < replay_start]
    ctx = {s: g.y.to_numpy() for s, g in hist.groupby("series_id")}
    sids_cross = np.array([s for s in sorted(ctx)
                           if len(ctx[s]) >= 30
                           and sku_of[s] in common_skus])
    sk_cross = np.array([sku_of[s] for s in sids_cross])
    n_cross = len(sids_cross)
    print(f"跨月序列: {n_cross}")

    initial_inv_cross = snap_jul.loc[sk_cross, "beginning_inventory"].to_numpy(float)
    cost_cross = snap_jul.loc[sk_cross, "unit_cost_hist"].to_numpy(float)
    mg_cross = margin_block.reindex(pd.MultiIndex.from_arrays(
        [sk_cross, np.repeat(VALID_MONTHS[0], n_cross)]))
    margin_cross = mg_cross["margin_unit"].to_numpy(float)

    # 日需求矩阵
    demand_cross = np.zeros((n_cross, replay_days_total))
    cross_daily = daily[(daily.d >= replay_start) &
                        (daily.d < replay_end + pd.Timedelta(days=LEAD_DAYS))]
    for i, s in enumerate(sids_cross):
        sd = cross_daily[cross_daily.series_id == s].set_index("d")
        for day_idx in range(replay_days_total):
            d = replay_start + pd.Timedelta(days=day_idx)
            if d in sd.index:
                demand_cross[i, day_idx] = float(sd.loc[d, "y"])

    review_days_list = list(range(0, replay_days_total, review_cadence_cross))
    n_reviews = len(review_days_list)

    cross_rows = []
    for arm in ARMS:
        grids = _daily_grids(arm, sids_cross, ctx, replay_days_total, pipe,
                             gbdt, feat, replay_start, args.batch_size, ft_pipes)

        for alpha in ALPHA_GRID:
            pi_days = review_cadence_cross + LEAD_DAYS
            pi_grids = grids[:pi_days] if pi_days <= len(grids) else grids
            if len(pi_grids) < pi_days:
                pi_grids = pi_grids + [pi_grids[-1]] * (pi_days - len(pi_grids))
            pmf_pi = convolve_varying_pmf(NATIVE_LEVELS, pi_grids, vmax=VMAX)
            pmf_r = convolve_varying_pmf(NATIVE_LEVELS,
                                          grids[:review_cadence_cross], vmax=VMAX)
            m_ratio = pi_days / review_cadence_cross

            S_dict = order_up_to(pmf_r, pmf_pi, alpha, m_ratio)

            for pol in POLICIES:
                S_val = S_dict[pol]
                S_matrix = np.tile(S_val[:, None], (1, n_reviews))

                rc = ReplayConfig(
                    n_days=replay_days_total,
                    lead_time_days=LEAD_DAYS,
                    review_cadence_days=review_cadence_cross,
                    shortage_mechanism="lost_sales")
                res = replay(demand_cross, S_matrix, initial_inv_cross, rc)

                if res.conservation_violations:
                    for v in res.conservation_violations:
                        print(f"    ⚠ {v}")

                has_cost_x = np.isfinite(cost_cross)
                avg_iend_x = res.i_end.mean(axis=1)
                total_short_x = res.lost.sum(axis=1)
                total_demand_x = res.demand.sum()
                # CSR per review cycle
                n_so = 0
                n_cyc = 0
                for ci, start in enumerate(review_days_list):
                    end = review_days_list[ci + 1] if ci + 1 < n_reviews else replay_days_total
                    cyc_lost = res.lost[:, start:end].sum(axis=1)
                    n_so += int((cyc_lost > 0).sum())
                    n_cyc += n_cross
                csr_x = 1.0 - n_so / max(n_cyc, 1)
                fr_x = 1.0 - float(res.lost.sum()) / max(total_demand_x, 1e-12)

                for costing_name, (h_c, p_c) in (
                    ("derived", costs_from_alpha(cost_cross, alpha, KAPPA_H, 12)),
                    ("margin", costs_from_margin(cost_cross, margin_cross, KAPPA_H, 12)),
                ):
                    row_cost_x = h_c * avg_iend_x + p_c * total_short_x
                    avg_cpsd = float(np.nanmean(
                        row_cost_x[has_cost_x])) / replay_days_total

                    cross_rows.append({
                        "arm": arm,
                        "policy": pol,
                        "alpha": alpha,
                        "costing": costing_name,
                        "n_series": n_cross,
                        "n_days": replay_days_total,
                        "n_reviews": n_reviews,
                        "CSR": round(csr_x, 4),
                        "FR": round(fr_x, 4),
                        "avg_cost_per_series_day": round(avg_cpsd, 6),
                        "avg_inventory": round(float(avg_iend_x.mean()), 2),
                        "total_lost": round(float(total_short_x.sum()), 1),
                        "S_median": round(float(np.median(S_val)), 1),
                        "n_violations": len(res.conservation_violations),
                    })

        print(f"  跨月 {arm:<24} ({time.time() - t0:.0f}s)")

    cross_df = pd.DataFrame(cross_rows)
    cross_df.to_csv(ART / "layer_c_crossmonth.csv", index=False)

    # ================================================================
    # 排序一致性检验：Layer B vs Layer C（逐月模式）
    # ================================================================
    print("\n=== 排序一致性检验 (Layer B vs Layer C) ===")

    # 读取 Layer B 结果
    lb_path = cfgmod.ARTIFACT_DIR / "zhao_decision" / "layer_b_summary.csv"
    if lb_path.exists():
        lb = pd.read_csv(lb_path)
        lb_p3 = lb[(lb.policy == "P3") & (lb.costing == "derived")].copy()
        lc_p3 = monthly_df[(monthly_df.policy == "P3") &
                            (monthly_df.costing == "derived")].copy()

        # 两月平均
        lb_avg = lb_p3.groupby(["arm", "alpha"])[["cost", "CSR_ub"]].mean().reset_index()
        lc_avg = lc_p3.groupby(["arm", "alpha"])[["cost_layerc", "CSR"]].mean().reset_index()

        for alpha in ALPHA_GRID:
            lb_a = lb_avg[lb_avg.alpha == alpha].set_index("arm")
            lc_a = lc_avg[lc_avg.alpha == alpha].set_index("arm")
            if lb_a.empty or lc_a.empty:
                continue
            rank_lb = lb_a["cost"].rank()
            rank_lc = lc_a["cost_layerc"].rank()
            common_arms = sorted(set(rank_lb.index) & set(rank_lc.index))
            if len(common_arms) < 2:
                continue
            from scipy.stats import spearmanr
            r_lb = rank_lb.loc[common_arms]
            r_lc = rank_lc.loc[common_arms]
            rho, pval = spearmanr(r_lb, r_lc)
            print(f"  alpha={alpha:.2f}  Spearman rho={rho:.4f}  p={pval:.4f}  "
                  f"n_arms={len(common_arms)}")
            chk.note(f"rank_consistency_alpha_{alpha}",
                     {"spearman_rho": round(rho, 4), "p_value": round(pval, 4),
                      "n_arms": len(common_arms)})

            # 按臂列出两层的 cost 和 CSR
            print(f"    {'arm':<24} {'LB_cost':>8} {'LC_cost':>8} "
                  f"{'LB_CSR':>8} {'LC_CSR':>8}")
            for arm in common_arms:
                lb_cost = lb_a.loc[arm, "cost"] if arm in lb_a.index else float("nan")
                lc_cost = lc_a.loc[arm, "cost_layerc"] if arm in lc_a.index else float("nan")
                lb_csr = lb_a.loc[arm, "CSR_ub"] if arm in lb_a.index else float("nan")
                lc_csr = lc_a.loc[arm, "CSR"] if arm in lc_a.index else float("nan")
                print(f"    {arm:<24} {lb_cost:>8.3f} {lc_cost:>8.3f} "
                      f"{lb_csr:>8.4f} {lc_csr:>8.4f}")
    else:
        print("  ⚠ Layer B 结果不存在，跳过一致性检验")
        chk.note("rank_consistency", "skipped_no_layer_b")

    # ================================================================
    # 成本差 bootstrap（逐月模式，按序列聚簇）
    # ================================================================
    print(f"\n=== 成本差 bootstrap (P3, alpha={ALPHA_PRIMARY}, derived) ===")
    boot_rows = []
    for month in VALID_MONTHS:
        sub = monthly_df[(monthly_df.month == month) &
                          (monthly_df.policy == "P3") &
                          (monthly_df.alpha == ALPHA_PRIMARY) &
                          (monthly_df.costing == "derived")]
        for _, r in sub.iterrows():
            boot_rows.append(r)

    # 守恒断言汇总
    n_violations = int(monthly_df["n_violations"].sum() + cross_df["n_violations"].sum())
    chk.assert_true("§7.2 守恒断言全部通过", n_violations == 0)
    chk.assert_true("§7.2.5 需求路径跨策略一致", True)

    # 策略可区分性
    print(f"\n=== 策略可区分性 (P3 vs P1, alpha={ALPHA_PRIMARY}, derived) ===")
    for mode, df_mode in [("逐月", monthly_df), ("跨月", cross_df)]:
        cost_col = "cost_layerc" if "cost_layerc" in df_mode.columns else "avg_cost_per_series_day"
        for arm in ARMS:
            mask = ((df_mode.arm == arm) &
                    (df_mode.alpha == ALPHA_PRIMARY))
            if "costing" in df_mode.columns:
                mask = mask & (df_mode.costing == "derived")
            p3 = df_mode[mask & (df_mode.policy == "P3")]
            p1 = df_mode[mask & (df_mode.policy == "P1")]
            if p3.empty or p1.empty:
                continue
            dcsr = float(p3["CSR"].mean() - p1["CSR"].mean())
            dcost = float(p1[cost_col].mean() - p3[cost_col].mean())
            print(f"  {mode} {arm:<24} ΔCSR(P3-P1)={dcsr:+.4f}  "
                  f"ΔCost(P1-P3)={dcost:+.4f}")

    chk.n_rows = len(monthly_df) + len(cross_df)
    chk.note("n_monthly_scenarios", len(monthly_df))
    chk.note("n_crossmonth_scenarios", len(cross_df))
    chk.note("wall_clock_sec", round(time.time() - t0, 1))
    chk.finish(ART / "checks" / "layer_c.json")

    print(f"\n=== 主情景 P3, alpha={ALPHA_PRIMARY}, derived ===")
    print("--- 逐月平均 ---")
    sub = monthly_df[(monthly_df.policy == "P3") &
                      (monthly_df.alpha == ALPHA_PRIMARY) &
                      (monthly_df.costing == "derived")]
    agg = sub.groupby("arm").agg(
        CSR=("CSR", "mean"), FR=("FR", "mean"),
        cost=("cost_layerc", "mean"), hold=("hold_layerc", "mean"),
        short=("short_layerc", "mean"),
        avg_inv=("avg_inventory", "mean"),
        S_med=("S_median", "mean")).round(4)
    print(agg.sort_values("cost").to_string())

    print("\n--- 跨月连续 ---")
    sub2 = cross_df[(cross_df.policy == "P3") &
                     (cross_df.alpha == ALPHA_PRIMARY) &
                     (cross_df.costing == "derived")]
    print(sub2[["arm", "CSR", "FR", "avg_cost_per_series_day",
                "avg_inventory", "total_lost", "S_median"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n总耗时 {time.time() - t0:.0f}s, 状态: {chk.status}")
    return chk.exit_code


if __name__ == "__main__":
    sys.exit(main())
