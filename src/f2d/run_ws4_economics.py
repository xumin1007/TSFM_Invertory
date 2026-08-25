"""WS-4 经济显著性与敏感性分析。

4a: 年化经济影响（单期成本差 × SKU 数 × 12 → 企业级年化节省）
4b: 敏感性扫描（κ_h, 半合成提前期, α 已有 Pareto 数据）
4c: 实践指南表

用法:  PYTHONPATH=src python -m f2d.run_ws4_economics [--n-series 2000] [--device mps]
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
                       layer_b, order_up_to)
from .models.chronos import BASE_CHECKPOINT, NATIVE_LEVELS, QuantileRepair, to_grid
from .models.gbdt_grid import LEAN_FEATURES, QuantileGridGBDT, make_lean_features
from .uncertainty import paired_bootstrap_mean, report

ART = cfgmod.ARTIFACT_DIR / "zhao_economics"
VMAX = 60
ALPHA_PRIMARY = 0.95
KAPPA_H_CENTER = 0.20
KAPPA_H_GRID = (0.10, 0.15, 0.20, 0.25, 0.30)
LEAD_DAYS_GRID = (1, 3, 7, 14)
VALID_MONTHS = [pd.Timestamp("2019-07-01"), pd.Timestamp("2019-08-01")]
GBDT_TRAIN_ORIGINS = pd.date_range("2019-02-01", "2019-05-01", freq="MS")

CORE_ARMS = ("chronos2-zs", "chronos2-ft-full", "emp-daily",
             "gbdt-lean", "always-zero")
FT_CKPTS = {a: cfgmod.ARTIFACT_DIR / "zhao_finetune" / a / "ft"
            for a in ("chronos2-ft-full",)}

FULL_SKU_COUNT = 28626  # Zhao dataset total unique SKUs


def _daily_grids(arm, sids, ctx, n_days, pipe, gbdt, feat, origin,
                 batch_size, ft_pipes=None):
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
    chk = CheckResult(step_id="ws4_economics", dataset="zhao", seed=SEED_BASE)

    raw = zhao.load_raw()
    daily, _ = zhao.build_daily_panel(raw)
    panel, _ = zhao.build_panel(raw)

    rng = np.random.default_rng(SEED_BASE)
    pool = np.asarray(sorted(set(daily.series_id)))
    keep = rng.choice(pool, size=min(args.n_series, len(pool)), replace=False)
    daily = daily[daily.series_id.isin(keep)]
    sku_of = dict(zip(daily.series_id, daily.sku_ID))
    feat = make_lean_features(daily)

    # GBDT training (same as Layer B)
    max_h_base = max(m.days_in_month for m in VALID_MONTHS) + max(LEAD_DAYS_GRID)
    fidx = feat.set_index(["series_id", "d"])
    tr = []
    for o in GBDT_TRAIN_ORIGINS:
        sids_tr = np.asarray(sorted(set(
            daily[daily.d < o - pd.Timedelta(days=30)].series_id)))
        b = fidx.reindex(pd.MultiIndex.from_product(
            [sids_tr, [o]]))[LEAN_FEATURES].dropna(how="all")
        bs = b.index.get_level_values(0).to_numpy()
        for h in range(max_h_base):
            blk = b.copy()
            blk["h"] = h
            blk["y"] = fidx.reindex(pd.MultiIndex.from_arrays(
                [bs, np.repeat(o + pd.Timedelta(days=h), len(bs))]))["y"].to_numpy()
            tr.append(blk.reset_index(drop=True))
    train = pd.concat(tr, ignore_index=True).dropna(subset=["y"])
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
    # Precompute grids for all arms × months (reused across sweeps)
    # ================================================================
    grids_cache = {}  # (arm, month) -> (sids, grids_max_h, snap_row)
    month_data = {}

    for month in VALID_MONTHS:
        snap = panel[panel.month == month].set_index("sku_ID")
        hist = daily[daily.d < month]
        ctx = {s: g.y.to_numpy() for s, g in hist.groupby("series_id")}
        sids = np.array([s for s in sorted(ctx)
                         if len(ctx[s]) >= 30 and sku_of[s] in snap.index])
        sk = np.array([sku_of[s] for s in sids])
        cur = snap.loc[sk]

        ip = cur["beginning_inventory"].to_numpy(float) + cur["on_order_inventory"].to_numpy(float)
        y = cur["observed_sales_next_month"].to_numpy(float)
        cost_i = cur["unit_cost_hist"].to_numpy(float)
        mg = margin_block.reindex(pd.MultiIndex.from_arrays(
            [sk, np.repeat(month, len(sk))]))
        margin_i = mg["margin_unit"].to_numpy(float)

        month_data[month] = dict(sids=sids, sk=sk, ip=ip, y=y,
                                  cost_i=cost_i, margin_i=margin_i)

        max_h = month.days_in_month + max(LEAD_DAYS_GRID)
        for arm in CORE_ARMS:
            grids = _daily_grids(arm, sids, ctx, max_h, pipe, gbdt, feat,
                                 month, args.batch_size, ft_pipes)
            grids_cache[(arm, month)] = grids
            print(f"  预测缓存: {month:%Y-%m} {arm:<20} ({time.time() - t0:.0f}s)")

    # ================================================================
    # 4b-1: κ_h 敏感性（固定 L=1, α=0.95, P3）
    # ================================================================
    print("\n=== 4b-1: κ_h 敏感性扫描 ===")
    kh_rows = []
    L = 1
    for kh in KAPPA_H_GRID:
        for month in VALID_MONTHS:
            md = month_data[month]
            n_days = month.days_in_month + L
            for arm in CORE_ARMS:
                grids = grids_cache[(arm, month)]
                pmf_pi = convolve_varying_pmf(NATIVE_LEVELS, grids[:n_days], vmax=VMAX)
                pmf_r = convolve_varying_pmf(NATIVE_LEVELS,
                                              grids[:month.days_in_month], vmax=VMAX)
                m_ratio = n_days / month.days_in_month
                S = order_up_to(pmf_r, pmf_pi, ALPHA_PRIMARY, m_ratio)["P3"]
                h_c, p_c = costs_from_alpha(md["cost_i"], ALPHA_PRIMARY, kh, 12)
                r = layer_b(S, md["ip"], md["y"], h_c, p_c)
                has_c = np.isfinite(md["cost_i"])
                cost_mean = float(r.cost[has_c].mean()) if has_c.any() else float("nan")
                kh_rows.append(dict(kappa_h=kh, month=month, arm=arm,
                                    CSR_ub=r.csr_upper_bound,
                                    cost=cost_mean, n=int(has_c.sum())))

    kh_df = pd.DataFrame(kh_rows)
    kh_agg = kh_df.groupby(["kappa_h", "arm"])[["cost", "CSR_ub"]].mean().reset_index()
    kh_agg.to_csv(ART / "sensitivity_kappa_h.csv", index=False)

    print(f"\n{'arm':<22}", end="")
    for kh in KAPPA_H_GRID:
        print(f"  κ={kh:.2f}", end="")
    print()
    for arm in CORE_ARMS:
        sub = kh_agg[kh_agg.arm == arm].set_index("kappa_h")
        print(f"{arm:<22}", end="")
        for kh in KAPPA_H_GRID:
            c = sub.loc[kh, "cost"] if kh in sub.index else float("nan")
            print(f"  {c:>7.2f}", end="")
        print()

    # Check rank stability
    for kh in KAPPA_H_GRID:
        sub = kh_agg[kh_agg.kappa_h == kh].set_index("arm")
        rank = sub["cost"].rank()
        chk.note(f"rank_kh_{kh}", {a: int(rank.loc[a]) for a in CORE_ARMS if a in rank.index})

    # ================================================================
    # 4b-2: 半合成提前期敏感性（Zhao 数据 + 人造 L）
    # ================================================================
    print("\n=== 4b-2: 提前期敏感性扫描 ===")
    lt_rows = []
    for L in LEAD_DAYS_GRID:
        for month in VALID_MONTHS:
            md = month_data[month]
            n_days = month.days_in_month + L
            for arm in CORE_ARMS:
                grids = grids_cache[(arm, month)]
                actual_len = len(grids)
                g = grids[:n_days] if n_days <= actual_len else \
                    grids + [grids[-1]] * (n_days - actual_len)
                pmf_pi = convolve_varying_pmf(NATIVE_LEVELS, g, vmax=VMAX)
                pmf_r = convolve_varying_pmf(NATIVE_LEVELS,
                                              grids[:month.days_in_month], vmax=VMAX)
                m_ratio = n_days / month.days_in_month
                S = order_up_to(pmf_r, pmf_pi, ALPHA_PRIMARY, m_ratio)["P3"]
                h_c, p_c = costs_from_alpha(md["cost_i"], ALPHA_PRIMARY,
                                             KAPPA_H_CENTER, 12)
                r = layer_b(S, md["ip"], md["y"], h_c, p_c)
                has_c = np.isfinite(md["cost_i"])
                cost_mean = float(r.cost[has_c].mean()) if has_c.any() else float("nan")
                lt_rows.append(dict(lead_time=L, month=month, arm=arm,
                                    CSR_ub=r.csr_upper_bound,
                                    cost=cost_mean,
                                    S_med=float(np.median(S)),
                                    n=int(has_c.sum())))

    lt_df = pd.DataFrame(lt_rows)
    lt_agg = lt_df.groupby(["lead_time", "arm"])[["cost", "CSR_ub", "S_med"]].mean().reset_index()
    lt_agg.to_csv(ART / "sensitivity_lead_time.csv", index=False)

    print(f"\n{'arm':<22}", end="")
    for L in LEAD_DAYS_GRID:
        print(f"  L={L:>2d}d", end="")
    print("    (cost)")
    for arm in CORE_ARMS:
        sub = lt_agg[lt_agg.arm == arm].set_index("lead_time")
        print(f"{arm:<22}", end="")
        for L in LEAD_DAYS_GRID:
            c = sub.loc[L, "cost"] if L in sub.index else float("nan")
            print(f"  {c:>6.1f}", end="")
        print()

    print(f"\n{'arm':<22}", end="")
    for L in LEAD_DAYS_GRID:
        print(f"  L={L:>2d}d", end="")
    print("    (CSR)")
    for arm in CORE_ARMS:
        sub = lt_agg[lt_agg.arm == arm].set_index("lead_time")
        print(f"{arm:<22}", end="")
        for L in LEAD_DAYS_GRID:
            c = sub.loc[L, "CSR_ub"] if L in sub.index else float("nan")
            print(f"  {c:>6.4f}", end="")
        print()

    # P3 vs P1 gap by lead time
    print("\n--- P3 vs P1 策略差距 vs 提前期 ---")
    p1_lt_rows = []
    for L in LEAD_DAYS_GRID:
        for month in VALID_MONTHS:
            md = month_data[month]
            n_days = month.days_in_month + L
            for arm in CORE_ARMS:
                if arm == "always-zero":
                    continue
                grids = grids_cache[(arm, month)]
                actual_len = len(grids)
                g = grids[:n_days] if n_days <= actual_len else \
                    grids + [grids[-1]] * (n_days - actual_len)
                pmf_pi = convolve_varying_pmf(NATIVE_LEVELS, g, vmax=VMAX)
                pmf_r = convolve_varying_pmf(NATIVE_LEVELS,
                                              grids[:month.days_in_month], vmax=VMAX)
                m_ratio = n_days / month.days_in_month
                S_all = order_up_to(pmf_r, pmf_pi, ALPHA_PRIMARY, m_ratio)
                h_c, p_c = costs_from_alpha(md["cost_i"], ALPHA_PRIMARY,
                                             KAPPA_H_CENTER, 12)
                has_c = np.isfinite(md["cost_i"])
                for pol in ("P1", "P3"):
                    r = layer_b(S_all[pol], md["ip"], md["y"], h_c, p_c)
                    p1_lt_rows.append(dict(lead_time=L, month=month, arm=arm,
                                           policy=pol, CSR_ub=r.csr_upper_bound,
                                           cost=float(r.cost[has_c].mean())))

    p1_lt_df = pd.DataFrame(p1_lt_rows)
    p1_lt_agg = p1_lt_df.groupby(["lead_time", "arm", "policy"])[["CSR_ub", "cost"]].mean().reset_index()
    for L in LEAD_DAYS_GRID:
        for arm in [a for a in CORE_ARMS if a != "always-zero"]:
            p3 = p1_lt_agg[(p1_lt_agg.lead_time == L) & (p1_lt_agg.arm == arm)
                            & (p1_lt_agg.policy == "P3")]
            p1 = p1_lt_agg[(p1_lt_agg.lead_time == L) & (p1_lt_agg.arm == arm)
                            & (p1_lt_agg.policy == "P1")]
            if p3.empty or p1.empty:
                continue
            dcsr = float(p3.CSR_ub.iloc[0] - p1.CSR_ub.iloc[0])
            dcost = float(p1.cost.iloc[0] - p3.cost.iloc[0])
            print(f"  L={L:>2d}  {arm:<20}  ΔCSR(P3-P1)={dcsr:+.4f}  "
                  f"ΔCost(P1-P3)={dcost:+.2f}")

    # ================================================================
    # 4a: 年化经济影响
    # ================================================================
    print("\n=== 4a: 年化经济影响 ===")

    # Use Layer B results (P3, α=0.95, derived)
    lb_path = cfgmod.ARTIFACT_DIR / "zhao_decision" / "layer_b_summary.csv"
    lb = pd.read_csv(lb_path)
    lb_main = lb[(lb.policy == "P3") & (lb.alpha == ALPHA_PRIMARY) &
                  (lb.costing == "derived")]

    baseline_cost = float(lb_main[lb_main.arm == "emp-daily"]["cost"].iloc[0])
    scale_factor = FULL_SKU_COUNT / args.n_series  # sample → full fleet

    print(f"基准: emp-daily, cost/SKU·月 = {baseline_cost:.2f}")
    print(f"全量 SKU: {FULL_SKU_COUNT}, 采样倍率: {scale_factor:.1f}x\n")

    annual_rows = []
    for arm in CORE_ARMS:
        if arm == "emp-daily":
            continue
        row = lb_main[lb_main.arm == arm]
        if row.empty:
            continue
        cost = float(row["cost"].iloc[0])
        delta = cost - baseline_cost
        annual_delta = delta * FULL_SKU_COUNT * 12
        csr = float(row["CSR_ub"].iloc[0])
        annual_rows.append(dict(
            arm=arm,
            cost_per_sku_month=round(cost, 2),
            delta_per_sku_month=round(delta, 2),
            delta_pct=round(100 * delta / baseline_cost, 1),
            annual_delta=round(annual_delta, 0),
            CSR_ub=round(csr, 4),
        ))
        print(f"  {arm:<22}  Δcost/SKU·月={delta:+.2f} ({100*delta/baseline_cost:+.1f}%)  "
              f"年化={annual_delta:+,.0f}  CSR={csr:.4f}")

    annual_df = pd.DataFrame(annual_rows)
    annual_df.to_csv(ART / "annualized_impact.csv", index=False)

    # TCO comparison (qualitative + rough numbers)
    print("\n--- 部署 TCO 对比 ---")
    print("  TSFM 零样本:  推理仅用 GPU, 无训练/特征工程/数据质量监控")
    print("  GBDT pipeline: 特征工程 + 月度重训 + 数据质量 + 模型监控")
    print("  TSFM 微调:    + 一次性微调成本（8h A100），后续与零样本相同")

    # ================================================================
    # 4b-3: α 敏感性（已有 Pareto 数据，补充经济解读）
    # ================================================================
    print("\n=== 4b-3: α 敏感性（经济解读）===")
    pareto_path = cfgmod.ARTIFACT_DIR / "zhao_analysis" / "pareto_agg.csv"
    if pareto_path.exists():
        pareto = pd.read_csv(pareto_path)
        pareto_all = pareto[pareto.slice == "all"]

        # 成本节省率 vs alpha
        print(f"\n{'alpha':>6}", end="")
        for arm in ["chronos2-zs", "chronos2-ft-full", "gbdt-lean"]:
            print(f"  {arm:>22}", end="")
        print("    (节省率 vs emp-daily)")
        for alpha in sorted(pareto_all.alpha.unique()):
            sub = pareto_all[pareto_all.alpha == alpha].set_index("arm")
            if "emp-daily" not in sub.index:
                continue
            base_c = sub.loc["emp-daily", "cost_mean"]
            print(f"{alpha:>6.3f}", end="")
            for arm in ["chronos2-zs", "chronos2-ft-full", "gbdt-lean"]:
                if arm in sub.index:
                    c = sub.loc[arm, "cost_mean"]
                    pct = 100 * (c - base_c) / base_c
                    print(f"  {pct:>+21.1f}%", end="")
                else:
                    print(f"  {'—':>22}", end="")
            print()

        # At extreme alpha, compute annual impact
        print("\n--- 极端 α 下的年化节省（chronos2-zs vs emp-daily）---")
        for alpha in [0.90, 0.95, 0.98, 0.99, 0.995]:
            sub = pareto_all[pareto_all.alpha == alpha].set_index("arm")
            if "emp-daily" not in sub.index or "chronos2-zs" not in sub.index:
                continue
            delta = sub.loc["chronos2-zs", "cost_mean"] - sub.loc["emp-daily", "cost_mean"]
            annual = delta * FULL_SKU_COUNT * 12
            print(f"  α={alpha:.3f}  Δcost={delta:+.2f}  年化={annual:+,.0f}")
            chk.note(f"annual_saving_alpha_{alpha}", round(annual, 0))

    # ================================================================
    # 4c: 实践指南
    # ================================================================
    print("\n=== 4c: 实践指南 ===")
    guide_rows = []
    for arm in CORE_ARMS:
        if arm == "always-zero":
            continue
        sub = kh_agg[kh_agg.arm == arm]
        dominated = True
        for kh in KAPPA_H_GRID:
            best = kh_agg[kh_agg.kappa_h == kh].sort_values("cost").iloc[0]
            if best.arm == arm:
                dominated = False
                break
        sub_lt = lt_agg[lt_agg.arm == arm]
        guide_rows.append(dict(
            arm=arm,
            cost_range=f"{sub['cost'].min():.1f}–{sub['cost'].max():.1f}",
            dominated_at_any_kh=not dominated,
            best_at_any_L=any(
                lt_agg[lt_agg.lead_time == L].sort_values("cost").iloc[0].arm == arm
                for L in LEAD_DAYS_GRID),
            training_required=arm in ("gbdt-lean", "chronos2-ft-full"),
            feature_eng_required=arm == "gbdt-lean",
        ))
    guide_df = pd.DataFrame(guide_rows)
    guide_df.to_csv(ART / "practical_guide.csv", index=False)
    print(guide_df.to_string(index=False))

    # ================================================================
    # Rank flip check
    # ================================================================
    print("\n=== 排序翻转检查 ===")
    flip_count = 0
    # Compare TSFM vs emp-daily rank across all κ_h
    for kh in KAPPA_H_GRID:
        sub = kh_agg[kh_agg.kappa_h == kh].set_index("arm")
        if "chronos2-zs" in sub.index and "emp-daily" in sub.index:
            if sub.loc["chronos2-zs", "cost"] > sub.loc["emp-daily", "cost"]:
                flip_count += 1
                print(f"  ⚠ κ_h={kh}: chronos2-zs 成本高于 emp-daily (翻转!)")
    # Compare TSFM vs emp-daily rank across all L
    for L in LEAD_DAYS_GRID:
        sub = lt_agg[lt_agg.lead_time == L].set_index("arm")
        if "chronos2-zs" in sub.index and "emp-daily" in sub.index:
            if sub.loc["chronos2-zs", "cost"] > sub.loc["emp-daily", "cost"]:
                flip_count += 1
                print(f"  ⚠ L={L}: chronos2-zs 成本高于 emp-daily (翻转!)")

    if flip_count == 0:
        print("  ✓ TSFM 在所有 κ_h 和 L 下均优于 emp-daily — 无排序翻转")
    chk.note("rank_flip_count", flip_count)

    # Compare GBDT vs emp-daily
    gbdt_flips = 0
    for kh in KAPPA_H_GRID:
        sub = kh_agg[kh_agg.kappa_h == kh].set_index("arm")
        if "gbdt-lean" in sub.index and "emp-daily" in sub.index:
            if sub.loc["gbdt-lean", "cost"] < sub.loc["emp-daily", "cost"]:
                gbdt_flips += 1
    print(f"  GBDT vs emp-daily: {'翻转(GBDT更优)' if gbdt_flips > 0 else '无翻转(GBDT始终更差)'} "
          f"({gbdt_flips}/{len(KAPPA_H_GRID)} κ_h 点)")
    chk.note("gbdt_flip_count", gbdt_flips)

    chk.n_rows = len(kh_df) + len(lt_df) + len(annual_df)
    chk.note("wall_clock_sec", round(time.time() - t0, 1))
    chk.finish(ART / "checks" / "ws4.json")
    print(f"\n总耗时 {time.time() - t0:.0f}s, 状态: {chk.status}")
    return chk.exit_code


if __name__ == "__main__":
    sys.exit(main())
