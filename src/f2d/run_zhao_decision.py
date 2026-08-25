"""层 B：Zhao 单期决策代理。对应 docs/07_decision_layer.md §6。

回答项目的核心问题：**预测层 0.049 的 NPL 优势能否转成服务率/成本上的差异，
还是被 newsvendor 的临界比吞掉。**

结构：每个 (sku, 月) 是一个独立补货周期，从观测快照重置。

  IP        = beginning_inventory + on_order_inventory      （月初快照，不推演）
  S         = policy(保护期需求 PMF, alpha)                  （§4，P1/P2/P3）
  position  = max(IP, S)                                    （§3.1 实测支持）
  y         = 当月实际销量（截断下界）

保护期 PI = R + L = 当月天数 + 1 天（`normal_lead_time` 中位 1 天，§3.1）。
$m = \\mathrm{PI}/R \\approx 1.03$，故 P1/P2/P3 在 Zhao 上预期差异极小 ——
这是 §3.1 预言的结果，不是实现缺陷。**Zhao 上变化的主轴是预测臂而非策略。**

预测臂全部走与预测层同一条路径（日级 21 点网格 -> 逐日卷积 -> PI 分布），
故聚合误差对各臂等同（§2.3.3）。GBDT 按 h in [0, PI) 重训 —— 预测层的模型
只训到 h=6，外推到 h=31 不合法。

用法:  PYTHONPATH=src python -m f2d.run_zhao_decision [--n-series 2000]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from . import config as cfgmod
from .aggregation import convolve_varying_pmf, pmf_quantile_rowwise
from .checks import CheckResult
from .conventions import SEED_BASE
from .datasets import zhao
from .decision import (POLICIES, costs_from_alpha, costs_from_margin,
                       implied_alpha, layer_b, order_up_to)
from .models.chronos import BASE_CHECKPOINT, NATIVE_LEVELS, QuantileRepair, to_grid
from .models.gbdt_grid import LEAN_FEATURES, QuantileGridGBDT, make_lean_features
from .uncertainty import paired_bootstrap_mean, report

ART = cfgmod.ARTIFACT_DIR / "zhao_decision"
VMAX = 60                     # 日级单期上界；PI≈32 天故卷积支撑到 1920
LEAD_DAYS = 1                 # §3.1 实测中位提前期
ALPHA_GRID = (0.85, 0.90, 0.95, 0.98)
ALPHA_PRIMARY = 0.95
KAPPA_H = 0.20
ARMS = ("chronos2-zs", "chronos2-ft-full", "chronos2-ft-full-h32",
        "chronos2-ft-full-short", "gbdt-lean", "emp-daily", "always-zero")
# 两个全参微调 checkpoint，唯一差别是微调时的 prediction_length。
# -h32 与决策视界一致，用于拆开「微调劣势 vs 视界失配」这一混淆。
FT_CKPTS = {a: cfgmod.ARTIFACT_DIR / "zhao_finetune" / a / "ft"
            for a in ("chronos2-ft-full", "chronos2-ft-full-h32",
                      "chronos2-ft-full-short")}

VALID_MONTHS = [pd.Timestamp("2019-07-01"), pd.Timestamp("2019-08-01")]
# GBDT 训练 origin：全部严格早于验证窗，且目标不越过 6-30
GBDT_TRAIN_ORIGINS = pd.date_range("2019-02-01", "2019-05-01", freq="MS")


def _daily_grids(arm, origin, sids, ctx, n_days, pipe, gbdt, feat, batch_size,
                 ft_pipes=None):
    """返回长度 n_days 的列表，每项 (n_series, 21) —— 该日的分位网格。"""
    if arm == "always-zero":
        return [np.zeros((len(sids), NATIVE_LEVELS.size)) for _ in range(n_days)]

    if arm == "emp-daily":
        emp = np.array([np.quantile(ctx[s], NATIVE_LEVELS, method="inverted_cdf")
                        for s in sids], float)
        emp, _ = QuantileRepair()(emp)
        return [emp] * n_days                      # 静态分布，逐日相同

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
        base = feat.set_index(["series_id", "d"]).reindex(
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
    chk = CheckResult(step_id="zhao_layer_b", dataset="zhao", seed=SEED_BASE)

    raw = zhao.load_raw()
    daily, _ = zhao.build_daily_panel(raw)
    panel, _ = zhao.build_panel(raw)

    rng = np.random.default_rng(SEED_BASE)
    pool = np.asarray(sorted(set(daily.series_id)))
    keep = rng.choice(pool, size=min(args.n_series, len(pool)), replace=False)
    daily = daily[daily.series_id.isin(keep)]
    sku_of = dict(zip(daily.series_id, daily.sku_ID))
    feat = make_lean_features(daily)

    # 提前期越界标记（§6.1）：normal_lead_time > 7 的 SKU 不并入主结论
    lt = raw["orders"].groupby("sku_ID")["normal_lead_time"].median()
    lt_exceeds = set(lt[lt > 7].index)
    chk.note("lead_time_exceeds_period_skus", len(lt_exceeds))

    # --- GBDT：按 h in [0, 32) 重训，训练 origin 全部早于验证窗 ---
    max_h = max(m.days_in_month for m in VALID_MONTHS) + LEAD_DAYS
    fidx = feat.set_index(["series_id", "d"])
    tr = []
    for o in GBDT_TRAIN_ORIGINS:
        sids = np.asarray(sorted(set(
            daily[daily.d < o - pd.Timedelta(days=30)].series_id)))
        b = fidx.reindex(pd.MultiIndex.from_product([sids, [o]]))[LEAN_FEATURES].dropna(how="all")
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
    print(f"GBDT 训练 {len(train)} 行, h in [0,{max_h}), 末目标日 {last_target:%Y-%m-%d}")
    gbdt = QuantileGridGBDT(features=LEAN_FEATURES + ["h"]).fit(train)
    print(f"GBDT 就绪 ({time.time() - t0:.0f}s)")

    # torch 必须在 LightGBM 训练**之后**导入：二者各自加载 OpenMP 运行时，
    # 在 macOS 上先导 torch 再 fit LightGBM 会直接段错误（实测 exit 139）。
    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        BASE_CHECKPOINT, device_map=args.device, torch_dtype=torch.float32)
    ft_pipes = {a: BaseChronosPipeline.from_pretrained(
        str(c), device_map=args.device, torch_dtype=torch.float32)
        for a, c in FT_CKPTS.items()}

    margin = zhao.build_margin_block(raw, VALID_MONTHS).set_index(["sku_ID", "month"])
    chk.note("margin_price_level", margin.price_level.value_counts(
        normalize=True).round(4).to_dict())
    chk.note("margin_cost_level", margin.cost_level.value_counts(
        normalize=True).round(4).to_dict())

    rows = []
    for month in VALID_MONTHS:
        n_days = month.days_in_month + LEAD_DAYS          # PI = R + L
        snap = panel[panel.month == month].set_index("sku_ID")
        hist = daily[daily.d < month]
        ctx = {s: g.y.to_numpy() for s, g in hist.groupby("series_id")}
        sids = np.array([s for s in sorted(ctx)
                         if len(ctx[s]) >= 30 and sku_of[s] in snap.index])
        sk = [sku_of[s] for s in sids]
        cur = snap.loc[sk]

        ip = (cur["beginning_inventory"].to_numpy(float)
              + cur["on_order_inventory"].to_numpy(float))
        y = cur["observed_sales_next_month"].to_numpy(float)
        cost_i = cur["unit_cost_hist"].to_numpy(float)
        mg = margin.reindex(pd.MultiIndex.from_arrays(
            [sk, np.repeat(month, len(sk))]))
        margin_i = mg["margin_unit"].to_numpy(float)
        # 毛利口径下的逐 SKU 临界比。h 仍用声明的 kappa_h，故 alpha_i 只是
        # **部分**由数据支持；unit_cost 缺失的行拿不到 alpha_i，置 NaN 后剔除。
        h_m, p_m = costs_from_margin(cost_i, margin_i, KAPPA_H, 12)
        alpha_i = implied_alpha(h_m, p_m)

        for arm in ARMS:
            grids = _daily_grids(arm, month, sids, ctx, n_days, pipe, gbdt,
                                 feat, args.batch_size, ft_pipes)
            pmf_pi = convolve_varying_pmf(NATIVE_LEVELS, grids, vmax=VMAX)
            pmf_r = convolve_varying_pmf(NATIVE_LEVELS, grids[:month.days_in_month],
                                         vmax=VMAX)
            m_ratio = n_days / month.days_in_month
            common = dict(sku_ID=sk, series_id=sids, month=month, ip=ip, y=y,
                          unit_cost=cost_i, margin_unit=margin_i,
                          lt_exceeds=[k in lt_exceeds for k in sk])
            for alpha in ALPHA_GRID:
                S = order_up_to(pmf_r, pmf_pi, alpha, m_ratio)
                for pol in POLICIES:
                    rows.append(pd.DataFrame(
                        {"arm": arm, "policy": pol, "alpha": alpha,
                         "S": S[pol], **common}))
            # P3m：订至水平取**逐 SKU** 的毛利临界比，而非全局声明的 alpha。
            # 这是唯一一个服务目标由数据定的策略，登记为独立策略而非 P3 的变体。
            rows.append(pd.DataFrame(
                {"arm": arm, "policy": "P3m", "alpha": np.nan,
                 "S": pmf_quantile_rowwise(pmf_pi, np.nan_to_num(alpha_i, nan=0.0)),
                 "alpha_i": alpha_i, **common}))
            print(f"  {month:%Y-%m} {arm:<12} n={len(sids)} PI={n_days}d "
                  f"({time.time() - t0:.0f}s)")

    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(ART / "layer_b_validation.parquet", index=False)
    chk.n_rows = len(df)

    # --- 评价（§8）。成本只在 unit_cost 可得的行上算，覆盖率必报 ---
    has_c = df.unit_cost.notna() & df.lt_exceeds.eq(False)
    chk.note("unit_cost_coverage", round(float(df.unit_cost.notna().mean()), 4))
    chk.note("lt_exceeds_share", round(float(df.lt_exceeds.mean()), 4))
    u = df[["sku_ID", "month", "ip"]].drop_duplicates()
    chk.note("negative_ip_backorder_rows", int((u.ip < 0).sum()))

    # 两个成本口径并列（§5.3）：
    #   derived  p = h*alpha/(1-alpha)，alpha 事前声明，p 是**导出量**
    #   margin   p = 售价 - 进价，由数据算出，逐 SKU 异质
    # 二者对同一组 S 各评一次；S 本身只由策略决定，不随成本口径变。
    out = []
    for (arm, pol, alpha), g in df.groupby(["arm", "policy", "alpha"], dropna=False):
        gg = g[has_c.loc[g.index]]
        if not len(gg):
            continue
        a_eff = ALPHA_PRIMARY if pd.isna(alpha) else alpha
        pairs = {
            "derived": costs_from_alpha(gg.unit_cost.to_numpy(), a_eff, KAPPA_H, 12),
            "margin": costs_from_margin(gg.unit_cost.to_numpy(),
                                        gg.margin_unit.to_numpy(), KAPPA_H, 12),
        }
        for costing, (h_c, p_c) in pairs.items():
            r = layer_b(gg.S.to_numpy(), gg.ip.to_numpy(), gg.y.to_numpy(), h_c, p_c)
            out.append(dict(arm=arm, policy=pol, alpha=alpha, costing=costing,
                            n=len(gg),
                            CSR_ub=r.csr_upper_bound,
                            FR_ub=r.fill_rate_upper_bound,
                            cost=float(r.cost.mean()),
                            hold=float((h_c * r.i_end).mean()),
                            short_c=float(
                                (p_c * r.observed_shortage_lower_bound).mean()),
                            S_med=float(np.median(gg.S)), n_orders=r.n_orders,
                            avg_inv=r.avg_inventory,
                            lost_lb=r.lost_units_lower_bound))
    res = pd.DataFrame(out).sort_values(["costing", "alpha", "policy", "cost"])
    res.to_csv(ART / "layer_b_summary.csv", index=False)

    print("\n" + "=" * 78)
    print(f"主情景 alpha={ALPHA_PRIMARY}, kappa_h={KAPPA_H}, p 由临界比导出 (=19h)")
    m = res[(res.alpha == ALPHA_PRIMARY) & (res.costing == "derived")]
    print(m[["policy", "arm", "n", "CSR_ub", "FR_ub", "cost", "hold", "short_c", "S_med"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n跨 alpha（仅 P3）")
    r3 = res[(res.policy == "P3") & (res.costing == "derived")]
    p3 = r3.pivot(index="arm", columns="alpha", values="cost")
    print(p3.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nCSR 上界 对照 alpha（可行性，§9.1；仅 P3。§10：真实值只会更低）")
    print(r3.pivot(index="arm", columns="alpha", values="CSR_ub")
          .to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n毛利口径（p = 售价 - 进价，逐 SKU）")
    mm = res[res.costing == "margin"]
    print(mm[(mm.alpha == ALPHA_PRIMARY) & (mm.policy == "P3")]
          [["arm", "CSR_ub", "FR_ub", "cost", "hold", "short_c"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nP3m：服务目标由逐 SKU 毛利临界比定（非事前声明），毛利口径计价")
    print(mm[mm.policy == "P3m"][["arm", "CSR_ub", "FR_ub", "cost", "S_med"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    ai = df[df.policy == "P3m"]["alpha_i"].dropna()
    print(f"\nalpha_i 分布: 中位 {ai.median():.4f}  "
          f"p05 {ai.quantile(.05):.4f}  p95 {ai.quantile(.95):.4f}  "
          f"(事前声明 {ALPHA_PRIMARY})")
    chk.note("alpha_i_median", round(float(ai.median()), 4))
    zero_m = float((df[df.policy == "P3m"]["margin_unit"] <= 0).mean())
    chk.note("zero_margin_share", round(zero_m, 4))
    print(f"margin<=0 的行占比 {zero_m:.4f} -> alpha_i=0 -> S=0（永不订货）")

    # --- 成本差的配对 bootstrap（按序列聚簇），主情景 P3 ---
    print(f"\n配对 bootstrap: 逐行成本差, P3, alpha={ALPHA_PRIMARY}（基准 = emp-daily）")
    cb = df[(df.policy == "P3") & (df.alpha == ALPHA_PRIMARY) & has_c].copy()
    pos = np.maximum(cb.ip.to_numpy(), cb.S.to_numpy())
    # 2x2 拆解：视界(7/32) x 训练数据截止(06-23/05-29)
    chk.note("ft_arms", {"chronos2-ft-full": "H=7, cutoff 06-23",
                         "chronos2-ft-full-short": "H=7, cutoff 05-29",
                         "chronos2-ft-full-h32": "H=32, cutoff 05-29"})
    chk.note("ft_finetune_horizons", {"chronos2-ft-full": 7,
                                      "chronos2-ft-full-h32": 32,
                                      "decision_H": int(max(
                                          m.days_in_month for m in VALID_MONTHS)
                                          + LEAD_DAYS)})
    for costing, (h_c, p_c) in (
            ("derived", costs_from_alpha(cb.unit_cost.to_numpy(), ALPHA_PRIMARY,
                                         KAPPA_H, 12)),
            ("margin", costs_from_margin(cb.unit_cost.to_numpy(),
                                         cb.margin_unit.to_numpy(), KAPPA_H, 12))):
        cb["row_cost"] = (h_c * np.clip(pos - cb.y, 0, None)
                          + p_c * np.clip(cb.y - pos, 0, None))
        cis = paired_bootstrap_mean(cb, "row_cost", "emp-daily",
                                    [a for a in ARMS if a != "emp-daily"],
                                    variant_col="arm")
        print(f"  -- {costing} 口径")
        print(report(cis)[["variant", "delta", "lo", "hi", "verdict"]]
              .round(4).to_string(index=False))
        for c in cis:
            chk.note(f"cost_delta_{costing}_{c.variant}",
                     [round(c.delta, 4), round(c.lo, 4), round(c.hi, 4),
                      c.significant])

    # --- §11 验收第 7 条：截断数据集的短缺列名合规 ---
    from .decision import LayerBResult as _LBR
    _fields = set(_LBR.__dataclass_fields__)
    chk.assert_true("§10 短缺列名合规",
                    "observed_shortage_lower_bound" in _fields
                    and "shortage" not in _fields and "short" not in _fields)
    chk.assert_true("§10 服务率名自带上界限定",
                    {"csr_upper_bound", "fill_rate_upper_bound"} <= _fields)

    # --- 可行性门（§9.1）：CSR 上界是否达到 alpha ---
    feas = res[(res.policy == "P3") & (res.alpha == ALPHA_PRIMARY)]
    for _, r in feas.iterrows():
        chk.note(f"feasible_{r.arm}", bool(r.CSR_ub >= ALPHA_PRIMARY))
    chk.note("infeasible_flag", bool(not (feas.CSR_ub >= ALPHA_PRIMARY).any()))

    chk.note("wall_clock_sec", round(time.time() - t0, 1))
    chk.finish(ART / "checks" / "layer_b.json")
    return chk.exit_code


if __name__ == "__main__":
    sys.exit(main())
