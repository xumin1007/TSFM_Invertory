"""实验 B：毛利口径下的 κ_h 敏感性 —— 修掉代数恒等式问题。

问题：`run_ws4_economics.py` 的 κ_h 扫描用 `costs_from_alpha`，其中
    h = κ_h · unit_cost / 12,   p = h · α/(1-α)
故改变 κ_h 会**同比例缩放 h 和 p**，临界比 α = p/(p+h) 恒定不变。又因订至
水平 S 用固定的 ALPHA_PRIMARY 计算、不随 κ_h 变，总成本对每个臂都严格线性
缩放，百分比排名在代数上不可能改变。原 Figure 2(b) 因此不是稳健性证据，
而是恒等式。

修法：改用 `costs_from_margin`，p = 实际单位毛利（由售价与进价算出，**不随
κ_h 变**），只有 h 随 κ_h 变。于是
    α_implied,i = p_i / (p_i + h_i)
真正随 κ_h 移动，且逐 SKU 异质。S 必须在这个逐行 α 上重算 —— 用
`pmf_quantile_rowwise`。这样成本曲线才携带真实信息。

同时输出隐含临界比随 κ_h 的移动，供论文正文引用。

用法:  PYTHONPATH=src python -m f2d.run_ws4_kappa_margin [--n-series 2000]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from . import config as cfgmod
from .aggregation import convolve_varying_pmf, pmf_quantile_rowwise
from .checks import CheckResult
from .conventions import SEED_BASE
from .datasets import zhao
from .decision import costs_from_alpha, costs_from_margin, layer_b
from .models.chronos import BASE_CHECKPOINT, NATIVE_LEVELS, QuantileRepair, to_grid
from .models.gbdt_grid import make_lean_features

ART = cfgmod.ARTIFACT_DIR / "zhao_kappa_margin"
VMAX = 60
LEAD_DAYS = 1
ALPHA_DECLARED = 0.85
KAPPA_H_GRID = (0.10, 0.15, 0.20, 0.25, 0.30)
VALID_MONTHS = [pd.Timestamp("2019-07-01"), pd.Timestamp("2019-08-01")]

ARMS = ("chronos2-ft-full", "chronos2-zs", "emp-daily", "emp-ewm")
EWM_HALFLIFE = 30.0
FT_CKPT = cfgmod.ARTIFACT_DIR / "zhao_finetune" / "chronos2-ft-full" / "ft"


def _daily_grids(arm, sids, ctx, n_days, pipe, ft_pipe, batch_size):
    if arm == "emp-daily":
        emp = np.array([np.quantile(ctx[s], NATIVE_LEVELS, method="inverted_cdf")
                        for s in sids], float)
        emp, _ = QuantileRepair()(emp)
        return [emp] * n_days

    if arm in ("chronos2-zs", "chronos2-ft-full"):
        import torch
        p = pipe if arm == "chronos2-zs" else ft_pipe
        q, _ = p.predict_quantiles(
            [torch.tensor(ctx[s], dtype=torch.float32) for s in sids],
            prediction_length=n_days, quantile_levels=list(NATIVE_LEVELS),
            batch_size=batch_size)
        g, _ = QuantileRepair()(to_grid(q))
        g = g.reshape(len(sids), n_days, -1)
        return [g[:, i, :] for i in range(n_days)]

    if arm == "emp-ewm":
        rows = []
        for s in sids:
            h = ctx[s]
            age = np.arange(h.size - 1, -1, -1, dtype=float)
            wt = 0.5 ** (age / EWM_HALFLIFE)
            order = np.argsort(h, kind="stable")
            xs, ws = h[order], wt[order]
            cw = np.cumsum(ws)
            cw = cw / cw[-1] if cw[-1] > 0 else cw
            idx = np.searchsorted(cw, NATIVE_LEVELS, side="left")
            rows.append(xs[np.clip(idx, 0, xs.size - 1)])
        g, _ = QuantileRepair()(np.asarray(rows, float))
        return [g] * n_days

    raise ValueError(arm)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=2000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args(argv)

    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    chk = CheckResult(step_id="zhao_kappa_margin", dataset="zhao",
                      seed=SEED_BASE)

    raw = zhao.load_raw()
    daily, _ = zhao.build_daily_panel(raw)
    panel, _ = zhao.build_panel(raw)
    margin_block = zhao.build_margin_block(raw, VALID_MONTHS).set_index(
        ["sku_ID", "month"])

    rng = np.random.default_rng(SEED_BASE)
    pool = np.asarray(sorted(set(daily.series_id)))
    keep = rng.choice(pool, size=min(args.n_series, len(pool)), replace=False)
    daily = daily[daily.series_id.isin(keep)]
    sku_of = dict(zip(daily.series_id, daily.sku_ID))
    make_lean_features(daily)

    from chronos import BaseChronosPipeline
    print(f"加载 Chronos-2 ({BASE_CHECKPOINT}) …")
    pipe = BaseChronosPipeline.from_pretrained(BASE_CHECKPOINT,
                                               device_map=args.device)
    ft_pipe = BaseChronosPipeline.from_pretrained(str(FT_CKPT),
                                                  device_map=args.device)

    rows = []
    alpha_rows = []

    for month in VALID_MONTHS:
        n_days = month.days_in_month + LEAD_DAYS
        snap = panel[panel.month == month].set_index("sku_ID")
        hist = daily[daily.d < month]
        ctx = {s: g.y.to_numpy() for s, g in hist.groupby("series_id")}
        sids = np.array([s for s in sorted(ctx)
                         if len(ctx[s]) >= 30 and sku_of[s] in snap.index])
        sk = np.array([sku_of[s] for s in sids])
        cur = snap.loc[sk]

        cost_i = cur["unit_cost_hist"].to_numpy(float)
        ip = cur["beginning_inventory"].to_numpy(float)
        y = cur["observed_sales_next_month"].to_numpy(float)

        mg = margin_block.reindex(pd.MultiIndex.from_arrays(
            [sk, np.repeat(month, len(sk))]))
        margin_i = mg["margin_unit"].to_numpy(float)

        usable = np.isfinite(cost_i) & np.isfinite(margin_i) & (margin_i > 0)
        print(f"\n{month:%Y-%m}  序列 {len(sids)}  "
              f"毛利可用 {int(usable.sum())}")

        grids_cache = {}
        for arm in ARMS:
            grids_cache[arm] = _daily_grids(arm, sids, ctx, n_days, pipe,
                                            ft_pipe, args.batch_size)
            print(f"  预测缓存 {arm}  ({time.time() - t0:.0f}s)")

        for kh in KAPPA_H_GRID:
            # 毛利口径：p 固定（数据），h 随 κ_h 变 → 临界比真正移动
            h_m, p_m = costs_from_margin(cost_i, margin_i, kh, 12)
            alpha_implied = p_m / np.clip(p_m + h_m, 1e-12, None)

            # 声明口径（原做法），用于对照
            h_a, p_a = costs_from_alpha(cost_i, ALPHA_DECLARED, kh, 12)

            au = alpha_implied[usable]
            alpha_rows.append(dict(
                month=month.date(), kappa_h=kh,
                alpha_implied_p50=float(np.median(au)),
                alpha_implied_p10=float(np.quantile(au, 0.10)),
                alpha_implied_p90=float(np.quantile(au, 0.90)),
                alpha_declared=ALPHA_DECLARED))

            for arm in ARMS:
                grids = grids_cache[arm]
                pmf_pi = convolve_varying_pmf(NATIVE_LEVELS, grids, vmax=VMAX)

                # 毛利口径：逐 SKU 的临界比 → S 随 κ_h 真实移动
                S_margin = pmf_quantile_rowwise(pmf_pi, alpha_implied)
                r_m = layer_b(S_margin, ip, y, h_m, p_m)

                # 声明口径：S 不随 κ_h 变（恒等式对照）
                S_decl = pmf_quantile_rowwise(
                    pmf_pi, np.full(len(sids), ALPHA_DECLARED))
                r_a = layer_b(S_decl, ip, y, h_a, p_a)

                rows.append(dict(
                    month=month.date(), kappa_h=kh, arm=arm,
                    cost_margin=float(np.mean(r_m.cost[usable])),
                    cost_declared=float(np.mean(r_a.cost[usable])),
                    S_med_margin=float(np.median(S_margin[usable])),
                    S_med_declared=float(np.median(S_decl[usable])),
                    CSR_ub_margin=float(r_m.csr_upper_bound),
                    n=int(usable.sum())))

    df = pd.DataFrame(rows)
    agg = df.groupby(["kappa_h", "arm"])[
        ["cost_margin", "cost_declared", "S_med_margin", "S_med_declared",
         "CSR_ub_margin"]].mean().reset_index()
    agg.to_csv(ART / "sensitivity_kappa_margin.csv", index=False)

    adf = pd.DataFrame(alpha_rows)
    a_agg = adf.groupby("kappa_h")[
        ["alpha_implied_p10", "alpha_implied_p50",
         "alpha_implied_p90"]].mean().reset_index()
    a_agg.to_csv(ART / "implied_alpha_by_kappa.csv", index=False)

    # ---- 报告 ----
    print("\n" + "=" * 72)
    print("隐含临界比随 κ_h 的移动（毛利口径）")
    print("=" * 72)
    print(f"{'κ_h':>6}  {'α_p10':>7} {'α_p50':>7} {'α_p90':>7}")
    for _, r in a_agg.iterrows():
        print(f"{r.kappa_h:6.2f}  {r.alpha_implied_p10:7.4f} "
              f"{r.alpha_implied_p50:7.4f} {r.alpha_implied_p90:7.4f}")

    for label, col in [("毛利口径（p 由数据定，α 随 κ_h 移动）", "cost_margin"),
                       ("声明口径（原做法，代数恒等）", "cost_declared")]:
        print("\n" + "=" * 72)
        print(label)
        print("=" * 72)
        print(f"{'arm':<20}", end="")
        for kh in KAPPA_H_GRID:
            print(f"{kh:>9.2f}", end="")
        print("   线性?")
        for arm in ARMS:
            sub = agg[agg.arm == arm].set_index("kappa_h")
            print(f"{arm:<20}", end="")
            vals = []
            for kh in KAPPA_H_GRID:
                v = sub.loc[kh, col]
                vals.append(v)
                print(f"{v:>9.2f}", end="")
            # 线性检验：与 κ_h 成正比则 cost/κ_h 恒定
            ratios = np.array(vals) / np.array(KAPPA_H_GRID)
            spread = (ratios.max() - ratios.min()) / ratios.mean()
            print(f"   {'是' if spread < 0.01 else f'否({spread:.1%})'}")

        # 相对经验基线的百分比是否随 κ_h 变化
        print(f"\n{'arm':<20} 相对 emp-daily 的 Δ%")
        base = agg[agg.arm == "emp-daily"].set_index("kappa_h")[col]
        for arm in ARMS:
            if arm == "emp-daily":
                continue
            sub = agg[agg.arm == arm].set_index("kappa_h")[col]
            print(f"{arm:<20}", end="")
            pcts = []
            for kh in KAPPA_H_GRID:
                pct = (sub.loc[kh] - base.loc[kh]) / base.loc[kh] * 100
                pcts.append(pct)
                print(f"{pct:>+8.2f}%", end="")
            print(f"   变动幅度 {max(pcts) - min(pcts):.2f}pp")

    lo, hi = a_agg.alpha_implied_p50.min(), a_agg.alpha_implied_p50.max()
    chk.note("alpha_implied_p50_range", [round(float(lo), 4), round(float(hi), 4)])
    chk.note("alpha_declared", ALPHA_DECLARED)
    chk.n_rows = len(agg)
    chk.finish(ART / "checks")

    print(f"\n隐含临界比中位数随 κ_h 从 {hi:.4f} 移到 {lo:.4f}"
          f"（κ_h {KAPPA_H_GRID[0]}→{KAPPA_H_GRID[-1]}）")
    print(f"结果已保存: {ART}")
    print(f"总耗时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
