"""实验 A：rolling / EW empirical baseline —— 机制叙事的关键对照。

动机：`run_zhao_layerc.py` 的 emp-daily 用 `hist = daily[daily.d < month]`，即
**全部**历史的经验分位数，而 Chronos-2 的有效 context 有界（WS-6 显示 30 天即饱和）。
这构成不对称：全历史分位数对 level shift 反应慢，而这恰是我们归因给"预训练先验"
的那部分优势。

若 30/60 天 rolling empirical 就能逼近 Chronos，则第 7 节机制叙事需整体重写：
真正的驱动不是 pretrained priors，而只是"最近窗口 > 全历史"。

对照臂：
  emp-daily      全历史（现有基线）
  emp-roll30/60/90  仅最近 W 天
  emp-ewm        指数加权（半衰期 30 天）
  chronos2-zs / chronos2-ft-full  参照

用法:  PYTHONPATH=src python -m f2d.run_zhao_rolling_baseline [--n-series 2000]
"""

from __future__ import annotations

import argparse
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
from .uncertainty import paired_bootstrap_mean

ART = cfgmod.ARTIFACT_DIR / "zhao_rolling"
VMAX = 60
LEAD_DAYS = 1
KAPPA_H = 0.20
WINDOWS = {
    "validation": [pd.Timestamp("2019-07-01"), pd.Timestamp("2019-08-01")],
    "test": [pd.Timestamp("2019-09-01"), pd.Timestamp("2019-10-01")],
}

ROLL_WINDOWS = (30, 60, 90)
EWM_HALFLIFE = 30.0

ARMS = (["emp-daily"]
        + [f"emp-roll{w}" for w in ROLL_WINDOWS]
        + ["emp-ewm", "chronos2-zs", "chronos2-ft-full"])

FT_CKPT = cfgmod.ARTIFACT_DIR / "zhao_finetune" / "chronos2-ft-full" / "ft"


def _weighted_quantiles(x: np.ndarray, w: np.ndarray,
                        levels: np.ndarray) -> np.ndarray:
    """加权经验分位数。x 升序排列后按累计权重取 inverted_cdf。"""
    order = np.argsort(x, kind="stable")
    xs, ws = x[order], w[order]
    cw = np.cumsum(ws)
    if cw[-1] <= 0:
        return np.zeros(levels.size)
    cw = cw / cw[-1]
    idx = np.searchsorted(cw, levels, side="left")
    return xs[np.clip(idx, 0, xs.size - 1)]


def _empirical_grid(arm: str, sids, ctx) -> np.ndarray:
    """按 arm 指定的加权方案算每序列的 21 个分位数。"""
    rows = []
    for s in sids:
        h = ctx[s]
        if arm == "emp-daily":
            q = np.quantile(h, NATIVE_LEVELS, method="inverted_cdf")
        elif arm.startswith("emp-roll"):
            w = int(arm.removeprefix("emp-roll"))
            tail = h[-w:] if h.size > w else h
            q = np.quantile(tail, NATIVE_LEVELS, method="inverted_cdf")
        elif arm == "emp-ewm":
            age = np.arange(h.size - 1, -1, -1, dtype=float)
            wt = 0.5 ** (age / EWM_HALFLIFE)
            q = _weighted_quantiles(h, wt, NATIVE_LEVELS)
        else:
            raise ValueError(arm)
        rows.append(q)
    g, _ = QuantileRepair()(np.asarray(rows, float))
    return g


def _daily_grids(arm, sids, ctx, n_days, pipe, ft_pipe, batch_size):
    if arm.startswith("emp-"):
        return [_empirical_grid(arm, sids, ctx)] * n_days

    import torch
    p = pipe if arm == "chronos2-zs" else ft_pipe
    q, _ = p.predict_quantiles(
        [torch.tensor(ctx[s], dtype=torch.float32) for s in sids],
        prediction_length=n_days, quantile_levels=list(NATIVE_LEVELS),
        batch_size=batch_size)
    g, _ = QuantileRepair()(to_grid(q))
    g = g.reshape(len(sids), n_days, -1)
    return [g[:, i, :] for i in range(n_days)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=2000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--split", choices=("validation", "test"),
                    default="validation")
    args = ap.parse_args(argv)
    ALPHA_PRIMARY = args.alpha
    MONTHS = WINDOWS[args.split]

    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    chk = CheckResult(step_id="zhao_rolling_baseline", dataset="zhao",
                      seed=SEED_BASE)

    raw = zhao.load_raw()
    daily, _ = zhao.build_daily_panel(raw)
    panel, _ = zhao.build_panel(raw)

    rng = np.random.default_rng(SEED_BASE)
    pool = np.asarray(sorted(set(daily.series_id)))
    keep = rng.choice(pool, size=min(args.n_series, len(pool)), replace=False)
    daily = daily[daily.series_id.isin(keep)]
    sku_of = dict(zip(daily.series_id, daily.sku_ID))
    make_lean_features(daily)  # 与主脚本保持同样的序列过滤副作用

    from chronos import BaseChronosPipeline
    print(f"加载 Chronos-2 ({BASE_CHECKPOINT}) …")
    pipe = BaseChronosPipeline.from_pretrained(BASE_CHECKPOINT,
                                               device_map=args.device)
    ft_pipe = BaseChronosPipeline.from_pretrained(str(FT_CKPT),
                                                  device_map=args.device)

    rows = []
    per_series_cost = {a: [] for a in ARMS}

    for month in MONTHS:
        n_days = month.days_in_month + LEAD_DAYS
        snap = panel[panel.month == month].set_index("sku_ID")
        hist = daily[daily.d < month]
        ctx = {s: g.y.to_numpy() for s, g in hist.groupby("series_id")}
        sids = np.array([s for s in sorted(ctx)
                         if len(ctx[s]) >= 30 and sku_of[s] in snap.index])
        sk = np.array([sku_of[s] for s in sids])
        cur = snap.loc[sk]

        ctx_len = np.array([len(ctx[s]) for s in sids])
        print(f"\n{month:%Y-%m}  序列 {len(sids)}  "
              f"历史长度 中位={np.median(ctx_len):.0f} "
              f"min={ctx_len.min()} max={ctx_len.max()}")

        initial_inv = cur["beginning_inventory"].to_numpy(float)
        cost_i = cur["unit_cost_hist"].to_numpy(float)

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
            grids = _daily_grids(arm, sids, ctx, n_days, pipe, ft_pipe,
                                 args.batch_size)
            pmf_pi = convolve_varying_pmf(NATIVE_LEVELS, grids, vmax=VMAX)
            pmf_r = convolve_varying_pmf(NATIVE_LEVELS,
                                         grids[:month.days_in_month], vmax=VMAX)
            m_ratio = n_days / month.days_in_month

            S_dict = order_up_to(pmf_r, pmf_pi, ALPHA_PRIMARY, m_ratio)
            S_val = S_dict["P3"]   # 保护期真实分位数（与论文主结果一致）

            rc = ReplayConfig(n_days=n_days, lead_time_days=LEAD_DAYS,
                              review_cadence_days=n_days,
                              shortage_mechanism="lost_sales")
            res = replay(demand_matrix, S_val[:, None], initial_inv, rc)

            h_c, p_c = costs_from_alpha(cost_i, ALPHA_PRIMARY, KAPPA_H, 12)
            avg_iend = res.i_end.mean(axis=1)
            total_short = res.lost.sum(axis=1)
            cost_series = h_c * avg_iend + p_c * total_short

            ok = np.isfinite(cost_series)
            per_series_cost[arm].append(
                pd.Series(cost_series[ok], index=sids[ok]))

            csr = float((total_short == 0).mean())
            fr = float(1.0 - total_short.sum() / max(res.demand.sum(), 1e-9))
            rows.append(dict(month=month.date(), arm=arm,
                             n=int(ok.sum()),
                             cost=float(np.mean(cost_series[ok])),
                             CSR_ub=csr, FR_ub=fr,
                             S_med=float(np.median(S_val))))
            print(f"  {arm:22s} cost={rows[-1]['cost']:.4f} "
                  f"CSR={csr:.4f} FR={fr:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(ART / f"rolling_monthly_{args.split}_a{int(ALPHA_PRIMARY*100)}.csv", index=False)

    # ---- 汇总 + 配对 bootstrap（以 emp-daily 为基准）----
    print("\n" + "=" * 64)
    print(f"汇总（{args.split} 两月合并，α={ALPHA_PRIMARY}）  Δ 相对 emp-daily")
    print("=" * 64)

    pooled = {a: pd.concat(v) for a, v in per_series_cost.items()}

    # 只保留在所有臂上都有成本的序列，保持配对完整
    common = None
    for s in pooled.values():
        idx = s.index.unique()
        common = idx if common is None else common.intersection(idx)

    long_rows = []
    for arm, month_series in per_series_cost.items():
        for mi, s in enumerate(month_series):
            sub = s.loc[s.index.isin(common)]
            for sid, c in sub.items():
                long_rows.append({"variant": arm, "series_id": sid,
                                  "month": mi, "cost": c})
    long_df = pd.DataFrame(long_rows)

    cis = paired_bootstrap_mean(
        long_df, value_col="cost", baseline="emp-daily",
        variants=[a for a in ARMS if a != "emp-daily"],
        variant_col="variant", series_col="series_id",
        b=10_000, ci=0.95, seed=SEED_BASE)
    ci_by_arm = {c.variant: c for c in cis}

    boot_rows = []
    for arm in ARMS:
        mean_cost = float(pooled[arm].loc[pooled[arm].index.isin(common)].mean())
        if arm == "emp-daily":
            delta, lo, hi, sig = 0.0, 0.0, 0.0, False
        else:
            c = ci_by_arm[arm]
            delta, lo, hi = c.delta, c.lo, c.hi
            sig = not (lo <= 0.0 <= hi)
        boot_rows.append(dict(arm=arm, cost=round(mean_cost, 4),
                              delta_cost=round(delta, 6),
                              ci_lo=round(lo, 6), ci_hi=round(hi, 6),
                              significant=sig, n_series=len(common)))
        star = "***" if sig else "n.s."
        print(f"  {arm:22s} cost={mean_cost:.4f}  "
              f"Δ={delta:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {star}")

    # 逐 (SKU, 月) 成本导出（供 run_frozen_baseline_test.py 做直接配对检验）
    # 用复合键，避免同一 series_id 在两个月上索引重复导致对齐错乱
    ps_parts = []
    for a, month_series in per_series_cost.items():
        for mi, sr in enumerate(month_series):
            ps_parts.append(pd.DataFrame({
                "series_id": sr.index, "month_idx": mi, "arm": a,
                "cost": sr.to_numpy()}))
    ps = (pd.concat(ps_parts, ignore_index=True)
            .pivot_table(index=["series_id", "month_idx"],
                         columns="arm", values="cost"))
    ps.columns.name = None
    ps.reset_index().to_csv(
        ART / f"rolling_perseries_{args.split}_a{int(ALPHA_PRIMARY*100)}.csv",
        index=False)

    bdf = pd.DataFrame(boot_rows)
    bdf.to_csv(ART / f"rolling_bootstrap_{args.split}_a{int(ALPHA_PRIMARY*100)}.csv", index=False)

    # ---- 关键判定 ----
    print("\n" + "=" * 64)
    best_emp = bdf[bdf.arm.str.startswith("emp-")].sort_values("cost").iloc[0]
    chr_zs = bdf[bdf.arm == "chronos2-zs"].iloc[0]
    chr_ft = bdf[bdf.arm == "chronos2-ft-full"].iloc[0]
    gap_zs = (chr_zs.cost - best_emp.cost) / best_emp.cost * 100
    gap_ft = (chr_ft.cost - best_emp.cost) / best_emp.cost * 100
    print(f"最佳经验臂:      {best_emp.arm} (cost={best_emp.cost:.4f})")
    print(f"Chronos-2 ZS:    {chr_zs.cost:.4f}  ({gap_zs:+.1f}% vs 最佳经验)")
    print(f"Chronos-2 FT:    {chr_ft.cost:.4f}  ({gap_ft:+.1f}% vs 最佳经验)")
    print()
    if best_emp.arm != "emp-daily":
        shrink = ((chr_ft.cost - pooled['emp-daily'].mean())
                  / (chr_ft.cost - best_emp.cost)) if chr_ft.cost != best_emp.cost else np.inf
        print(f"⚠ 最佳经验臂不是全历史 —— rolling 窗口确实改善了基线。")
        print(f"  机制叙事需重新表述：优势的一部分来自'最近窗口 > 全历史'，")
        print(f"  而非纯粹的 pretrained priors。")
    else:
        print("✓ 全历史仍是最佳经验臂 —— rolling 窗口未能缩小差距，")
        print("  '预训练先验'的归因得到支持。")

    chk.note("best_empirical_arm", str(best_emp.arm))
    chk.note("best_empirical_cost", float(best_emp.cost))
    chk.note("chronos_ft_cost", float(chr_ft.cost))
    chk.note("gap_ft_vs_best_emp_pct", round(float(gap_ft), 2))
    chk.n_rows = len(bdf)
    chk.finish(ART / f"checks_{args.split}_a{int(ALPHA_PRIMARY*100)}")

    print(f"\n结果已保存: {ART}")
    print(f"总耗时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
