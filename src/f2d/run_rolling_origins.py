"""Rolling-origin 检验：服务水平依赖性的两个预设 contrast。

设计（每个 origin 月 t 独立）：
  1. 只用 t 之前的数据为每个 α 选择经验加权方案 → Emp-retuned（主比较对象）
  2. Emp-fixed：第一个 origin 选定后永久冻结，衡量参数迁移稳定性
  3. Emp-oracle：在 t 上事后取最优，**不可实现**，仅用于量化 selection regret
  4. Chronos-2 zero-shot 评价（无训练、无泄漏，故可用于全部 origin）
  5. 保存逐 series_id × month × alpha × arm 成本

两个**预先设定**的 contrast（不看结果再改分组）：
  H1 斜率      d_itα = β0 + β1·z_α + λ_t + ε，  z_α = Φ⁻¹(α)，检验 β1 < 0
  H2 尾部水平  d_high < 0，  high = {0.95, 0.98}，low = {0.80, 0.85, 0.90}

两者必须分开报告：β1 < 0 只证明"随尾部改善"，不证明"在尾部胜出"。

outcome 分两种口径，避免"斜率只是成本尺度随 α 上升"的批评：
  管理口径  相对成本差 (%)
  机制口径  scaled α-pinball 差（与决策同一目标，去掉成本尺度）

bootstrap 重采样单位 = series_id（抽中一个 SKU 保留其全部 month × α × arm 路径）。

用法:  PYTHONPATH=src python -m f2d.run_rolling_origins [--n-series 2000]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from . import config as cfgmod
from .aggregation import convolve_varying_pmf
from .checks import CheckResult
from .conventions import SEED_BASE
from .datasets import zhao
from .decision import costs_from_alpha, order_up_to
from .models.chronos import BASE_CHECKPOINT, NATIVE_LEVELS, QuantileRepair, to_grid
from .models.gbdt_grid import make_lean_features
from .simulation import ReplayConfig, replay

ART = cfgmod.ARTIFACT_DIR / "zhao_rolling_origins"
VMAX = 60
LEAD_DAYS = 1
KAPPA_H = 0.20
ALPHAS = (0.80, 0.85, 0.90, 0.95, 0.98)
HIGH_ALPHAS = (0.95, 0.98)
LOW_ALPHAS = (0.80, 0.85, 0.90)

# origin 月：需要 ≥30 天历史，故从 3 月起；Zhao 覆盖至 10 月
ORIGINS = [pd.Timestamp(f"2019-{m:02d}-01") for m in (3, 4, 5, 6, 7, 8, 9, 10)]

EMP_SCHEMES = ("emp-daily", "emp-roll30", "emp-roll60", "emp-roll90", "emp-ewm")
EWM_HALFLIFE = 30.0
B = 10_000


# ---------------------------------------------------------------- 经验分位数
def _emp_grid(scheme: str, sids, ctx) -> np.ndarray:
    rows = []
    for s in sids:
        h = ctx[s]
        if scheme == "emp-daily":
            q = np.quantile(h, NATIVE_LEVELS, method="inverted_cdf")
        elif scheme.startswith("emp-roll"):
            w = int(scheme.removeprefix("emp-roll"))
            tail = h[-w:] if h.size > w else h
            q = np.quantile(tail, NATIVE_LEVELS, method="inverted_cdf")
        elif scheme == "emp-ewm":
            age = np.arange(h.size - 1, -1, -1, dtype=float)
            wt = 0.5 ** (age / EWM_HALFLIFE)
            order = np.argsort(h, kind="stable")
            xs, ws = h[order], wt[order]
            cw = np.cumsum(ws)
            cw = cw / cw[-1] if cw[-1] > 0 else cw
            idx = np.searchsorted(cw, NATIVE_LEVELS, side="left")
            q = xs[np.clip(idx, 0, xs.size - 1)]
        else:
            raise ValueError(scheme)
        rows.append(q)
    g, _ = QuantileRepair()(np.asarray(rows, float))
    return g


def _eff_n(h_size: int, scheme: str) -> float:
    """有效样本量 n_eff = (Σw)²/Σw²。尾部方差机制的直接证据之一。"""
    if scheme == "emp-daily":
        return float(h_size)
    if scheme.startswith("emp-roll"):
        return float(min(h_size, int(scheme.removeprefix("emp-roll"))))
    age = np.arange(h_size - 1, -1, -1, dtype=float)
    w = 0.5 ** (age / EWM_HALFLIFE)
    return float(w.sum() ** 2 / (w ** 2).sum())


def _alpha_pinball(q_alpha: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """与决策同一目标的 α-pinball loss（newsvendor cost 的比例缩放版）。"""
    d = y - q_alpha
    return np.where(d >= 0, alpha * d, (alpha - 1.0) * d)


# ---------------------------------------------------------------- 主流程
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=2000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--b", type=int, default=B)
    args = ap.parse_args(argv)

    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    chk = CheckResult(step_id="zhao_rolling_origins", dataset="zhao",
                      seed=SEED_BASE)

    raw = zhao.load_raw()
    daily, _ = zhao.build_daily_panel(raw)
    panel, _ = zhao.build_panel(raw)

    rng = np.random.default_rng(SEED_BASE)
    pool = np.asarray(sorted(set(daily.series_id)))
    keep = rng.choice(pool, size=min(args.n_series, len(pool)), replace=False)
    daily = daily[daily.series_id.isin(keep)]
    sku_of = dict(zip(daily.series_id, daily.sku_ID))
    make_lean_features(daily)

    from chronos import BaseChronosPipeline
    from .models.chronos import BASE_REVISION
    print(f"加载 Chronos-2 ({BASE_CHECKPOINT}) …")
    pipe = BaseChronosPipeline.from_pretrained(BASE_CHECKPOINT,
                                               revision=BASE_REVISION,
                                               device_map=args.device)

    long_rows = []
    neff_rows = []

    for oi, month in enumerate(ORIGINS):
        n_days = month.days_in_month + LEAD_DAYS
        snap = panel[panel.month == month]
        if snap.empty:
            print(f"{month:%Y-%m}  跳过（panel 无该月）")
            continue
        snap = snap.set_index("sku_ID")
        hist = daily[daily.d < month]
        ctx = {s: g.y.to_numpy() for s, g in hist.groupby("series_id")}
        sids = np.array([s for s in sorted(ctx)
                         if len(ctx[s]) >= 30 and sku_of[s] in snap.index])
        if len(sids) < 50:
            print(f"{month:%Y-%m}  跳过（可用序列 {len(sids)} < 50）")
            continue
        sk = np.array([sku_of[s] for s in sids])
        cur = snap.loc[sk]

        initial_inv = cur["beginning_inventory"].to_numpy(float)
        cost_i = cur["unit_cost_hist"].to_numpy(float)

        demand_matrix = np.zeros((len(sids), n_days))
        md = daily[(daily.d >= month) &
                   (daily.d < month + pd.DateOffset(months=1)
                    + pd.Timedelta(days=LEAD_DAYS))]
        for i, s in enumerate(sids):
            sd = md[md.series_id == s].set_index("d")
            for k in range(n_days):
                d = month + pd.Timedelta(days=k)
                if d in sd.index:
                    demand_matrix[i, k] = float(sd.loc[d, "y"])
        y_pi = demand_matrix.sum(axis=1)   # 保护期实现需求（观测销量，截断下界）

        print(f"\n{month:%Y-%m}  序列 {len(sids)}  ({time.time()-t0:.0f}s)")

        # n_eff 记录（尾部方差证据）
        med_hist = int(np.median([len(ctx[s]) for s in sids]))
        for sch in EMP_SCHEMES:
            neff_rows.append(dict(month=month.date(), scheme=sch,
                                  median_hist_days=med_hist,
                                  n_eff=_eff_n(med_hist, sch)))

        # 所有臂的日分位网格
        grids = {sch: [_emp_grid(sch, sids, ctx)] * n_days for sch in EMP_SCHEMES}
        import torch
        q, _ = pipe.predict_quantiles(
            [torch.tensor(ctx[s], dtype=torch.float32) for s in sids],
            prediction_length=n_days, quantile_levels=list(NATIVE_LEVELS),
            batch_size=args.batch_size)
        g, _ = QuantileRepair()(to_grid(q))
        g = g.reshape(len(sids), n_days, -1)
        grids["chronos2-zs"] = [g[:, i, :] for i in range(n_days)]

        for arm, gr in grids.items():
            pmf_pi = convolve_varying_pmf(NATIVE_LEVELS, gr, vmax=VMAX)
            pmf_r = convolve_varying_pmf(NATIVE_LEVELS,
                                         gr[:month.days_in_month], vmax=VMAX)
            m_ratio = n_days / month.days_in_month
            for alpha in ALPHAS:
                S = order_up_to(pmf_r, pmf_pi, alpha, m_ratio)["P3"]
                rc = ReplayConfig(n_days=n_days, lead_time_days=LEAD_DAYS,
                                  review_cadence_days=n_days,
                                  shortage_mechanism="lost_sales")
                res = replay(demand_matrix, S[:, None], initial_inv, rc)
                h_c, p_c = costs_from_alpha(cost_i, alpha, KAPPA_H, 12)
                cost = h_c * res.i_end.mean(axis=1) + p_c * res.lost.sum(axis=1)
                pin = _alpha_pinball(S, y_pi, alpha)
                ok = np.isfinite(cost)
                long_rows.append(pd.DataFrame({
                    "series_id": sids[ok], "month": month.date(),
                    "origin_idx": oi, "alpha": alpha, "arm": arm,
                    "cost": cost[ok], "pinball": pin[ok]}))

    df = pd.concat(long_rows, ignore_index=True)
    df.to_csv(ART / "rolling_origins_long.csv", index=False)
    pd.DataFrame(neff_rows).to_csv(ART / "effective_n.csv", index=False)
    print(f"\n逐行成本已保存 ({len(df)} 行)。开始构造三种经验基线 …")

    # ------------------------------------------------ 三种经验基线构造
    origins = sorted(df.origin_idx.unique())
    piv = df.pivot_table(index=["series_id", "origin_idx", "alpha"],
                         columns="arm", values=["cost", "pinball"])

    mean_cost = (df[df.arm.isin(EMP_SCHEMES)]
                 .groupby(["origin_idx", "alpha", "arm"])["cost"].mean()
                 .reset_index())

    sel_rows = []
    for a in ALPHAS:
        for oi in origins:
            prior = mean_cost[(mean_cost.alpha == a) &
                              (mean_cost.origin_idx < oi)]
            if prior.empty:                      # 首个 origin 无历史可选
                retuned = "emp-daily"
            else:
                retuned = (prior.groupby("arm")["cost"].mean().idxmin())
            here = mean_cost[(mean_cost.alpha == a) &
                             (mean_cost.origin_idx == oi)]
            oracle = here.set_index("arm")["cost"].idxmin()
            sel_rows.append(dict(alpha=a, origin_idx=oi,
                                 retuned=retuned, oracle=oracle))
    sel = pd.DataFrame(sel_rows)

    # Emp-fixed：用第一个 origin 之后的选择永久冻结
    fixed_map = {}
    for a in ALPHAS:
        s0 = sel[(sel.alpha == a) & (sel.origin_idx == origins[min(1, len(origins)-1)])]
        fixed_map[a] = str(s0.iloc[0].retuned) if not s0.empty else "emp-daily"
    sel["fixed"] = sel.alpha.map(fixed_map)
    sel.to_csv(ART / "baseline_selection.csv", index=False)

    print("\n经验基线选择（每个 α × origin）:")
    for a in ALPHAS:
        sub = sel[sel.alpha == a]
        rt = sub.retuned.value_counts().to_dict()
        print(f"  α={a:.2f}  fixed={fixed_map[a]:<11} retuned={rt}")

    # ------------------------------------------------ 构造差值 d
    def _diff(kind: str, metric: str) -> pd.DataFrame:
        out = []
        for _, r in sel.iterrows():
            base_arm = r[kind]
            idx = (slice(None), r.origin_idx, r.alpha)
            try:
                blk = piv.loc[idx, :]
            except KeyError:
                continue
            if (metric, base_arm) not in blk.columns or \
               (metric, "chronos2-zs") not in blk.columns:
                continue
            b = blk[(metric, base_arm)]
            c = blk[(metric, "chronos2-zs")]
            m = b.notna() & c.notna()
            if not m.any():
                continue
            out.append(pd.DataFrame({
                "series_id": blk.index.get_level_values("series_id")[m],
                "origin_idx": r.origin_idx, "alpha": r.alpha,
                "baseline": base_arm,
                "d": (c - b)[m].to_numpy(),
                "base": b[m].to_numpy()}))
        return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

    results = {}
    for kind in ("retuned", "fixed", "oracle"):
        for metric in ("cost", "pinball"):
            results[(kind, metric)] = _diff(kind, metric)

    # ------------------------------------------------ 聚簇 bootstrap
    # 用**每簇充分统计量**实现，与逐行重采样精确等价但快约三个数量级：
    #   H1 斜率：预存每簇 X'X (k×k) 与 X'y (k)，每次重采样只需累加抽中簇的
    #            矩阵再解 k×k 正规方程（k = 2 + 月固定效应数）。固定效应在
    #            每次重采样中重新估计，故为**精确**的 cluster bootstrap。
    #   H2 尾部：预存每簇 (Σd_high, Σbase_high)，比值即统计量。
    def _prep(d: pd.DataFrame):
        z = sp_stats.norm.ppf(d.alpha.to_numpy())
        scale = max(np.abs(d.base.to_numpy()).mean(), 1e-9)
        rel = d.d.to_numpy() / scale
        oh = pd.get_dummies(d.origin_idx, drop_first=True).to_numpy(float)
        X = (np.column_stack([np.ones(len(z)), z, oh]) if oh.size
             else np.column_stack([np.ones(len(z)), z]))
        codes, uniq = pd.factorize(d.series_id)
        n_clu, k = len(uniq), X.shape[1]

        XtX = np.zeros((n_clu, k, k))
        Xty = np.zeros((n_clu, k))
        hi = d.alpha.isin(HIGH_ALPHAS).to_numpy()
        dh = np.zeros(n_clu)
        bh = np.zeros(n_clu)
        for c in range(n_clu):
            m = codes == c
            Xc, yc = X[m], rel[m]
            XtX[c] = Xc.T @ Xc
            Xty[c] = Xc.T @ yc
            if hi[m].any():
                dh[c] = d.d.to_numpy()[m][hi[m]].sum()
                bh[c] = d.base.to_numpy()[m][hi[m]].sum()
        return XtX, Xty, dh, bh, n_clu

    def _solve_slope(A: np.ndarray, b: np.ndarray) -> float:
        try:
            return float(np.linalg.solve(A, b)[1])
        except np.linalg.LinAlgError:
            return float(np.linalg.lstsq(A, b, rcond=None)[0][1])

    def _contrasts(d: pd.DataFrame, b: int, seed: int):
        XtX, Xty, dh, bh, n_clu = _prep(d)
        pt_slope = _solve_slope(XtX.sum(0), Xty.sum(0))
        pt_tail = float(dh.sum() / bh.sum() * 100) if bh.sum() else np.nan

        rng = np.random.default_rng(seed)
        sl = np.empty(b)
        tl = np.empty(b)
        for k in range(b):
            picks = rng.integers(0, n_clu, size=n_clu)
            sl[k] = _solve_slope(XtX[picks].sum(0), Xty[picks].sum(0))
            bs = bh[picks].sum()
            tl[k] = dh[picks].sum() / bs * 100 if bs else np.nan
        return (pt_slope, float(np.quantile(sl, .025)), float(np.quantile(sl, .975)),
                pt_tail, float(np.nanquantile(tl, .025)),
                float(np.nanquantile(tl, .975)))

    print("\n" + "=" * 78)
    print("预设 contrast（重采样单位 = series_id）")
    print("=" * 78)
    summ = []
    for (kind, metric), d in results.items():
        if d.empty:
            continue
        b1, b1lo, b1hi, tl, tllo, tlhi = _contrasts(d, args.b, SEED_BASE)
        summ.append(dict(baseline_kind=kind, metric=metric,
                         slope_beta1=b1, slope_lo=b1lo, slope_hi=b1hi,
                         slope_sig=not (b1lo <= 0 <= b1hi),
                         tail_pct=tl, tail_lo=tllo, tail_hi=tlhi,
                         tail_sig=not (tllo <= 0 <= tlhi),
                         n_clusters=d.series_id.nunique()))
        print(f"\n[{kind} / {metric}]  clusters={d.series_id.nunique()}")
        print(f"  H1 斜率 β1     = {b1:+.4f}  [{b1lo:+.4f}, {b1hi:+.4f}]  "
              f"{'*** 显著为负' if (b1hi < 0) else '*** 显著为正' if b1lo > 0 else 'n.s.'}")
        print(f"  H2 尾部水平 d_high = {tl:+.2f}%  [{tllo:+.2f}%, {tlhi:+.2f}%]  "
              f"{'*** 显著优于基线' if tlhi < 0 else '*** 显著劣于基线' if tllo > 0 else 'n.s.'}")

    sdf = pd.DataFrame(summ)
    sdf.to_csv(ART / "contrasts.csv", index=False)

    # selection regret
    print("\n" + "=" * 78)
    print("selection regret（retuned/fixed 相对 oracle 的额外成本）")
    print("=" * 78)
    for kind in ("retuned", "fixed"):
        dk, do = results[(kind, "cost")], results[("oracle", "cost")]
        if dk.empty or do.empty:
            continue
        reg = (dk.base.sum() - do.base.sum()) / do.base.sum() * 100
        print(f"  {kind:<9} 相对 oracle 基线贵 {reg:+.2f}%")
        chk.note(f"selection_regret_{kind}_pct", round(float(reg), 3))

    main_row = sdf[(sdf.baseline_kind == "retuned") & (sdf.metric == "cost")]
    if not main_row.empty:
        r = main_row.iloc[0]
        chk.note("primary_slope_beta1", round(float(r.slope_beta1), 4))
        chk.note("primary_slope_sig", bool(r.slope_sig))
        chk.note("primary_tail_pct", round(float(r.tail_pct), 3))
        chk.note("primary_tail_sig", bool(r.tail_sig))
        print("\n" + "=" * 78)
        print("主结论（Emp-retuned / 成本口径）")
        print("=" * 78)
        print(f"  随尾部改善 (β1<0):  "
              f"{'✓ 显著' if r.slope_sig and r.slope_beta1 < 0 else '✗ 不显著'}")
        print(f"  尾部真正胜出 (d_high<0): "
              f"{'✓ 显著' if r.tail_sig and r.tail_pct < 0 else '✗ 不显著'}")

    chk.n_rows = len(df)
    chk.finish(ART / "checks")
    print(f"\n结果已保存: {ART}")
    print(f"总耗时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
