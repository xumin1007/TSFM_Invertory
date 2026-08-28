"""恒等式审计：realized cost 与 α-pinball 为何得出不同结论。

代数事实：若 α = p/(p+h)，则**静态**单期 newsvendor 损失逐观测严格等于
    h(q−Y)⁺ + p(Y−q)⁺ = (p+h)·ρ_α(Y−q)
故在同一 (Y, q, α, 样本, 权重) 下，二者不可能给出不同结论。

但本项目的 realized cost **不是**静态损失，而是回放模拟量：
    cost = h·mean_t(i_end) + p·Σ_t(lost)
其中 i_end 是逐日期末库存的时间平均，lost 是给定初始库存、订货、提前期
到货、逐日消耗后的实际失销。因此恒等式在构造上即不成立。

本脚本把差异分解为三层，逐层量化：
    L0  静态 newsvendor 损失  h(S−y_PI)⁺ + p(y_PI−S)⁺      —— 应与加权 pinball 精确相等
    L1  (p+h)-加权 pinball    (p+h)·ρ_α(y_PI−S)             —— 恒等式检验目标
    L2  等权 pinball          ρ_α(y_PI−S)                    —— 当前 contrasts 用的口径
    L3  回放实现成本          h·mean(i_end) + p·Σ(lost)      —— 当前 cost 口径

并检验：预测优势是否与经济权重 (p+h) 错配 —— 即改善是否集中在低价值 SKU。

用法:  PYTHONPATH=src python -m f2d.audit_pinball_cost [--n-series 2000]
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
from .run_rolling_origins import (ALPHAS, EMP_SCHEMES, HIGH_ALPHAS, ORIGINS,
                                  _alpha_pinball, _emp_grid)

ART = cfgmod.ARTIFACT_DIR / "zhao_audit"
VMAX = 60
LEAD_DAYS = 1
KAPPA_H = 0.20
TOL = 1e-8


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=2000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args(argv)

    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    chk = CheckResult(step_id="zhao_audit_pinball_cost", dataset="zhao",
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
    print(f"加载 Chronos-2 …")
    pipe = BaseChronosPipeline.from_pretrained(BASE_CHECKPOINT,
                                               device_map=args.device)

    rows = []
    for oi, month in enumerate(ORIGINS):
        n_days = month.days_in_month + LEAD_DAYS
        snap = panel[panel.month == month]
        if snap.empty:
            continue
        snap = snap.set_index("sku_ID")
        hist = daily[daily.d < month]
        ctx = {s: g.y.to_numpy() for s, g in hist.groupby("series_id")}
        sids = np.array([s for s in sorted(ctx)
                         if len(ctx[s]) >= 30 and sku_of[s] in snap.index])
        if len(sids) < 50:
            continue
        sk = np.array([sku_of[s] for s in sids])
        cur = snap.loc[sk]
        initial_inv = cur["beginning_inventory"].to_numpy(float)
        cost_i = cur["unit_cost_hist"].to_numpy(float)

        dm = np.zeros((len(sids), n_days))
        md = daily[(daily.d >= month) &
                   (daily.d < month + pd.DateOffset(months=1)
                    + pd.Timedelta(days=LEAD_DAYS))]
        for i, s in enumerate(sids):
            sd = md[md.series_id == s].set_index("d")
            for k in range(n_days):
                d = month + pd.Timedelta(days=k)
                if d in sd.index:
                    dm[i, k] = float(sd.loc[d, "y"])
        y_pi = dm.sum(axis=1)

        grids = {sch: [_emp_grid(sch, sids, ctx)] * n_days for sch in EMP_SCHEMES}
        import torch
        q, _ = pipe.predict_quantiles(
            [torch.tensor(ctx[s], dtype=torch.float32) for s in sids],
            prediction_length=n_days, quantile_levels=list(NATIVE_LEVELS),
            batch_size=args.batch_size)
        g, _ = QuantileRepair()(to_grid(q))
        g = g.reshape(len(sids), n_days, -1)
        grids["chronos2-zs"] = [g[:, i, :] for i in range(n_days)]

        print(f"{month:%Y-%m}  序列 {len(sids)}  ({time.time()-t0:.0f}s)")

        for arm, gr in grids.items():
            pmf_pi = convolve_varying_pmf(NATIVE_LEVELS, gr, vmax=VMAX)
            pmf_r = convolve_varying_pmf(NATIVE_LEVELS,
                                         gr[:month.days_in_month], vmax=VMAX)
            m_ratio = n_days / month.days_in_month
            for alpha in ALPHAS:
                S = order_up_to(pmf_r, pmf_pi, alpha, m_ratio)["P3"]
                h_c, p_c = costs_from_alpha(cost_i, alpha, KAPPA_H, 12)

                # L0 静态 newsvendor 损失
                l0 = h_c * np.maximum(S - y_pi, 0) + p_c * np.maximum(y_pi - S, 0)
                # L1 (p+h)-加权 pinball
                pin = _alpha_pinball(S, y_pi, alpha)
                l1 = (p_c + h_c) * pin
                # L2 等权 pinball
                l2 = pin
                # L3 回放实现成本
                rc = ReplayConfig(n_days=n_days, lead_time_days=LEAD_DAYS,
                                  review_cadence_days=n_days,
                                  shortage_mechanism="lost_sales")
                res = replay(dm, S[:, None], initial_inv, rc)
                l3 = h_c * res.i_end.mean(axis=1) + p_c * res.lost.sum(axis=1)

                ok = np.isfinite(l3) & np.isfinite(l0)
                rows.append(pd.DataFrame({
                    "series_id": sids[ok], "origin_idx": oi, "alpha": alpha,
                    "arm": arm, "L0_static": l0[ok], "L1_wpin": l1[ok],
                    "L2_pin": l2[ok], "L3_replay": l3[ok],
                    "econ_w": (p_c + h_c)[ok], "unit_cost": cost_i[ok]}))

    df = pd.concat(rows, ignore_index=True)
    df.to_csv(ART / "audit_long.csv", index=False)

    # ---------------------------------------------------- 恒等式检验
    print("\n" + "=" * 74)
    print("检验 1  静态损失 L0 == (p+h)·ρ_α  [L1]   —— 应精确成立")
    print("=" * 74)
    dev = np.abs(df.L0_static - df.L1_wpin).max()
    print(f"  max |L0 − L1| = {dev:.3e}   {'✓ 通过' if dev < TOL else '✗ 不成立'}")
    chk.assert_true("identity_L0_L1", bool(dev < TOL), f"max dev {dev:.3e}")

    print("\n" + "=" * 74)
    print("检验 2  回放成本 L3 vs 静态损失 L0   —— 预期**不**相等（动态项）")
    print("=" * 74)
    d30 = (df.L3_replay - df.L0_static)
    rel = np.abs(d30).sum() / df.L0_static.sum() * 100
    print(f"  max |L3 − L0| = {np.abs(d30).max():.3e}")
    print(f"  Σ|L3 − L0| / ΣL0 = {rel:.1f}%   相关系数 = "
          f"{np.corrcoef(df.L3_replay, df.L0_static)[0,1]:.4f}")
    chk.note("replay_vs_static_rel_gap_pct", round(float(rel), 2))

    # ---------------------------------------------------- 三种口径下的 H2
    print("\n" + "=" * 74)
    print("检验 3  同一样本、同一基线下，四种口径的尾部 contrast d_high")
    print("=" * 74)

    piv = df.pivot_table(index=["series_id", "origin_idx", "alpha"],
                         columns="arm",
                         values=["L0_static", "L1_wpin", "L2_pin", "L3_replay"])
    sel = pd.read_csv(cfgmod.ARTIFACT_DIR / "zhao_rolling_origins" /
                      "baseline_selection.csv")

    def _tail(metric: str) -> tuple[float, int]:
        num = den = 0.0
        n = 0
        for _, r in sel.iterrows():
            try:
                blk = piv.loc[(slice(None), r.origin_idx, r.alpha), :]
            except KeyError:
                continue
            if r.alpha not in HIGH_ALPHAS:
                continue
            if (metric, r.retuned) not in blk.columns:
                continue
            b = blk[(metric, r.retuned)]
            c = blk[(metric, "chronos2-zs")]
            m = b.notna() & c.notna()
            num += (c - b)[m].sum()
            den += b[m].sum()
            n += int(m.sum())
        return (num / den * 100 if den else np.nan), n

    print(f"{'口径':<26} {'d_high':>10}  说明")
    for metric, desc in [("L0_static", "静态 newsvendor（=加权 pinball）"),
                         ("L1_wpin", "(p+h)-加权 pinball"),
                         ("L2_pin", "等权 pinball（当前 contrasts 口径）"),
                         ("L3_replay", "回放实现成本（当前 cost 口径）")]:
        v, n = _tail(metric)
        print(f"{metric:<26} {v:>+9.2f}%  {desc}")
        chk.note(f"d_high_{metric}", round(float(v), 3))

    # ---------------------------------------------------- 经济权重错配检验
    print("\n" + "=" * 74)
    print("检验 4  预测优势是否与经济权重错配（按 p+h 分五组）")
    print("=" * 74)
    hi = df[df.alpha.isin(HIGH_ALPHAS)]
    w = (hi[hi.arm == "chronos2-zs"]
         .set_index(["series_id", "origin_idx", "alpha"])["econ_w"])
    imp = {}
    for sch in EMP_SCHEMES:
        a = hi[hi.arm == "chronos2-zs"].set_index(
            ["series_id", "origin_idx", "alpha"])["L2_pin"]
        b = hi[hi.arm == sch].set_index(
            ["series_id", "origin_idx", "alpha"])["L2_pin"]
        common = a.index.intersection(b.index)
        imp[sch] = (a.loc[common] - b.loc[common])
    best = min(imp, key=lambda k: imp[k].mean())
    d = imp[best]
    ww = w.loc[d.index]
    qs = pd.qcut(ww.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    print(f"  对照臂 = {best}（尾部平均 pinball 差最小者）")
    print(f"  {'econ_w 五分位':<16} {'平均 pinball 差':>16} {'平均 p+h':>12}")
    for qi in [1, 2, 3, 4, 5]:
        m = qs == qi
        print(f"  Q{qi} {'(低价值)' if qi==1 else '(高价值)' if qi==5 else '':<11}"
              f"{d[m].mean():>+16.4f} {ww[m].mean():>12.2f}")
    rho, pv = sp_stats.spearmanr(ww, d)
    print(f"\n  Spearman ρ(econ_w, pinball 差) = {rho:+.4f}  (p = {pv:.3e})")
    print(f"  {'→ 改善集中在低价值 SKU（错配）' if rho > 0.02 else ''}"
          f"{'→ 改善集中在高价值 SKU（对齐）' if rho < -0.02 else ''}"
          f"{'→ 与经济权重基本无关' if abs(rho) <= 0.02 else ''}")
    chk.note("spearman_econw_vs_pinball_gain", round(float(rho), 4))

    chk.n_rows = len(df)
    chk.finish(ART / "checks")
    print(f"\n结果已保存: {ART}")
    print(f"总耗时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
