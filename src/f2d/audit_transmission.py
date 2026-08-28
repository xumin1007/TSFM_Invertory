"""审计：静态尾部优势有多少传导进动态回放，以及为什么衰减。

背景：`audit_pinball_cost` 已确认
  - 恒等式 L0(静态 newsvendor) == (p+h)·ρ_α 精确成立
  - 加权与等权 pinball 的尾部 contrast 几乎相同（−8.26% vs −8.22%）
  - 断点在 静态(−8.26%) → 回放(−3.78%)

本脚本回答"为什么"，三部分：

A. 订货区分度诊断
   回放中每月只有 t=0 一个复核时点，初始库存 IP0 = beginning_inventory
   （所有 arm 共用），pipeline 从 0 起。若 S ≤ IP0 则订货量为 0，该
   (SKU, month, α) 上**所有 arm 的轨迹完全相同**，对 arm 差异贡献为零。
   量化：零订货占比、全臂同订占比、以及剔除这些行后的传导率。

B. arm 差的描述性映射
   δ0 = L0(ZS) − L0(Emp),  δ3 = L3(ZS) − L3(Emp)
   报告 Corr(δ0, δ3)、同号比例，以及原始成本单位和基线归一化单位下的
   固定效应回归斜率。原始成本单位的 γ 受 SKU 成本尺度影响，不能解释为
   "传导了多少"；主效应始终使用 C 部分的相对效果与加性衰减。

C. 衰减率的置信区间
   R0 = ΣδL0 / ΣL0(Emp),  R3 = ΣδL3 / ΣL3(Emp)
   乘性衰减 A = 1 − R3/R0（R0 近零时不稳定），同时报告更稳健的
   加性衰减 R3 − R0（百分点）。均按 SKU cluster 重采样。

用法:  PYTHONPATH=src python -m f2d.audit_transmission
前置:  artifacts/zhao_audit/audit_long_v2.csv（由 --rebuild 生成）
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
from .run_rolling_origins import (ALPHAS, EMP_SCHEMES, HIGH_ALPHAS, ORIGINS,
                                  _alpha_pinball, _emp_grid)

ART = cfgmod.ARTIFACT_DIR / "zhao_audit"
LONG = ART / "audit_long_v2.csv"
VMAX = 60
LEAD_DAYS = 1
KAPPA_H = 0.20
B = 10_000


def rebuild(args) -> pd.DataFrame:
    """重建逐行表，额外记录 S、初始库存与订货量（区分度诊断需要）。"""
    t0 = time.time()
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
    print("加载 Chronos-2 …")
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
        inv0 = cur["beginning_inventory"].to_numpy(float)
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
        print(f"{month:%Y-%m}  {len(sids)} 序列  ({time.time()-t0:.0f}s)")

        for arm, gr in grids.items():
            pmf_pi = convolve_varying_pmf(NATIVE_LEVELS, gr, vmax=VMAX)
            pmf_r = convolve_varying_pmf(NATIVE_LEVELS,
                                         gr[:month.days_in_month], vmax=VMAX)
            m_ratio = n_days / month.days_in_month
            for alpha in ALPHAS:
                S = order_up_to(pmf_r, pmf_pi, alpha, m_ratio)["P3"]
                h_c, p_c = costs_from_alpha(cost_i, alpha, KAPPA_H, 12)
                l0 = h_c * np.maximum(S - y_pi, 0) + p_c * np.maximum(y_pi - S, 0)
                rc = ReplayConfig(n_days=n_days, lead_time_days=LEAD_DAYS,
                                  review_cadence_days=n_days,
                                  shortage_mechanism="lost_sales")
                res = replay(dm, S[:, None], inv0, rc)
                l3 = h_c * res.i_end.mean(axis=1) + p_c * res.lost.sum(axis=1)
                ok = np.isfinite(l3) & np.isfinite(l0)
                rows.append(pd.DataFrame({
                    "series_id": sids[ok], "origin_idx": oi, "alpha": alpha,
                    "arm": arm, "L0": l0[ok], "L3": l3[ok],
                    "S": S[ok], "inv0": inv0[ok],
                    "order": res.order[:, 0][ok]}))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(LONG, index=False)
    print(f"已保存 {LONG} ({len(df)} 行)")
    return df


def _cluster_boot_stats(d: pd.DataFrame, b: int, seed: int):
    """按 series_id 聚簇，同时重算 R0、R3、加性与乘性衰减。"""
    codes, uniq = pd.factorize(d.series_id)
    n_clu = len(uniq)
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(n_clu))
    ends = np.searchsorted(codes[order], np.arange(n_clu), side="right")

    d0, b0 = d.d0.to_numpy(), d.b0.to_numpy()
    d3, b3 = d.d3.to_numpy(), d.b3.to_numpy()

    def stats(rows):
        r0 = d0[rows].sum() / b0[rows].sum() * 100
        r3 = d3[rows].sum() / b3[rows].sum() * 100
        add = r3 - r0
        mult = (1 - r3 / r0) * 100 if abs(r0) > 1e-9 else np.nan
        return r0, r3, add, mult

    pt = stats(np.arange(len(d)))
    rng = np.random.default_rng(seed)
    draws = np.empty((b, 4))
    for k in range(b):
        picks = rng.integers(0, n_clu, size=n_clu)
        rows = np.concatenate([order[starts[c]:ends[c]] for c in picks])
        draws[k] = stats(rows)
    lo = np.nanquantile(draws, .025, axis=0)
    hi = np.nanquantile(draws, .975, axis=0)
    return pt, lo, hi


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--n-series", type=int, default=2000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--b", type=int, default=B)
    args = ap.parse_args(argv)

    ART.mkdir(parents=True, exist_ok=True)
    chk = CheckResult(step_id="zhao_audit_transmission", dataset="zhao",
                      seed=SEED_BASE)
    df = rebuild(args) if (args.rebuild or not LONG.exists()) \
        else pd.read_csv(LONG)

    sel = pd.read_csv(cfgmod.ARTIFACT_DIR / "zhao_rolling_origins" /
                      "baseline_selection.csv")
    base_of = {(r.alpha, r.origin_idx): r.retuned for _, r in sel.iterrows()}

    # ---------------- A. 订货区分度 ----------------
    print("\n" + "=" * 74)
    print("A  订货区分度诊断（每月单一复核点，初始库存全臂共用）")
    print("=" * 74)
    hi_df = df[df.alpha.isin(HIGH_ALPHAS)]
    zero_share = (hi_df.order <= 1e-9).mean()
    grp = hi_df.groupby(["series_id", "origin_idx", "alpha"])["order"]
    all_zero = (grp.max() <= 1e-9).mean()
    ident = (grp.std().fillna(0.0) <= 1e-9).mean()
    print(f"  单臂零订货占比          {zero_share:6.1%}")
    print(f"  全臂皆零订货的 SKU-month {all_zero:6.1%}   ← 对 arm 差异贡献为 0")
    print(f"  全臂订货量相同的占比     {ident:6.1%}")
    chk.note("zero_order_share_high_alpha", round(float(zero_share), 4))
    chk.note("all_arms_zero_order_share", round(float(all_zero), 4))

    # ---------------- 构造 δ ----------------
    piv = df.pivot_table(index=["series_id", "origin_idx", "alpha"],
                         columns="arm", values=["L0", "L3"])
    recs = []
    for (sid, oi, a), row in piv.iterrows():
        if a not in HIGH_ALPHAS:
            continue
        ba = base_of.get((a, oi))
        if ba is None or ("L0", ba) not in piv.columns:
            continue
        l0z, l0b = row[("L0", "chronos2-zs")], row[("L0", ba)]
        l3z, l3b = row[("L3", "chronos2-zs")], row[("L3", ba)]
        if not np.isfinite([l0z, l0b, l3z, l3b]).all():
            continue
        recs.append((sid, oi, a, l0z - l0b, l0b, l3z - l3b, l3b))
    dd = pd.DataFrame(recs, columns=["series_id", "origin_idx", "alpha",
                                     "d0", "b0", "d3", "b3"])

    # ---------------- B. arm 差的描述性映射 ----------------
    print("\n" + "=" * 74)
    print("B  arm 差的描述性映射（不把回归斜率解释为传导率）")
    print("=" * 74)
    corr = float(np.corrcoef(dd.d0, dd.d3)[0, 1])
    nz = dd[(dd.d0.abs() > 1e-9) & (dd.d3.abs() > 1e-9)]
    same = float((np.sign(nz.d0) == np.sign(nz.d3)).mean())
    print(f"  Corr(δ0, δ3)          = {corr:+.4f}")
    print(f"  静态改善在动态中同号   = {same:.1%}  (n={len(nz)})")

    z = pd.get_dummies(dd.origin_idx, prefix="t", drop_first=True)
    za = pd.get_dummies(dd.alpha, prefix="a", drop_first=True)
    X = np.column_stack([dd.d0.to_numpy(), np.ones(len(dd)),
                         z.to_numpy(float), za.to_numpy(float)])
    gam_raw = float(np.linalg.lstsq(X, dd.d3.to_numpy(), rcond=None)[0][0])
    norm_ok = (dd.b0 > 1e-9) & (dd.b3 > 1e-9)
    dn = dd[norm_ok]
    x0 = dn.d0.to_numpy() / dn.b0.to_numpy()
    x3 = dn.d3.to_numpy() / dn.b3.to_numpy()
    zn = pd.get_dummies(dn.origin_idx, prefix="t", drop_first=True)
    zan = pd.get_dummies(dn.alpha, prefix="a", drop_first=True)
    Xn = np.column_stack([x0, np.ones(len(dn)),
                          zn.to_numpy(float), zan.to_numpy(float)])
    gam_norm = float(np.linalg.lstsq(Xn, x3, rcond=None)[0][0])
    corr_norm = float(np.corrcoef(x0, x3)[0, 1])
    print(f"  γ_raw  (成本单位；仅描述) = {gam_raw:+.4f}")
    print(f"  γ_norm (基线归一化；仅描述) = {gam_norm:+.4f}")
    print(f"  Corr(δ0/L0_base, δ3/L3_base) = {corr_norm:+.4f}")
    chk.note("corr_d0_d3", round(corr, 4))
    chk.note("sign_agreement", round(same, 4))
    chk.note("gamma_raw_cost_units_descriptive", round(gam_raw, 4))
    chk.note("gamma_normalized_descriptive", round(gam_norm, 4))
    chk.note("corr_normalized_differences", round(corr_norm, 4))
    chk.note("n_normalized_regression", int(len(dn)))

    # ---------------- C. 衰减率 CI ----------------
    print("\n" + "=" * 74)
    print("C  衰减率（SKU-cluster bootstrap, B=%d）" % args.b)
    print("=" * 74)
    pt, lo, hi = _cluster_boot_stats(dd, args.b, SEED_BASE)
    names = ["R0 静态相对效果", "R3 回放相对效果",
             "加性衰减 R3−R0", "乘性衰减 1−R3/R0"]
    units = ["%", "%", " pp", "%"]
    for i, (nm, u) in enumerate(zip(names, units)):
        print(f"  {nm:<20} {pt[i]:>+8.2f}{u}  [{lo[i]:>+8.2f}, {hi[i]:>+8.2f}]")
    chk.note("R0_static_pct", round(float(pt[0]), 3))
    chk.note("R3_replay_pct", round(float(pt[1]), 3))
    chk.note("additive_attenuation_pp", round(float(pt[2]), 3))
    chk.note("multiplicative_attenuation_pct", round(float(pt[3]), 3))

    # 剔除无区分度行后重算
    disc = dd.merge(
        grp.std().fillna(0.0).rename("osd").reset_index(),
        on=["series_id", "origin_idx", "alpha"], how="left")
    disc = disc[disc.osd > 1e-9]
    if len(disc) > 100:
        pt2, lo2, hi2 = _cluster_boot_stats(disc, args.b, SEED_BASE)
        print(f"\n  仅保留订货量有差异的行 (n={len(disc)}, "
              f"{len(disc)/len(dd):.0%}):")
        for i, (nm, u) in enumerate(zip(names, units)):
            print(f"  {nm:<20} {pt2[i]:>+8.2f}{u}  [{lo2[i]:>+8.2f}, {hi2[i]:>+8.2f}]")
        chk.note("R3_discriminating_only_pct", round(float(pt2[1]), 3))

    chk.n_rows = len(dd)
    chk.finish(ART / "checks_transmission")
    print(f"\n结果已保存: {ART}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
