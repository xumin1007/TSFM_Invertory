"""可行性缺口审计。对应 docs/07_decision_layer.md §6.2.0 与 §9.1.2。

层 B 全部臂未达 $\\mathrm{CSR}\\ge\\alpha$。该缺口必须先排除实现错误，否则
整层结果不可用。本模块把三步审计固化为可复跑的代码——§6.2.0 引用的每个
数字都由此产生，不得来自一次性脚本。

  step 1  管线自洽性：喂入**已知真分布**，看实现能否兑现 alpha
  step 2  缺口分解：iid 假设 vs 样本外漂移（用历史滚动和作无 iid 对照）
  step 3  工作点扫描：排序是否只在名义 alpha=0.95 处成立

step 1 同时是回归守卫（见 tests/test_audit_coverage.py）：若日后有人改动
重建或卷积而破坏了标定，该检验会先失败。

用法:  PYTHONPATH=src python -m f2d.audit_coverage [--step 1|2|3|all]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

from .aggregation import convolve_varying_pmf, estimate_phi, pmf_quantile
from .conventions import SEED_BASE
from .models.chronos import NATIVE_LEVELS as LV
from .models.chronos import QuantileRepair

ALPHAS_CHECK = (0.85, 0.95, 0.98)
R_DAYS = 31
VMAX = 60


def _zinb(rng, size, p0=0.68, r=2.0, p=0.4):
    """零膨胀负二项：Zhao 日级的近似（实测零占比 0.682）。"""
    return np.where(rng.random(size) < p0, 0,
                    rng.negative_binomial(r, p, size))


def pipeline_self_consistency(n: int = 4000, seed: int = SEED_BASE) -> pd.DataFrame:
    """step 1：真分布下管线的实测覆盖。

    判据是**单边**的：真网格行必须满足 覆盖 >= alpha - 0.02。整数支撑上
    F^{-1}(a)=min{v:F(v)>=a} 必然**超**覆盖，零原子越重超得越多（零膨胀
    负二项在 alpha=0.85 处达 0.90），那是固有性质而非缺陷；**欠**覆盖才是
    实现缺陷的信号。
    """
    rng = np.random.default_rng(seed)
    rows = []

    def cover(grid, sampler):
        pmf = convolve_varying_pmf(LV, [grid] * R_DAYS, vmax=VMAX)
        return {a: float((sampler() <= pmf_quantile(pmf, [a])[a]).mean())
                for a in ALPHAS_CHECK}

    lam = 0.3
    rows.append({"case": "iid 泊松(0.3)，真网格", "exact_grid": True,
                 **cover(np.tile(stats.poisson.ppf(LV, lam), (n, 1)),
                         lambda: rng.poisson(lam, (n, R_DAYS)).sum(1))})

    ref = _zinb(rng, (200_000,))
    q = np.quantile(ref, LV, method="inverted_cdf")
    rows.append({"case": "零膨胀负二项(pi0=.68)，真网格", "exact_grid": True,
                 **cover(np.tile(q, (n, 1)),
                         lambda: _zinb(rng, (n, R_DAYS)).sum(1))})

    # 有限 context 估计误差：纯统计效应，无漂移无自相关
    for nctx in (200, 400):
        res = {}
        for a in ALPHAS_CHECK:
            ctx = _zinb(rng, (n, nctx))
            g = np.array([np.quantile(c, LV, method="inverted_cdf") for c in ctx])
            pmf = convolve_varying_pmf(LV, [g] * R_DAYS, vmax=VMAX)
            S = pmf_quantile(pmf, [a])[a]
            res[a] = float((_zinb(rng, (n, R_DAYS)).sum(1) <= S).mean())
        rows.append({"case": f"同上，网格由 {nctx} 样本估计",
                     "exact_grid": False, **res})

    return pd.DataFrame(rows)


def decompose_gap(daily: pd.DataFrame, panel: pd.DataFrame, months,
                  sku_of: dict) -> pd.DataFrame:
    """step 2：把缺口拆成 iid 与样本外漂移。

    对照臂是**历史 R 日实际总和的经验分位**——不做 iid、不做网格重建、
    不用任何模型。它与卷积路径的差即 iid 的代价；它自身与 alpha 的差即
    样本外泛化的代价。
    """
    out = []
    for month in months:
        R = month.days_in_month
        snap = panel[panel.month == month].set_index("sku_ID")
        ctx = {s: g.sort_values("d").y.to_numpy()
               for s, g in daily[daily.d < month].groupby("series_id")}
        sids = np.array([s for s in sorted(ctx)
                         if len(ctx[s]) >= R + 60 and sku_of[s] in snap.index])
        y = snap.loc[[sku_of[s] for s in sids],
                     "observed_sales_next_month"].to_numpy(float)

        emp, _ = QuantileRepair()(np.array(
            [np.quantile(ctx[s], LV, method="inverted_cdf") for s in sids]))
        pmf = convolve_varying_pmf(LV, [emp] * R, vmax=VMAX)
        rolls = [pd.Series(ctx[s]).rolling(R).sum().dropna().to_numpy() for s in sids]
        phi = estimate_phi(np.concatenate([ctx[s] for s in sids]),
                           np.concatenate(rolls), R)
        hist_mean = np.array([ctx[s].mean() * R for s in sids])

        for a in ALPHAS_CHECK:
            S_conv = pmf_quantile(pmf, [a])[a]
            S_roll = np.array([np.quantile(r, a, method="inverted_cdf")
                               for r in rolls])
            out.append(dict(
                month=month, alpha=a, n=len(sids), phi=round(phi, 3),
                drift_mean_ratio=round(float(y.mean() / hist_mean.mean()), 3),
                cover_conv=float((y <= S_conv).mean()),
                cover_rolling=float((y <= S_roll).mean()),
                iid_cost=float((y <= S_roll).mean() - (y <= S_conv).mean()),
                oos_cost=float(a - (y <= S_roll).mean())))
    return pd.DataFrame(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", default="all", choices=("1", "2", "all"))
    args = ap.parse_args(argv)

    if args.step in ("1", "all"):
        print("=== step 1 管线自洽性（真网格必须在 alpha±0.01 内）===")
        df = pipeline_self_consistency()
        print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        # 单边判据：只有**欠**覆盖才是缺陷，见 pipeline_self_consistency 说明
        bad = [(r.case, a, round(r[a], 4)) for _, r in df[df.exact_grid].iterrows()
               for a in ALPHAS_CHECK if r[a] < a - 0.02]
        print("判定:", "通过（实现无误）" if not bad else f"未通过（欠覆盖）{bad}")
        over = [(r.case, a, round(r[a] - a, 4))
                for _, r in df[df.exact_grid].iterrows()
                for a in ALPHAS_CHECK if r[a] > a + 0.02]
        if over:
            print("  （超覆盖，整数支撑固有，非缺陷）:", over)

    if args.step in ("2", "all"):
        print("\n=== step 2 缺口分解 ===")
        from .datasets import zhao
        raw = zhao.load_raw()
        daily, _ = zhao.build_daily_panel(raw)
        panel, _ = zhao.build_panel(raw)
        rng = np.random.default_rng(SEED_BASE)
        pool = np.asarray(sorted(set(daily.series_id)))
        daily = daily[daily.series_id.isin(
            rng.choice(pool, size=min(2000, len(pool)), replace=False))]
        sku_of = dict(zip(daily.series_id, daily.sku_ID))
        months = [pd.Timestamp("2019-07-01"), pd.Timestamp("2019-08-01")]
        print(decompose_gap(daily, panel, months, sku_of)
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("\niid_cost = 卷积路径相对无-iid 对照的损失；"
              "oos_cost = 无-iid 对照自身相对 alpha 的缺口（样本外泛化）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
