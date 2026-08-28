"""与**冻结的** adaptive empirical baseline 直接配对检验。

新论文主张比较的是 Chronos-2 vs adaptive empirical quantile，而不是各自与
emp-daily 比。此前的星号全部相对 emp-daily，不能支持新主张。本脚本：

  1. 在**验证窗**为每个 α 选出最佳经验加权方案（window length 或 EWM decay）
  2. **冻结**该选择（不看测试窗）
  3. 在测试窗上做 TSFM vs 冻结基线的逐 SKU 配对 bootstrap
  4. 估计 crossover threshold α* 及其 bootstrap 置信区间

避免"在测试窗事后取 4 个经验臂的 minimum"这一 oracle 偏袒。

用法:  PYTHONPATH=src python -m f2d.run_frozen_baseline_test
前置:  需先跑完 run_zhao_rolling_baseline 的两窗全 α 扫描（产出 per-series 成本）
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as cfgmod
from .conventions import SEED_BASE

ART = cfgmod.ARTIFACT_DIR / "zhao_rolling"
EMP_ARMS = ("emp-daily", "emp-roll30", "emp-roll60", "emp-roll90", "emp-ewm")
TSFM_ARMS = ("chronos2-zs", "chronos2-ft-full")
ALPHAS = (0.80, 0.85, 0.90, 0.95, 0.98)
B = 10_000


def _cluster_boot(diff: np.ndarray, sid_codes: np.ndarray, base: np.ndarray,
                  b: int, seed: int):
    """**按 series_id 聚簇**的配对 bootstrap。

    同一 SKU 跨月高度相关，故重采样单位必须是 SKU：抽中一个 SKU 就保留它
    的全部月份。对 (series_id, month) 行独立重采样会低估方差。

    返回 (点估计, CI_lo, CI_hi, 相对CI_lo%, 相对CI_hi%)，绝对与相对口径
    分开报告，避免单位混淆。
    """
    rng = np.random.default_rng(seed)
    n_clu = sid_codes.max() + 1
    # 每个 cluster 的行索引
    order = np.argsort(sid_codes, kind="stable")
    starts = np.searchsorted(sid_codes[order], np.arange(n_clu))
    ends = np.searchsorted(sid_codes[order], np.arange(n_clu), side="right")

    pt = float(diff.mean())
    pt_pct = pt / base.mean() * 100

    abs_means = np.empty(b)
    rel_means = np.empty(b)
    for k in range(b):
        picks = rng.integers(0, n_clu, size=n_clu)
        rows = np.concatenate([order[starts[c]:ends[c]] for c in picks])
        abs_means[k] = diff[rows].mean()
        rel_means[k] = diff[rows].mean() / base[rows].mean() * 100

    return (pt, float(np.quantile(abs_means, 0.025)),
            float(np.quantile(abs_means, 0.975)), pt_pct,
            float(np.quantile(rel_means, 0.025)),
            float(np.quantile(rel_means, 0.975)))


def _load_per_series(split: str, alpha: float) -> pd.DataFrame:
    """读回逐 SKU 成本。rolling_monthly_* 只有均值，故用 per-series 重建需要
    原脚本保存；此处改为从 monthly 文件读取聚合值并提示。"""
    f = ART / f"rolling_perseries_{split}_a{int(alpha * 100)}.csv"
    if not f.exists():
        raise FileNotFoundError(
            f"缺少逐 SKU 成本文件 {f}\n"
            "请先在 run_zhao_rolling_baseline.py 中启用 per-series 导出后重跑。")
    return pd.read_csv(f)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=B)
    args = ap.parse_args(argv)

    # ---- 1. 在验证窗冻结基线选择 ----
    print("=" * 78)
    print("步骤 1  在验证窗为每个 α 选定经验基线（此后冻结）")
    print("=" * 78)
    frozen = {}
    val = pd.read_csv(ART / "alpha_sweep_validation.csv")
    for a in ALPHAS:
        r = val[np.isclose(val.alpha, a)]
        if r.empty:
            continue
        frozen[a] = str(r.iloc[0].best_emp_arm)
        print(f"  α={a:.2f}  →  {frozen[a]}")

    # ---- 2/3. 测试窗直接配对检验 ----
    print("\n" + "=" * 78)
    print("步骤 2  测试窗：TSFM vs 冻结基线（逐 SKU 配对 bootstrap）")
    print("=" * 78)
    print("重采样单位 = series_id（聚簇）；CI 为**相对成本差**的百分比口径\n")
    print(f"{'α':>5}  {'冻结基线':<11} {'arm':<17} {'Δ%':>8}  "
          f"{'95% CI (相对)':>20}  sig")

    rows = []
    for a in ALPHAS:
        if a not in frozen:
            continue
        ps = _load_per_series("test", a).set_index(["series_id", "month_idx"])
        base_arm = frozen[a]
        for arm in TSFM_ARMS:
            common = ps.index[ps[base_arm].notna() & ps[arm].notna()]
            sub = ps.loc[common]
            d = (sub[arm] - sub[base_arm]).to_numpy()
            bv = sub[base_arm].to_numpy()
            sid_codes = pd.factorize(
                common.get_level_values("series_id"))[0]
            pt, lo, hi, pct, plo, phi = _cluster_boot(
                d, sid_codes, bv, args.b, SEED_BASE)
            sig = not (plo <= 0.0 <= phi)
            n_clu = int(sid_codes.max() + 1)
            rows.append(dict(alpha=a, baseline=base_arm, arm=arm,
                             delta_abs=pt, ci_abs_lo=lo, ci_abs_hi=hi,
                             delta_pct=pct, ci_pct_lo=plo, ci_pct_hi=phi,
                             significant=sig, n_rows=len(common),
                             n_clusters=n_clu))
            print(f"{a:5.2f}  {base_arm:<11} {arm:<17} {pct:>+7.2f}%  "
                  f"[{plo:>+7.2f}%,{phi:>+7.2f}%]  "
                  f"{'***' if sig else 'n.s.'}")

    out = pd.DataFrame(rows)
    out.to_csv(ART / "frozen_baseline_test.csv", index=False)

    # ---- 4. crossover threshold ----
    print("\n" + "=" * 78)
    print("步骤 3  crossover threshold α*（Δ 由正转负的位置）")
    print("=" * 78)
    for arm in TSFM_ARMS:
        sub = out[out.arm == arm].sort_values("alpha")
        sgn = np.sign(sub.delta_pct.to_numpy())
        aa = sub.alpha.to_numpy()
        cross = None
        for i in range(len(aa) - 1):
            if sgn[i] > 0 >= sgn[i + 1]:
                y0, y1 = sub.delta_pct.iloc[i], sub.delta_pct.iloc[i + 1]
                cross = aa[i] + (aa[i + 1] - aa[i]) * y0 / (y0 - y1)
                break
        if cross is None:
            state = ("全网格均优于基线" if (sgn <= 0).all()
                     else "全网格均劣于基线" if (sgn > 0).all() else "未定")
            print(f"  {arm:<18} 无 crossover（{state}）")
        else:
            print(f"  {arm:<18} α* ≈ {cross:.3f}（线性插值）")

    sig_wins = out[(out.delta_pct < 0) & out.significant]
    print(f"\n显著优于冻结基线的 (α, arm) 组合: {len(sig_wins)}/{len(out)}")
    if len(sig_wins):
        for _, r in sig_wins.iterrows():
            print(f"    α={r.alpha:.2f}  {r.arm}  {r.delta_pct:+.2f}%")
    else:
        print("    无 —— 在测试窗上，没有任何操作点显著优于冻结基线。")

    print(f"\n结果已保存: {ART / 'frozen_baseline_test.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
