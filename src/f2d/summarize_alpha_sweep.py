"""汇总 α 扫描：TSFM 相对**最佳经验臂**的优势如何随服务水平变化。

读取 `run_zhao_rolling_baseline.py` 在各 α 下产生的 rolling_bootstrap_a*.csv，
输出两张表：
  1. 各 α 下每臂的成本与相对 emp-daily 的 Δ
  2. 新叙事的核心证据 —— TSFM 相对最佳经验臂的优势 vs α，
     以及经验加权能复制的优势份额

用法:  PYTHONPATH=src python -m f2d.summarize_alpha_sweep
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfgmod

ART = cfgmod.ARTIFACT_DIR / "zhao_rolling"
EMP_ARMS = ("emp-daily", "emp-roll30", "emp-roll60", "emp-roll90", "emp-ewm")
TSFM_ARMS = ("chronos2-zs", "chronos2-ft-full")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("validation", "test"),
                    default="validation")
    args = ap.parse_args(argv)
    split = args.split
    files = sorted(ART.glob(f"rolling_bootstrap_{split}_a*.csv"))
    if not files:
        print(f"未找到结果文件于 {ART}")
        return 1

    frames = []
    for f in files:
        a = int(f.stem.removeprefix(f"rolling_bootstrap_{split}_a")) / 100
        d = pd.read_csv(f)
        d["alpha"] = a
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    alphas = sorted(df.alpha.unique())

    # ---- 表 1：成本与 Δ ----
    print("=" * 78)
    print(f"表 1  各服务水平下的成本（{split} 窗，两月合并，P3，κ_h=0.20）")
    print("=" * 78)
    print(f"{'arm':<20}", end="")
    for a in alphas:
        print(f"{a:>11.2f}", end="")
    print()
    for arm in list(EMP_ARMS) + list(TSFM_ARMS):
        sub = df[df.arm == arm].set_index("alpha")
        if sub.empty:
            continue
        print(f"{arm:<20}", end="")
        for a in alphas:
            print(f"{sub.loc[a, 'cost']:>11.3f}", end="")
        print()

    # ---- 表 2：核心证据 ----
    print("\n" + "=" * 78)
    print("表 2  TSFM 相对**最佳经验臂**的优势 vs 服务水平")
    print("=" * 78)
    rows = []
    for a in alphas:
        sub = df[df.alpha == a].set_index("arm")
        emp = sub.loc[[x for x in EMP_ARMS if x in sub.index]]
        best_emp = emp.cost.idxmin()
        c_best = emp.cost.min()
        c_daily = sub.loc["emp-daily", "cost"]
        c_zs = sub.loc["chronos2-zs", "cost"]
        c_ft = sub.loc["chronos2-ft-full", "cost"]

        # 经验加权能复制的份额：(emp-daily - best_emp) / (emp-daily - ft)
        gain_ft = c_daily - c_ft
        gain_emp = c_daily - c_best
        share = gain_emp / gain_ft if gain_ft > 0 else np.nan

        rows.append(dict(
            alpha=a, best_emp_arm=best_emp,
            cost_best_emp=c_best, cost_zs=c_zs, cost_ft=c_ft,
            zs_vs_best_emp_pct=(c_zs - c_best) / c_best * 100,
            ft_vs_best_emp_pct=(c_ft - c_best) / c_best * 100,
            emp_replicated_share=share,
            ft_sig=bool(sub.loc["chronos2-ft-full", "significant"]),
            zs_sig=bool(sub.loc["chronos2-zs", "significant"])))

    out = pd.DataFrame(rows)
    print(f"{'α':>5}  {'最佳经验臂':<12} {'ZS vs 最佳':>11} {'FT vs 最佳':>11} "
          f"{'经验可复制份额':>14}")
    for _, r in out.iterrows():
        print(f"{r.alpha:5.2f}  {r.best_emp_arm:<12} "
              f"{r.zs_vs_best_emp_pct:>+10.2f}% {r.ft_vs_best_emp_pct:>+10.2f}% "
              f"{r.emp_replicated_share:>13.0%}")

    out.to_csv(ART / f"alpha_sweep_{split}.csv", index=False)

    # ---- 判定 ----
    print("\n" + "=" * 78)
    lo_a, hi_a = out.alpha.min(), out.alpha.max()
    lo = out[out.alpha == lo_a].iloc[0]
    hi = out[out.alpha == hi_a].iloc[0]
    print(f"α={lo_a:.2f}: FT 相对最佳经验臂 {lo.ft_vs_best_emp_pct:+.2f}%，"
          f"经验加权复制了 {lo.emp_replicated_share:.0%}")
    print(f"α={hi_a:.2f}: FT 相对最佳经验臂 {hi.ft_vs_best_emp_pct:+.2f}%，"
          f"经验加权复制了 {hi.emp_replicated_share:.0%}")
    trend = hi.ft_vs_best_emp_pct - lo.ft_vs_best_emp_pct
    if trend < -0.5:
        print(f"\n✓ 优势随服务水平单调扩大（{trend:+.2f}pp）——"
              f"支持'价值集中在尾部'的机制叙事。")
    elif trend > 0.5:
        print(f"\n✗ 优势随服务水平缩小（{trend:+.2f}pp）——尾部叙事不成立。")
    else:
        print(f"\n~ 优势对服务水平不敏感（{trend:+.2f}pp）——尾部叙事证据不足。")

    n_zs_lose = int((out.zs_vs_best_emp_pct > 0).sum())
    print(f"\nzero-shot 输给最佳经验臂的 α 点数: {n_zs_lose}/{len(out)}")
    print(f"\n结果已保存: {ART / f'alpha_sweep_{split}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
