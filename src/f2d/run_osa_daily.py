"""OSA 日级预测：验证窗机制演示。

OSA 统计功效不足以做模型排名（验证窗仅 6 完整周行，relative MDE 41%），
故本脚本定位为**机制演示**：证明同一预测-聚合-评价管线在不同数据结构
（连续需求、稀疏面板、EXCLUDE 缺失处理）上可运行且结果合理。

不做模型排名声明，不做配对 bootstrap（样本量不支撑）。

与 run_zhao_daily 的区别：
  1. OSA gap_decision=EXCLUDE（不补零），上下文仅用有观测的日
  2. 周级目标要求 7 天完整观测（完整周仅 ~9.5%）
  3. 100 条序列全部使用（不抽样）

用法:  PYTHONPATH=src python -m f2d.run_osa_daily [--device mps]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from . import config as cfgmod
from .aggregation import convolve_varying
from .checks import CheckResult
from .conventions import SEED_BASE
from .datasets import osa
from .metrics import evaluate_slice
from .models.chronos import (BASE_CHECKPOINT, NATIVE_LEVELS, QuantileRepair,
                             to_grid)

ART = cfgmod.ARTIFACT_DIR / "osa"
HORIZON = 7
VMAX = 300
MIN_CONTEXT = 30


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)

    import torch
    from chronos import BaseChronosPipeline

    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    chk = CheckResult(step_id="osa_daily_validation", dataset="osa", seed=SEED_BASE)
    cfg = cfgmod.load("osa")

    raw = osa.load_raw()
    daily, audit = osa.build_daily_panel(raw)
    osa.write_audit(audit)
    chk.note("panel_audit", audit)
    print(f"面板: {audit['n_rows']} 行, {audit['n_series']} 序列, "
          f"密度 {audit['density_median']:.2%}, 零占比 {audit['zero_share']:.2%}")

    weekly = osa.aggregate_to_weekly(daily)
    n_excl = weekly.attrs.get("n_excluded_incomplete", 0)
    print(f"完整周: {len(weekly)} 行 (排除不完整 {n_excl})")

    # 验证窗 origins
    val_start = pd.Timestamp(cfg.splits["validation"]["origins"][0])
    val_end = pd.Timestamp(cfg.splits["validation"]["origins"][1])
    origins = pd.date_range(val_start, val_end, freq="7D")
    target = weekly[weekly.origin.isin(origins)][["series_id", "origin", "y"]].copy()
    print(f"验证窗: {len(origins)} 个 origin, {len(target)} 个完整周行, "
          f"{target.series_id.nunique()} 条序列")

    if len(target) == 0:
        print("⚠ 验证窗无完整周行，跳过预测")
        chk.note("skip_reason", "no_complete_weeks_in_validation")
        chk.finish(ART / "checks" / "daily_validation.json")
        return chk.exit_code

    # 加载 Chronos-2
    pipe = BaseChronosPipeline.from_pretrained(
        BASE_CHECKPOINT, device_map=args.device, torch_dtype=torch.float32)
    repair = QuantileRepair()
    rows = []

    for origin in origins:
        cur = target[target.origin == origin]
        if not len(cur):
            continue
        sids = cur.series_id.to_numpy()

        # 上下文：origin 之前的有观测日（不补零）
        hist = daily[(daily.d < origin) & (daily.series_id.isin(sids))]
        ctx = {s: g.y.to_numpy() for s, g in hist.groupby("series_id")}
        sids = np.array([s for s in sids if len(ctx.get(s, [])) >= MIN_CONTEXT])
        cur = cur[cur.series_id.isin(sids)]
        if not len(sids):
            print(f"  {origin:%Y-%m-%d}  跳过（无足够上下文的序列）")
            continue

        # --- chronos2-zs ---
        q, _ = pipe.predict_quantiles(
            [torch.tensor(ctx[s], dtype=torch.float32) for s in sids],
            prediction_length=HORIZON, quantile_levels=list(NATIVE_LEVELS),
            batch_size=args.batch_size)
        grid = to_grid(q)
        grid, flags = repair(grid)
        per_step = [grid.reshape(len(sids), HORIZON, -1)[:, h, :] for h in range(HORIZON)]
        ch = convolve_varying(NATIVE_LEVELS, per_step, taus=(.5, .85), vmax=VMAX)

        # --- emp-daily ---
        emp = np.array([np.quantile(ctx[s], NATIVE_LEVELS, method="inverted_cdf")
                        for s in sids], float)
        ed = convolve_varying(NATIVE_LEVELS, [emp] * HORIZON, taus=(.5, .85), vmax=VMAX)

        y = cur.set_index("series_id").loc[sids, "y"].to_numpy()
        for mid, res in (("chronos2-zs", ch), ("emp-daily", ed),
                         ("always-zero", {.5: np.zeros(len(sids)),
                                          .85: np.zeros(len(sids))})):
            rows.append(pd.DataFrame({
                "variant": mid, "series_id": sids, "origin": origin,
                "split": "validation", "y": y,
                "q50": res[.5], "q85": res[.85], "w": 1.0}))
        print(f"  {origin:%Y-%m-%d}  n={len(sids):<4} "
              f"neg={flags.get('neg_share', 0):.3f} "
              f"q50z={flags['q50_zero_share']:.3f}  ({time.time() - t0:.0f}s)")

    if not rows:
        print("⚠ 无预测输出")
        chk.note("skip_reason", "no_predictions_produced")
        chk.finish(ART / "checks" / "daily_validation.json")
        return chk.exit_code

    pred = pd.concat(rows, ignore_index=True)
    pred.to_parquet(ART / "predictions_validation.parquet", index=False)

    y_min = float(cfg.metric["y_min"])
    chk.assert_true("q50 <= q85", bool((pred.q50 <= pred.q85).all()))
    chk.assert_true("预测非负有限",
                    bool(np.isfinite(pred[["q50", "q85"]].to_numpy()).all()
                         and (pred[["q50", "q85"]] >= 0).all().all()))

    # 指标表（描述性，非排名）
    print("\n" + "=" * 66)
    print("%-14s %-10s %-11s %-11s %-8s" % ("model", "NPL", "cov50_pos", "cov85_pos", "n"))
    for v in ("chronos2-zs", "emp-daily", "always-zero"):
        s = pred[pred.variant == v]
        if len(s) == 0:
            continue
        r = evaluate_slice(s.y, s.q50, s.q85, s.w, y_min)
        print("%-14s %-10.5f %-11.4f %-11.4f %-8d" % (
            v, r.npl, r.cov_50_pos, r.cov_85_pos, r.n))

    # 不做配对 bootstrap（验证窗样本量不支撑模型排名）
    print(f"\n⚠ 验证窗仅 {target.series_id.nunique()} 序列 / "
          f"{len(target)} 完整周行，不足以做配对 bootstrap 模型比较")
    print("  定位：机制演示（管线可运行 + 结果合理），非模型排名")

    chk.n_rows = len(pred)
    chk.note("wall_clock_sec", round(time.time() - t0, 1))
    chk.note("role", "MECHANISM_DEMONSTRATOR_ONLY")
    chk.note("n_complete_weeks_validation", int(len(target)))
    chk.finish(ART / "checks" / "daily_validation.json")
    print(f"\n总耗时 {time.time() - t0:.0f}s, 状态: {chk.status}")
    return chk.exit_code


if __name__ == "__main__":
    sys.exit(main())
