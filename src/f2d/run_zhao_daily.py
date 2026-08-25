"""Zhao 日级：Chronos-2 零样本 vs 基线，第一次真实运行。

评价在**周级**（日级 q50 因零占比 0.682 退化，见 00 §5.1）。
三个模型共走同一条路径 —— 日级 21 点分位网格 -> 逐步卷积 -> 周级分位数 ——
故比较的是完整管线，聚合误差对各方等同（07 §2.3.3）。

  chronos2-zs   Chronos-2 零样本，7 步日预测
  emp-daily     context 的经验日分位网格（同路径的强基线）
  always-zero   退化对照（00 §5.1 强制项）

只用**验证窗**，不碰测试窗。

用法:  PYTHONPATH=src python -m f2d.run_zhao_daily [--n-series 2000]
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
from .datasets import zhao
from .metrics import evaluate_slice
from .models.chronos import (BASE_CHECKPOINT, NATIVE_LEVELS, QuantileRepair,
                             to_grid)
from .uncertainty import paired_bootstrap, report

ART = cfgmod.ARTIFACT_DIR / "zhao_daily"
HORIZON = 7
VMAX = 300


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=2000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args(argv)

    import torch
    from chronos import BaseChronosPipeline

    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    chk = CheckResult(step_id="zhao_daily_first_run", dataset="zhao", seed=SEED_BASE)
    cfg = cfgmod.load("zhao")

    daily, audit = zhao.build_daily_panel(zhao.load_raw())
    chk.note("panel_audit", {k: audit[k] for k in
                             ("n_rows", "n_sku_kept", "kept_share", "zero_share_after_fill")})

    # 验证窗 origin：周一，与 configs/zhao.yaml 的月度验证窗（7-8 月）对齐
    origins = pd.date_range("2019-07-01", "2019-08-26", freq="7D")
    weekly = zhao.aggregate_to_period(daily, "W")
    target = weekly[weekly.origin.isin(origins)][["series_id", "origin", "y"]]

    rng = np.random.default_rng(SEED_BASE)
    pool = np.asarray(sorted(set(target.series_id)))
    keep = rng.choice(pool, size=min(args.n_series, len(pool)), replace=False)
    target = target[target.series_id.isin(keep)].reset_index(drop=True)
    print(f"验证窗: {len(origins)} 个 origin, {target.series_id.nunique()} 条序列, "
          f"{len(target)} 个 (序列,origin) 对")

    pipe = BaseChronosPipeline.from_pretrained(
        BASE_CHECKPOINT, device_map=args.device, torch_dtype=torch.float32)
    repair = QuantileRepair()          # clip + round，splice 关闭
    rows = []

    for origin in origins:
        cur = target[target.origin == origin]
        if not len(cur):
            continue
        sids = cur.series_id.to_numpy()
        hist = daily[(daily.d < origin) & (daily.series_id.isin(sids))]
        ctx = {s: g.y.to_numpy() for s, g in hist.groupby("series_id")}
        sids = np.array([s for s in sids if len(ctx.get(s, [])) >= 30])
        cur = cur[cur.series_id.isin(sids)]
        if not len(sids):
            continue

        # --- chronos2-zs ---
        q, _ = pipe.predict_quantiles(
            [torch.tensor(ctx[s], dtype=torch.float32) for s in sids],
            prediction_length=HORIZON, quantile_levels=list(NATIVE_LEVELS),
            batch_size=args.batch_size)
        grid = to_grid(q)                                   # (n*H, 21)
        grid, flags = repair(grid)
        per_step = [grid.reshape(len(sids), HORIZON, -1)[:, h, :] for h in range(HORIZON)]
        ch = convolve_varying(NATIVE_LEVELS, per_step, taus=(.5, .85), vmax=VMAX)

        # --- emp-daily：context 经验分位网格，同一条聚合路径 ---
        emp = np.array([np.quantile(ctx[s], NATIVE_LEVELS, method="inverted_cdf")
                        for s in sids], float)
        ed = convolve_varying(NATIVE_LEVELS, [emp] * HORIZON, taus=(.5, .85), vmax=VMAX)

        y = cur.set_index("series_id").loc[sids, "y"].to_numpy()
        for mid, res in (("chronos2-zs", ch), ("emp-daily", ed),
                         ("always-zero", {.5: np.zeros(len(sids)), .85: np.zeros(len(sids))})):
            rows.append(pd.DataFrame({
                "variant": mid, "series_id": sids, "month": origin, "split": "validation",
                "y": y, "q50": res[.5], "q85": res[.85], "w": 1.0}))
        print(f"  {origin:%Y-%m-%d}  n={len(sids):<5} 负值率={flags.get('neg_share', 0):.3f} "
              f"q50为零率={flags['q50_zero_share']:.3f}  ({time.time() - t0:.0f}s)")

    pred = pd.concat(rows, ignore_index=True)
    pred.to_parquet(ART / "predictions_validation.parquet", index=False)

    y_min = float(cfg.metric["y_min"])
    chk.assert_true("q50 <= q85", bool((pred.q50 <= pred.q85).all()))
    chk.assert_true("预测非负有限",
                    bool(np.isfinite(pred[["q50", "q85"]].to_numpy()).all()
                         and (pred[["q50", "q85"]] >= 0).all().all()))

    print("\n" + "=" * 66)
    print("%-14s %-10s %-11s %-11s %-8s" % ("model", "NPL", "cov50_pos", "cov85_pos", "n"))
    for v in ("chronos2-zs", "emp-daily", "always-zero"):
        s = pred[pred.variant == v]
        r = evaluate_slice(s.y, s.q50, s.q85, s.w, y_min)
        print("%-14s %-10.5f %-11.4f %-11.4f %-8d" % (v, r.npl, r.cov_50_pos, r.cov_85_pos, r.n))

    print("\n配对 bootstrap（基准 = emp-daily）")
    cis = paired_bootstrap(pred, "emp-daily", ["chronos2-zs", "always-zero"],
                           y_min=y_min, split="validation")
    print(report(cis)[["variant", "delta", "lo", "hi", "verdict"]].round(5).to_string(index=False))

    chk.n_rows = len(pred)
    chk.note("wall_clock_sec", round(time.time() - t0, 1))
    chk.finish(ART / "checks" / "first_run.json")
    return chk.exit_code


if __name__ == "__main__":
    sys.exit(main())
