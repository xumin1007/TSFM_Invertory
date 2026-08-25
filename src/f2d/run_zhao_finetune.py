"""Chronos-2 微调臂。对应 docs/06_model_hyperparams.md §5。

  chronos2-ft-lora   LoRA 微调，lr=1e-5    （主微调臂）
  chronos2-ft-full   全参微调，lr=1e-6     （微调上限，与 LoRA 对照）

**对 §5.1 的一处刻意偏离，须与结果一并陈述。** 规格写的是 `validation_inputs`
取**验证窗**做 checkpoint 选择。那个设计假定最终只在测试窗报数。本轮实验在
验证窗上报数，若再用验证窗选 checkpoint，报出的微调增益就是乐观偏倚的。
故本脚本从**训练窗末尾**切出内部验证段（最后 7 天）用于 checkpoint 选择，
验证窗在微调的任何阶段都不出现。断言固化在 `chk` 中。

代价：内部验证段与真实验证窗的分布可能不同（6 月末 vs 7-8 月），checkpoint
选择因此不是对目标分布最优的。这个代价是**必须付的** —— 反过来会污染结论。

评价路径与 run_zhao_daily 完全一致（日级 21 点网格 -> 逐日卷积 -> 周级），
故与 `chronos2-zs` 逐行可配对。

用法:  PYTHONPATH=src python -m f2d.run_zhao_finetune [--mode lora|full|both]
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
FT_DIR = cfgmod.ARTIFACT_DIR / "zhao_finetune"
HORIZON = 7
VMAX = 300
CONTEXT_LENGTH = 400
MIN_PAST = max(2 * HORIZON, 28)
NUM_STEPS = 1000
BATCH_SIZE = 32

TRAIN_END = pd.Timestamp("2019-06-30")          # 训练窗末（configs/zhao.yaml）
VALID_ORIGINS = pd.date_range("2019-07-01", "2019-08-26", freq="7D")


def _windows(ft_horizon: int):
    """内部验证段长度 = 微调视界。返回 (训练截止, 内部验证目标起点)。

    ft_horizon 越大，训练输入被砍得越多（H=32 时训练窗少一个月）。这是
    「不用验证窗选 checkpoint」这一决定的直接代价，须与结果一并陈述。
    """
    ivt = TRAIN_END - pd.Timedelta(days=ft_horizon - 1)
    return ivt - pd.Timedelta(days=1), ivt

ARM_PARAMS = {
    "lora": dict(finetune_mode="lora", learning_rate=1e-5),
    "full": dict(finetune_mode="full", learning_rate=1e-6),
}


def _series_dict(daily, upto):
    """截断到 upto（含）的逐序列日值。"""
    h = daily[daily.d <= upto]
    return {s: g.sort_values("d").y.to_numpy(float) for s, g in h.groupby("series_id")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=2000)
    ap.add_argument("--mode", default="lora", choices=("lora", "full", "both"))
    ap.add_argument("--ft-horizon", type=int, default=HORIZON,
                    help="微调时的 prediction_length。默认 7（=评价视界）；"
                         "决策层需要 32，用 --ft-horizon 32 消除该失配。")
    ap.add_argument("--train-cutoff", default=None,
                    help="覆盖训练输入截止日（YYYY-MM-DD）。用于构造「同样"
                         "截短数据但 H 不同」的控制臂，拆开数据量与视界。")
    ap.add_argument("--arm-suffix", default=None,
                    help="覆盖臂名后缀，供控制臂登记为独立臂。")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args(argv)

    import torch
    from chronos import BaseChronosPipeline

    t0 = time.time()
    FT_DIR.mkdir(parents=True, exist_ok=True)
    chk = CheckResult(step_id="zhao_finetune", dataset="zhao", seed=SEED_BASE)
    cfg = cfgmod.load("zhao")
    y_min = float(cfg.metric["y_min"])

    daily, _ = zhao.build_daily_panel(zhao.load_raw())
    weekly = zhao.aggregate_to_period(daily, "W")
    target = weekly[weekly.origin.isin(VALID_ORIGINS)][["series_id", "origin", "y"]]

    # 与 run_zhao_daily 完全相同的抽样，保证逐行可配对
    rng = np.random.default_rng(SEED_BASE)
    pool = np.asarray(sorted(set(target.series_id)))
    keep = rng.choice(pool, size=min(args.n_series, len(pool)), replace=False)
    target = target[target.series_id.isin(keep)].reset_index(drop=True)
    daily = daily[daily.series_id.isin(keep)]

    # --- 时间隔离（§5.1，可断言）---
    fh = args.ft_horizon
    train_cutoff, inner_start = _windows(fh)
    if args.train_cutoff:
        train_cutoff = pd.Timestamp(args.train_cutoff)
        inner_start = train_cutoff + pd.Timedelta(days=1)
        chk.note("train_cutoff_overridden", str(train_cutoff.date()))
    min_past = max(2 * fh, 28)
    tr = _series_dict(daily, train_cutoff)
    iv = _series_dict(daily, TRAIN_END)
    tr = {s: v for s, v in tr.items() if len(v) >= min_past + fh}
    iv = {s: v for s, v in iv.items() if len(v) >= min_past + fh}
    chk.assert_true("微调输入不越过内部验证目标段",
                    bool(train_cutoff < inner_start))
    chk.assert_true("内部验证不越过验证窗",
                    bool(TRAIN_END < VALID_ORIGINS[0]))
    chk.note("ft_horizon", fh)
    chk.note("finetune_max_ts", str(train_cutoff.date()))
    chk.note("inner_val_max_ts", str(TRAIN_END.date()))
    chk.note("n_train_series", len(tr))
    print(f"微调视界 H={fh}, min_past={min_past}")
    print(f"微调输入 {len(tr)} 条序列（截至 {train_cutoff:%m-%d}），"
          f"内部验证 {len(iv)} 条（截至 {TRAIN_END:%m-%d}，目标段 "
          f"{inner_start:%m-%d}~{TRAIN_END:%m-%d}）")

    train_inputs = [torch.tensor(v, dtype=torch.float32) for v in tr.values()]
    val_inputs = [torch.tensor(v, dtype=torch.float32) for v in iv.values()]

    modes = ["lora", "full"] if args.mode == "both" else [args.mode]
    suffix = (args.arm_suffix if args.arm_suffix is not None
              else ("" if fh == HORIZON else f"-h{fh}"))
    arms = [f"chronos2-ft-{m}{suffix}" for m in modes]

    rows = []
    for arm, mode in zip(arms, modes):
        p = ARM_PARAMS[mode]
        print(f"\n=== {arm}: {p['finetune_mode']} lr={p['learning_rate']} "
              f"steps={NUM_STEPS} bs={BATCH_SIZE} ===")
        pipe = BaseChronosPipeline.from_pretrained(
            BASE_CHECKPOINT, device_map=args.device, torch_dtype=torch.float32)
        torch.manual_seed(SEED_BASE)
        ft = pipe.fit(
            inputs=train_inputs, validation_inputs=val_inputs,
            prediction_length=fh, context_length=CONTEXT_LENGTH,
            num_steps=NUM_STEPS, batch_size=BATCH_SIZE, min_past=min_past,
            output_dir=FT_DIR / arm, finetuned_ckpt_name="ft",
            remove_printer_callback=True, disable_data_parallel=True, **p)
        print(f"微调完毕 ({time.time() - t0:.0f}s)")

        for origin in VALID_ORIGINS:
            cur = target[target.origin == origin]
            sids = cur.series_id.to_numpy()
            hist = daily[(daily.d < origin) & (daily.series_id.isin(sids))]
            ctx = {s: g.y.to_numpy() for s, g in hist.groupby("series_id")}
            sids = np.array([s for s in sids if len(ctx.get(s, [])) >= 30])
            if not len(sids):
                continue
            cur = cur[cur.series_id.isin(sids)]
            q, _ = ft.predict_quantiles(
                [torch.tensor(ctx[s], dtype=torch.float32) for s in sids],
                prediction_length=HORIZON, quantile_levels=list(NATIVE_LEVELS),
                batch_size=args.batch_size)
            g, _ = QuantileRepair()(to_grid(q))
            g = g.reshape(len(sids), HORIZON, -1)
            r = convolve_varying(NATIVE_LEVELS, [g[:, h, :] for h in range(HORIZON)],
                                 taus=(.5, .85), vmax=VMAX)
            rows.append(pd.DataFrame({
                "variant": arm, "series_id": sids, "month": origin,
                "split": "validation",
                "y": cur.set_index("series_id").loc[sids, "y"].to_numpy(),
                "q50": r[.5], "q85": r[.85], "w": 1.0}))
            print(f"  {origin:%Y-%m-%d} n={len(sids):<5} ({time.time() - t0:.0f}s)")

    ftp = pd.concat(rows, ignore_index=True)
    prev = pd.read_parquet(ART / "predictions_validation.parquet")
    pred = pd.concat([prev[~prev.variant.isin(arms)], ftp], ignore_index=True)

    keys = {v: set(map(tuple, pred[pred.variant == v][["series_id", "month"]].to_numpy()))
            for v in pred.variant.unique()}
    common = set.intersection(*keys.values())
    pred = pred[[k in common for k in
                 map(tuple, pred[["series_id", "month"]].to_numpy())]]
    pred = pred.sort_values(["variant", "month", "series_id"]).reset_index(drop=True)
    pred.to_parquet(ART / "predictions_validation.parquet", index=False)
    chk.note("paired_rows", len(common))

    order = [v for v in (*arms, "chronos2-ft-lora", "chronos2-ft-full", "chronos2-zs",
                         "gbdt-rich", "gbdt-lean", "emp-daily", "always-zero")
             if v in set(pred.variant)]
    order = list(dict.fromkeys(order))
    print("\n" + "=" * 66)
    print("%-18s %-10s %-11s %-11s %-8s" % ("model", "NPL", "cov50_pos", "cov85_pos", "n"))
    for v in order:
        s = pred[pred.variant == v]
        r = evaluate_slice(s.y, s.q50, s.q85, s.w, y_min)
        print("%-18s %-10.5f %-11.4f %-11.4f %-8d" % (
            v, r.npl, r.cov_50_pos, r.cov_85_pos, r.n))

    print("\n配对 bootstrap（基准 = chronos2-zs，即微调的净增量）")
    cis = paired_bootstrap(pred, "chronos2-zs", [v for v in order if v != "chronos2-zs"],
                           y_min=y_min, split="validation")
    print(report(cis)[["variant", "delta", "lo", "hi", "verdict"]]
          .round(5).to_string(index=False))
    for c in cis:
        if c.variant in arms:
            chk.note(f"ft_gain_{c.variant}",
                     [round(c.delta, 5), round(c.lo, 5), round(c.hi, 5), c.significant])

    chk.n_rows = len(pred)
    chk.note("wall_clock_sec", round(time.time() - t0, 1))
    chk.finish(ART / "checks" / "finetune.json")
    return chk.exit_code


if __name__ == "__main__":
    sys.exit(main())
