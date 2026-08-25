"""Chronos-2 微调 + 库存协变量实验。

将月初库存水位（beginning_inventory）作为 known covariate 传入 Chronos-2，
在整个 context + prediction window 内重复月初值（常数协变量）。

思路来自 Sui et al. (2026)：用少量蒸馏后的协变量增强 TSFM。
我们用库存水位代替促销强度，测试 TSFM 能否利用 ERM-rich 使用的信息。

用法:  PYTHONPATH=src python -m f2d.run_zhao_finetune_cov [--mode full] [--device mps]
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

TRAIN_END = pd.Timestamp("2019-06-30")
VALID_ORIGINS = pd.date_range("2019-07-01", "2019-08-26", freq="7D")


def _windows(ft_horizon: int):
    ivt = TRAIN_END - pd.Timedelta(days=ft_horizon - 1)
    return ivt - pd.Timedelta(days=1), ivt


def _build_inv_lookup(raw) -> dict[tuple[str, str], float]:
    """月初库存 lookup: (sku_ID, month_str) -> beginning_inventory."""
    inv = raw["inventory"]
    keys = list(zip(inv.sku_ID, inv.month.dt.strftime("%Y-%m")))
    vals = inv.beginning_inventory.astype(float).tolist()
    return dict(zip(keys, vals))


def _make_long_df(daily: pd.DataFrame, sids, upto: pd.Timestamp,
                  inv_lookup: dict, sku_of: dict) -> pd.DataFrame:
    """构造 from_data_frame 需要的长格式 DataFrame，含 inv_level 协变量。"""
    h = daily[(daily.series_id.isin(set(sids))) & (daily.d <= upto)].copy()
    h = h.sort_values(["series_id", "d"])

    skus = h.series_id.map(sku_of).fillna("")
    month_keys = h.d.dt.strftime("%Y-%m")
    h["inv_level"] = [inv_lookup.get((s, m), 0.0)
                      for s, m in zip(skus, month_keys)]

    h = h.rename(columns={"series_id": "item_id", "d": "timestamp", "y": "target"})
    return h[["item_id", "timestamp", "target", "inv_level"]]


def _make_prepared_inputs(daily, sids, upto, inv_lookup, sku_of,
                          prediction_length):
    """用 from_data_frame 构造 PreparedInput list。"""
    from chronos.chronos2.preprocess import from_data_frame

    df = _make_long_df(daily, sids, upto, inv_lookup, sku_of)
    return from_data_frame(
        df, target_columns=["target"], prediction_length=prediction_length,
        known_covariates_names=["inv_level"],
        use_target_encoding=False, validate_inputs=True)


def _make_predict_inputs(daily, sids, origin, inv_lookup, sku_of,
                         prediction_length):
    """推理时构造输入：context + future covariate。"""
    from chronos.chronos2.preprocess import from_data_frame

    hist = daily[(daily.series_id.isin(set(sids))) & (daily.d < origin)].copy()
    hist = hist.sort_values(["series_id", "d"])

    skus = hist.series_id.map(sku_of).fillna("")
    month_keys = hist.d.dt.strftime("%Y-%m")
    hist["inv_level"] = [inv_lookup.get((s, m), 0.0)
                         for s, m in zip(skus, month_keys)]
    hist = hist.rename(columns={"series_id": "item_id", "d": "timestamp", "y": "target"})

    # future_df: prediction_length rows per series with inv_level
    future_rows = []
    for s in sids:
        sku = sku_of.get(s, "")
        origin_month = origin.strftime("%Y-%m")
        inv_val = inv_lookup.get((sku, origin_month), 0.0)
        for h in range(prediction_length):
            future_rows.append({
                "item_id": s,
                "timestamp": origin + pd.Timedelta(days=h),
                "inv_level": inv_val,
            })
    future_df = pd.DataFrame(future_rows)

    return from_data_frame(
        hist[["item_id", "timestamp", "target", "inv_level"]],
        target_columns=["target"], prediction_length=prediction_length,
        future_df=future_df,
        use_target_encoding=False, validate_inputs=True)


ARM_PARAMS = {
    "full": dict(finetune_mode="full", learning_rate=1e-6),
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=2000)
    ap.add_argument("--mode", default="full", choices=("full",))
    ap.add_argument("--ft-horizon", type=int, default=HORIZON)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args(argv)

    import torch
    from chronos import BaseChronosPipeline

    t0 = time.time()
    FT_DIR.mkdir(parents=True, exist_ok=True)

    raw = zhao.load_raw()
    daily, _ = zhao.build_daily_panel(raw)
    weekly = zhao.aggregate_to_period(daily, "W")
    target = weekly[weekly.origin.isin(VALID_ORIGINS)][["series_id", "origin", "y"]]

    rng = np.random.default_rng(SEED_BASE)
    pool = np.asarray(sorted(set(target.series_id)))
    keep = rng.choice(pool, size=min(args.n_series, len(pool)), replace=False)
    target = target[target.series_id.isin(keep)].reset_index(drop=True)
    daily = daily[daily.series_id.isin(keep)]

    sku_of = dict(zip(daily.series_id, daily.sku_ID))
    inv_lookup = _build_inv_lookup(raw)

    fh = args.ft_horizon
    train_cutoff, inner_start = _windows(fh)
    min_past = max(2 * fh, 28)

    # 构造 PreparedInput
    tr_sids = sorted(set(
        daily[daily.d <= train_cutoff].groupby("series_id")
        .filter(lambda g: len(g) >= min_past + fh).series_id))
    iv_sids = sorted(set(
        daily[daily.d <= TRAIN_END].groupby("series_id")
        .filter(lambda g: len(g) >= min_past + fh).series_id))

    print(f"微调视界 H={fh}, min_past={min_past}")
    print(f"微调 {len(tr_sids)} 条序列, 内部验证 {len(iv_sids)} 条")
    print("构造 PreparedInput（含 inv_level 协变量）...")

    train_inputs = _make_prepared_inputs(
        daily, tr_sids, train_cutoff, inv_lookup, sku_of, fh)
    val_inputs = _make_prepared_inputs(
        daily, iv_sids, TRAIN_END, inv_lookup, sku_of, fh)
    print(f"PreparedInput 构造完毕 ({time.time() - t0:.0f}s)")

    suffix = f"-cov" if fh == HORIZON else f"-cov-h{fh}"
    arm = f"chronos2-ft-full{suffix}"
    p = ARM_PARAMS["full"]

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

    # 保存 checkpoint 路径
    ft_ckpt = FT_DIR / arm / "ft"
    print(f"Checkpoint: {ft_ckpt}")

    # --- 验证窗推理 ---
    rows = []
    for origin in VALID_ORIGINS:
        cur = target[target.origin == origin]
        sids = cur.series_id.to_numpy()
        hist = daily[(daily.d < origin) & (daily.series_id.isin(sids))]
        ctx_lens = hist.groupby("series_id").size()
        sids = np.array([s for s in sids if ctx_lens.get(s, 0) >= 30])
        if not len(sids):
            continue
        cur = cur[cur.series_id.isin(sids)]

        pred_inputs = _make_predict_inputs(
            daily, sids, origin, inv_lookup, sku_of, HORIZON)

        q, _ = ft.predict_quantiles(
            pred_inputs,
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

    # 合并到已有预测
    pred_path = ART / "predictions_validation.parquet"
    prev = pd.read_parquet(pred_path)
    pred = pd.concat([prev[prev.variant != arm], ftp], ignore_index=True)

    keys = {v: set(map(tuple, pred[pred.variant == v][["series_id", "month"]].to_numpy()))
            for v in pred.variant.unique()}
    common = set.intersection(*keys.values())
    pred = pred[[k in common for k in
                 map(tuple, pred[["series_id", "month"]].to_numpy())]]
    pred = pred.sort_values(["variant", "month", "series_id"]).reset_index(drop=True)
    pred.to_parquet(pred_path, index=False)

    cfg = cfgmod.load("zhao")
    y_min = float(cfg.metric["y_min"])

    order = [arm, "chronos2-ft-full", "chronos2-zs", "emp-daily"]
    order = [v for v in order if v in set(pred.variant)]
    print("\n" + "=" * 66)
    print("%-22s %-10s %-11s %-11s %-8s" % ("model", "NPL", "cov50_pos", "cov85_pos", "n"))
    for v in order:
        s = pred[pred.variant == v]
        r = evaluate_slice(s.y, s.q50, s.q85, s.w, y_min)
        print("%-22s %-10.5f %-11.4f %-11.4f %-8d" % (
            v, r.npl, r.cov_50_pos, r.cov_85_pos, r.n))

    print("\n配对 bootstrap（基准 = chronos2-ft-full）")
    cis = paired_bootstrap(pred, "chronos2-ft-full",
                           [v for v in order if v != "chronos2-ft-full"],
                           y_min=y_min, split="validation")
    print(report(cis)[["variant", "delta", "lo", "hi", "verdict"]]
          .round(5).to_string(index=False))

    print(f"\n总耗时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
