"""Zhao 日级：DeepAR + TFT 深度概率预测基线。

与 run_zhao_daily 共走**同一条聚合路径**（日级 21 点分位网格 -> 逐步
卷积 -> 周级分位数），故比较落在模型上而非聚合方式上。

  deepar    DeepAR (NeuralForecast)，2层LSTM，仅用历史销量
  tft       Temporal Fusion Transformer (NeuralForecast)，仅用历史销量

信息集与 chronos2-zs / emp-daily 对齐：仅用历史销量，无结构化特征。
训练一次（用验证窗前的数据），对每个验证 origin 分别 predict。

用法:  PYTHONPATH=src python -m f2d.run_zhao_deepprob [--n-series 2000]
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings

import numpy as np
import pandas as pd

from . import config as cfgmod
from .aggregation import convolve_varying
from .checks import CheckResult
from .conventions import SEED_BASE
from .datasets import zhao
from .metrics import evaluate_slice
from .models.chronos import NATIVE_LEVELS
from .uncertainty import paired_bootstrap, report

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ART = cfgmod.ARTIFACT_DIR / "zhao_daily"
HORIZON = 7
VMAX = 300

VALID_ORIGINS = pd.date_range("2019-07-01", "2019-08-26", freq="7D")


def _train_model(model_name: str, train_df: pd.DataFrame, horizon: int,
                 max_steps: int, input_size: int, accelerator: str):
    """训练一个 NeuralForecast 模型，返回 nf 对象。"""
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import DeepAR, TFT

    levels = list(NATIVE_LEVELS)
    loss = MQLoss(quantiles=levels)
    valid_loss = MQLoss(quantiles=levels)

    common_kw = dict(
        h=horizon, loss=loss, valid_loss=valid_loss,
        max_steps=max_steps, input_size=input_size,
        random_seed=SEED_BASE, accelerator=accelerator,
        enable_progress_bar=True, enable_model_summary=False,
    )

    if model_name == "deepar":
        model = DeepAR(lstm_hidden_size=64, lstm_n_layers=2,
                       learning_rate=1e-3, batch_size=64, **common_kw)
    elif model_name == "tft":
        model = TFT(hidden_size=64, n_head=4,
                    learning_rate=1e-3, batch_size=64, **common_kw)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=train_df)
    return nf


def _predict_origin(nf, daily: pd.DataFrame, origin, sids: list,
                    horizon: int) -> dict[str, np.ndarray]:
    """对一个 origin 的所有序列预测，返回 {sid: (horizon, 21) grid}。"""
    hist = daily[(daily.d < origin) & (daily.series_id.isin(sids))]
    rows = []
    valid_sids = []
    for sid, grp in hist.groupby("series_id"):
        arr = grp.sort_values("d")
        if len(arr) < 30:
            continue
        valid_sids.append(sid)
        rows.append(pd.DataFrame({
            "unique_id": str(sid),
            "ds": arr.d.values,
            "y": arr.y.values.astype(float),
        }))
    if not rows:
        return {}

    pred_df = pd.concat(rows, ignore_index=True)
    fc = nf.predict(df=pred_df).reset_index()

    meta_cols = {"unique_id", "ds", "index"}
    q_cols = [c for c in fc.columns if c not in meta_cols]

    result = {}
    for sid, grp in fc.groupby("unique_id"):
        grp = grp.sort_values("ds")
        grid = grp[q_cols].values[:horizon]  # (horizon, 21)
        if grid.shape[0] < horizon:
            continue
        grid = np.clip(grid, 0.0, None)
        for h in range(grid.shape[0]):
            grid[h] = np.maximum.accumulate(grid[h])
        result[str(sid)] = grid

    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--input-size", type=int, default=60)
    ap.add_argument("--accelerator", default="auto")
    ap.add_argument("--models", nargs="+", default=["deepar", "tft"],
                    choices=["deepar", "tft"])
    args = ap.parse_args(argv)

    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    chk = CheckResult(step_id="zhao_deepprob", dataset="zhao", seed=SEED_BASE)
    cfg = cfgmod.load("zhao")

    daily, audit = zhao.build_daily_panel(zhao.load_raw())

    weekly = zhao.aggregate_to_period(daily, "W")
    target = weekly[weekly.origin.isin(VALID_ORIGINS)][["series_id", "origin", "y"]]

    rng = np.random.default_rng(SEED_BASE)
    pool = np.asarray(sorted(set(target.series_id)))
    keep = rng.choice(pool, size=min(args.n_series, len(pool)), replace=False)
    target = target[target.series_id.isin(keep)].reset_index(drop=True)
    all_sids = sorted(set(target.series_id))
    print(f"验证窗: {len(VALID_ORIGINS)} origins, {len(all_sids)} 序列, "
          f"{len(target)} 个 (序列,origin) 对")

    # 构建训练 DataFrame：用验证窗前的所有数据
    train_cutoff = VALID_ORIGINS[0]
    train_data = daily[(daily.d < train_cutoff) & (daily.series_id.isin(all_sids))]
    # 序列需要 >= input_size + horizon + val_size 才能训练
    min_len = args.input_size + HORIZON + 1
    train_rows = []
    for sid, grp in train_data.groupby("series_id"):
        arr = grp.sort_values("d")
        if len(arr) >= min_len:
            train_rows.append(pd.DataFrame({
                "unique_id": str(sid),
                "ds": arr.d.values,
                "y": arr.y.values.astype(float),
            }))
    train_df = pd.concat(train_rows, ignore_index=True)
    valid_sids = sorted(set(train_df.unique_id.unique()) & set(str(s) for s in all_sids))
    print(f"训练序列: {len(valid_sids)} (>= {min_len}天)")

    all_rows = []

    for model_name in args.models:
        print(f"\n{'='*60}")
        print(f"训练 {model_name.upper()} (max_steps={args.max_steps}) ...")
        mt0 = time.time()

        nf = _train_model(model_name, train_df, HORIZON,
                          args.max_steps, args.input_size, args.accelerator)
        print(f"训练完成: {time.time() - mt0:.0f}s")

        for origin in VALID_ORIGINS:
            cur = target[(target.origin == origin) &
                         (target.series_id.isin(all_sids))]
            if not len(cur):
                continue

            sids = sorted(set(cur.series_id))
            grids = _predict_origin(nf, daily, origin, sids, HORIZON)

            pred_sids = np.array([s for s in sids if str(s) in grids])
            if not len(pred_sids):
                continue

            per_step = []
            for h in range(HORIZON):
                step_grid = np.array([grids[str(s)][h] for s in pred_sids])
                per_step.append(step_grid)

            res = convolve_varying(NATIVE_LEVELS, per_step,
                                  taus=(.5, .85), vmax=VMAX)

            y = cur.set_index("series_id").loc[pred_sids, "y"].to_numpy()
            all_rows.append(pd.DataFrame({
                "variant": model_name, "series_id": pred_sids,
                "month": origin, "split": "validation",
                "y": y, "q50": res[.5], "q85": res[.85], "w": 1.0,
            }))

            print(f"  {origin:%Y-%m-%d}  n={len(pred_sids):<5}  "
                  f"({time.time() - mt0:.0f}s)")

        print(f"{model_name.upper()} 总耗时: {time.time() - mt0:.0f}s")

    if not all_rows:
        print("无预测结果")
        return 1

    pred = pd.concat(all_rows, ignore_index=True)
    pred.to_parquet(ART / "predictions_deepprob_validation.parquet", index=False)

    # 联合比较：加载已有基线
    base_path = ART / "predictions_validation.parquet"
    if base_path.exists():
        base = pd.read_parquet(base_path)
        combined = pd.concat([base, pred], ignore_index=True)
    else:
        combined = pred

    y_min = float(cfg.metric["y_min"])

    print("\n" + "=" * 66)
    print("%-14s %-10s %-11s %-11s %-8s" % ("model", "NPL", "cov50_pos", "cov85_pos", "n"))
    for v in sorted(combined.variant.unique()):
        s = combined[combined.variant == v]
        r = evaluate_slice(s.y, s.q50, s.q85, s.w, y_min)
        print("%-14s %-10.5f %-11.4f %-11.4f %-8d" % (
            v, r.npl, r.cov_50_pos, r.cov_85_pos, r.n))

    # 配对 bootstrap：只用 deepprob 模型与 emp-daily 中重叠的序列-origin 对
    print("\n配对 bootstrap（基准 = emp-daily，限重叠行）")
    if "emp-daily" in combined.variant.values:
        emp = combined[combined.variant == "emp-daily"]
        emp_keys = set(zip(emp.series_id, emp.month))
        for mn in args.models:
            m_df = combined[combined.variant == mn]
            m_keys = set(zip(m_df.series_id, m_df.month))
            overlap = emp_keys & m_keys
            if not overlap:
                print(f"  {mn}: 无重叠行")
                continue
            mask_e = emp.apply(lambda r: (r.series_id, r.month) in overlap, axis=1)
            mask_m = m_df.apply(lambda r: (r.series_id, r.month) in overlap, axis=1)
            sub = pd.concat([emp[mask_e], m_df[mask_m]], ignore_index=True)
            cis = paired_bootstrap(sub, "emp-daily", [mn],
                                   y_min=y_min, split="validation")
            print(report(cis)[["variant", "delta", "lo", "hi", "verdict"]]
                  .round(5).to_string(index=False))

    chk.n_rows = len(pred)
    chk.note("wall_clock_sec", round(time.time() - t0, 1))
    chk.note("models", args.models)
    chk.finish(ART / "checks" / "deepprob_run.json")
    return chk.exit_code


if __name__ == "__main__":
    sys.exit(main())
