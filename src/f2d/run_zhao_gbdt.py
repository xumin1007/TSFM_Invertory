"""Zhao 日级：把 LightGBM 加入正式对比。

与 run_zhao_daily 的三臂共走**同一条聚合路径**（日级 21 点分位网格 -> 逐步
卷积 -> 周级分位数），故比较落在模型上而非聚合方式上（07 §2.3.3）。

  gbdt-lean   21 分位点 LightGBM，特征仅用序列自身滞后/滚动统计
              —— 与 Chronos-2 零样本的信息集对齐（06 §6.1）
  gbdt-rich   在 lean 之上加结构化特征（库存快照、在订/ETA、静态属性、类别）

**gbdt-rich 与 chronos2-zs 不构成模型比较。** 它多了 TSFM 根本看不到的信息，
两者之差是**信息集之差**，不是模型能力之差。故其结论必须与 lean 分开陈述，
且不得用于支持或否定「TSFM 优于 GBDT」这一命题。它回答的是另一个问题：
*在这个数据集上，结构化的库存/供应信息还能带来多少预测增益。*

**结构化特征的时间粒度是月，origin 是周。** 月级面板在月初 m 处的特征只用
< m 的数据，故对任一 origin $o \\ge m$ 都是可观察的（§2.4.1）；本脚本取
$m = $ 不晚于 $o$ 的最近月初。代价是最多 30 天的陈旧度，对所有 rich 特征
一致，并记入 `run_manifest`。**不使用 $o$ 当月之后的月级行**。

**为何是「直接多步 + horizon 作特征」**：递归多步会把第 1 步的误差喂回特征，
与 Chronos-2 的一次性多步输出不可比；而每个 h 各训 21 个模型则是 147 个模型。
折中是把 h 作为特征、其余特征在 origin 处**全部冻结**，得到 21 个模型且每步
预测都只用 origin 前的信息 —— 与 TSFM 的信息边界一致。

用法:  PYTHONPATH=src python -m f2d.run_zhao_gbdt [--n-series 2000]
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
from .encoding import VocabStore
from .metrics import evaluate_slice
from .models.chronos import NATIVE_LEVELS
from .models.gbdt_grid import LEAN_FEATURES, QuantileGridGBDT, make_lean_features
from .uncertainty import paired_bootstrap, report

ART = cfgmod.ARTIFACT_DIR / "zhao_daily"
HORIZON = 7
VMAX = 300

TRAIN_ORIGINS = pd.date_range("2019-02-04", "2019-06-24", freq="7D")
VALID_ORIGINS = pd.date_range("2019-07-01", "2019-08-26", freq="7D")


def build_rich_block(raw, vocab_frozen_on: str = "2019-06-30"):
    """月级结构化特征块，键为 (sku_ID, month)。

    直接复用月级面板，避免在日级重算一遍口径不同的版本 —— 若两条路径各自
    实现「在订量」「ETA 误差」，比较就掺入了实现差异。
    """
    panel, _ = zhao.build_panel(raw)
    cols = ["sku_ID", "month"] + zhao.FEATURES_NUM + zhao.CAT_COLS
    block = panel[[c for c in cols if c in panel.columns]].copy()
    return block


RICH_NUM = [c for c in zhao.FEATURES_NUM
            # 目标自身的月级滞后与日级 lean 特征重复，且粒度更粗，剔除
            if c not in ("sales_lag1", "sales_roll3", "sales_roll6",
                         "sales_roll6_std")]
RICH_CAT = zhao.CAT_COLS


def _stack(feat: pd.DataFrame, origins, sids_by_origin, with_y: bool,
           rich: pd.DataFrame | None = None, sku_of: dict | None = None):
    """把 (origin, h) 展开成建模行。特征在 origin 处冻结，只有 h 变化。"""
    idx = feat.set_index(["series_id", "d"])
    ridx = rich.set_index(["sku_ID", "month"]) if rich is not None else None
    out = []
    for o in origins:
        sids = sids_by_origin.get(o)
        if sids is None or not len(sids):
            continue
        key = pd.MultiIndex.from_product([sids, [o]])
        base = idx.reindex(key)[LEAN_FEATURES].dropna(how="all")
        if not len(base):
            continue
        base_sids = base.index.get_level_values(0).to_numpy()
        if ridx is not None:
            m = pd.Timestamp(o).to_period("M").to_timestamp()   # 不晚于 o 的月初
            assert m <= o, "月级特征块越过 origin"
            skus = [sku_of[s] for s in base_sids]
            rb = ridx.reindex(pd.MultiIndex.from_arrays(
                [skus, np.repeat(m, len(skus))]))
            for c in RICH_NUM + RICH_CAT:
                base[c] = rb[c].to_numpy()
        for h in range(HORIZON):
            blk = base.copy()
            blk["h"] = h
            blk["series_id"] = base_sids
            blk["origin"] = o
            if with_y:
                tgt = idx.reindex(pd.MultiIndex.from_arrays(
                    [base_sids, np.repeat(o + pd.Timedelta(days=h), len(base_sids))]))["y"]
                blk["y"] = tgt.to_numpy()
            out.append(blk.reset_index(drop=True))
    df = pd.concat(out, ignore_index=True)
    return df.dropna(subset=["y"]) if with_y else df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=2000)
    args = ap.parse_args(argv)

    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    chk = CheckResult(step_id="zhao_daily_gbdt", dataset="zhao", seed=SEED_BASE)
    cfg = cfgmod.load("zhao")
    y_min = float(cfg.metric["y_min"])

    daily, _ = zhao.build_daily_panel(zhao.load_raw())
    weekly = zhao.aggregate_to_period(daily, "W")

    # 与 run_zhao_daily 完全相同的抽样，保证两次运行可配对比较
    target = weekly[weekly.origin.isin(VALID_ORIGINS)][["series_id", "origin", "y"]]
    rng = np.random.default_rng(SEED_BASE)
    pool = np.asarray(sorted(set(target.series_id)))
    keep = rng.choice(pool, size=min(args.n_series, len(pool)), replace=False)
    daily = daily[daily.series_id.isin(keep)]
    target = target[target.series_id.isin(keep)]

    feat = make_lean_features(daily)
    print(f"特征表 {len(feat)} 行 ({time.time() - t0:.0f}s)")

    # context 门槛与 chronos 臂一致：origin 前至少 30 天
    hist_len = (daily.groupby("series_id")["d"].min()
                .rename("first_d").reset_index())
    fd = dict(zip(hist_len.series_id, hist_len.first_d))

    def elig(o, sids):
        return np.array([s for s in sids if (o - fd[s]).days >= 30])

    all_sids = np.asarray(sorted(set(daily.series_id)))
    tr_sids = {o: elig(o, all_sids) for o in TRAIN_ORIGINS}
    va_sids = {o: elig(o, target[target.origin == o].series_id.to_numpy())
               for o in VALID_ORIGINS}

    raw = zhao.load_raw()
    rich = build_rich_block(raw)
    sku_of = dict(zip(daily.series_id, daily.sku_ID))
    print(f"结构化特征块 {len(rich)} 行，{len(RICH_NUM)} 数值 + {len(RICH_CAT)} 类别")

    train = _stack(feat, TRAIN_ORIGINS, tr_sids, True, rich, sku_of)
    valid = _stack(feat, VALID_ORIGINS, va_sids, False, rich, sku_of)
    print(f"训练 {len(train)} 行 / 验证 {len(valid)} 行 ({time.time() - t0:.0f}s)")
    chk.assert_true("训练窗不越界", bool(train.origin.max() < VALID_ORIGINS[0]))

    # 词表只由训练窗构建并冻结；验证窗未见值 -> __UNK__（§7.1）
    vs = VocabStore.fit(train, RICH_CAT, frozen_on=str(TRAIN_ORIGINS[-1].date()))
    vs.save(ART / "vocab_rich.json")
    unk = vs.unk_rate(valid)
    chk.note("unk_rate_validation", {k: round(v, 4) for k, v in unk.items()})
    print("验证窗 __UNK__ 率:", {k: round(v, 4) for k, v in unk.items()})
    train, valid = vs.transform(train), vs.transform(valid)

    arms = {
        "gbdt-lean": LEAN_FEATURES + ["h"],
        "gbdt-rich": LEAN_FEATURES + ["h"] + RICH_NUM + RICH_CAT,
    }
    rows = []
    for arm, features in arms.items():
        model = QuantileGridGBDT(features=features).fit(train)
        grid = model.predict_grid(valid)                   # (n_rows, 21)
        grid = np.round(np.clip(grid, 0.0, None))          # 与 QuantileRepair 一致
        grid = np.maximum.accumulate(grid, axis=1)
        print(f"{arm}: {len(features)} 特征, 21 个分位模型完毕 ({time.time() - t0:.0f}s)")

        for o in VALID_ORIGINS:
            sel = (valid.origin == o).to_numpy()
            sub = valid[sel]
            if not len(sub):
                continue
            sids = sub[sub.h == 0].series_id.to_numpy()
            g = grid[sel]
            per_step = [g[(sub.h == h).to_numpy()] for h in range(HORIZON)]
            if any(len(p) != len(sids) for p in per_step):
                raise RuntimeError(f"origin {o}: 各 h 的行数不齐")
            r = convolve_varying(NATIVE_LEVELS, per_step, taus=(.5, .85), vmax=VMAX)
            y = target[target.origin == o].set_index("series_id").loc[sids, "y"].to_numpy()
            rows.append(pd.DataFrame({
                "variant": arm, "series_id": sids, "month": o,
                "split": "validation", "y": y,
                "q50": r[.5], "q85": r[.85], "w": 1.0}))

    gb = pd.concat(rows, ignore_index=True)
    prev = pd.read_parquet(ART / "predictions_validation.parquet")
    pred = pd.concat([prev[~prev.variant.isin(arms)], gb], ignore_index=True)

    # 三臂与 GBDT 必须落在同一组 (series_id, origin) 上，否则配对无效
    keys = {v: set(map(tuple, pred[pred.variant == v][["series_id", "month"]].to_numpy()))
            for v in pred.variant.unique()}
    common = set.intersection(*keys.values())
    chk.note("paired_rows", len(common))
    pred = pred[[k in common for k in
                 map(tuple, pred[["series_id", "month"]].to_numpy())]].reset_index(drop=True)
    pred = pred.sort_values(["variant", "month", "series_id"]).reset_index(drop=True)
    pred.to_parquet(ART / "predictions_validation.parquet", index=False)

    print("\n" + "=" * 66)
    print("%-14s %-10s %-11s %-11s %-8s" % ("model", "NPL", "cov50_pos", "cov85_pos", "n"))
    order = ["chronos2-zs", "gbdt-rich", "gbdt-lean", "emp-daily", "always-zero"]
    for v in order:
        s = pred[pred.variant == v]
        r = evaluate_slice(s.y, s.q50, s.q85, s.w, y_min)
        print("%-14s %-10.5f %-11.4f %-11.4f %-8d" % (
            v, r.npl, r.cov_50_pos, r.cov_85_pos, r.n))

    for base in ("emp-daily", "gbdt-lean", "gbdt-rich"):
        print(f"\n配对 bootstrap（基准 = {base}）")
        cis = paired_bootstrap(pred, base, [v for v in order if v != base],
                               y_min=y_min, split="validation")
        print(report(cis)[["variant", "delta", "lo", "hi", "verdict"]]
              .round(5).to_string(index=False))

    chk.n_rows = len(pred)
    chk.note("wall_clock_sec", round(time.time() - t0, 1))
    chk.finish(ART / "checks" / "gbdt_run.json")
    return chk.exit_code


if __name__ == "__main__":
    sys.exit(main())
