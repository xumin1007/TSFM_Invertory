"""Zhao: Ban & Rudin (2019) ERM baseline — direct newsvendor cost minimization.

ERM 跳过分布估计，直接在 (特征, 周级需求) 上最小化 newsvendor 成本：
  min_q  (1/n) Σ [b·(d_i - q(x_i))⁺ + h·(q(x_i) - d_i)⁺]

这等价于在 critical ratio α = b/(b+h) 处做分位数回归。

与 ETO 路线（TSFM/GBDT 预测日级分布 → 卷积 → 周级分位数）的关键区别：
ERM 直接在周级上操作，省去了日级预测和卷积，是真正的"端到端"方法。

实现：
  erm-linear    线性分位数回归 (statsmodels QuantReg, 对应 Ban & Rudin ERM)
  erm-gbdt      LightGBM 分位数回归在 α 处 (非线性扩展)

两种特征集：lean（仅日级滞后统计）和 rich（加月级结构化特征）。

用法:  PYTHONPATH=src python -m f2d.run_zhao_erm [--n-series 2000]
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config as cfgmod
from .checks import CheckResult
from .conventions import SEED_BASE
from .datasets import zhao
from .metrics import evaluate_slice
from .models.gbdt_grid import LEAN_FEATURES, make_lean_features
from .run_zhao_gbdt import RICH_CAT, RICH_NUM, build_rich_block
from .encoding import VocabStore
from .uncertainty import paired_bootstrap, report

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ART = cfgmod.ARTIFACT_DIR / "zhao_daily"
HORIZON = 7

TRAIN_ORIGINS = pd.date_range("2019-02-04", "2019-06-24", freq="7D")
VALID_ORIGINS = pd.date_range("2019-07-01", "2019-08-26", freq="7D")

ALPHA = 0.85


def _build_weekly_features(daily_feat: pd.DataFrame, weekly: pd.DataFrame,
                           origins, sids, rich=None, sku_of=None):
    """为每个 (origin, series_id) 构造周级建模行。"""
    idx = daily_feat.set_index(["series_id", "d"])
    ridx = rich.set_index(["sku_ID", "month"]) if rich is not None else None

    rows = []
    for o in origins:
        o_sids = sids.get(o)
        if o_sids is None or not len(o_sids):
            continue

        key = pd.MultiIndex.from_product([o_sids, [o]])
        base = idx.reindex(key)[LEAN_FEATURES].dropna(how="all")
        if not len(base):
            continue

        base_sids = base.index.get_level_values(0).to_numpy()

        if ridx is not None and sku_of is not None:
            m = pd.Timestamp(o).to_period("M").to_timestamp()
            skus = [sku_of[s] for s in base_sids]
            rb = ridx.reindex(pd.MultiIndex.from_arrays(
                [skus, np.repeat(m, len(skus))]))
            for c in RICH_NUM + RICH_CAT:
                base[c] = rb[c].to_numpy()

        blk = base.copy()
        blk["series_id"] = base_sids
        blk["origin"] = o

        w = weekly[(weekly.origin == o) & weekly.series_id.isin(base_sids)]
        w_map = dict(zip(w.series_id, w.y))
        blk["y_week"] = [w_map.get(s, np.nan) for s in base_sids]

        rows.append(blk.reset_index(drop=True))

    df = pd.concat(rows, ignore_index=True)
    return df.dropna(subset=["y_week"])


def _erm_linear(X_train, y_train, X_pred, alpha):
    """Ban & Rudin 线性 ERM via statsmodels QuantReg (interior point)."""
    import statsmodels.api as sm
    X_c = sm.add_constant(X_train)
    model = sm.QuantReg(y_train, X_c)
    res = model.fit(q=alpha, max_iter=1000)
    X_pred_c = sm.add_constant(X_pred)
    return X_pred_c @ res.params


def _erm_gbdt(X_train, y_train, X_pred, alpha):
    """非线性 ERM：LightGBM 分位数回归在 α 处。"""
    m = lgb.LGBMRegressor(
        objective="quantile", alpha=float(alpha),
        n_estimators=300, learning_rate=0.05,
        num_leaves=31, min_child_samples=40,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
        lambda_l1=0.1, lambda_l2=1.0,
        seed=SEED_BASE, deterministic=True, force_row_wise=True, verbose=-1)
    m.fit(X_train, y_train)
    return m.predict(X_pred)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    args = ap.parse_args(argv)

    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    chk = CheckResult(step_id="zhao_erm", dataset="zhao", seed=SEED_BASE)
    cfg = cfgmod.load("zhao")
    y_min = float(cfg.metric["y_min"])
    alpha = args.alpha

    daily, _ = zhao.build_daily_panel(zhao.load_raw())
    weekly = zhao.aggregate_to_period(daily, "W")

    target = weekly[weekly.origin.isin(VALID_ORIGINS)][["series_id", "origin", "y"]]
    rng = np.random.default_rng(SEED_BASE)
    pool = np.asarray(sorted(set(target.series_id)))
    keep = rng.choice(pool, size=min(args.n_series, len(pool)), replace=False)
    daily = daily[daily.series_id.isin(keep)]
    target = target[target.series_id.isin(keep)]
    weekly = weekly[weekly.series_id.isin(keep)]

    feat = make_lean_features(daily)
    print(f"特征表 {len(feat)} 行 ({time.time() - t0:.0f}s)")

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
    print(f"结构化特征块 {len(rich)} 行")

    train_df = _build_weekly_features(feat, weekly, TRAIN_ORIGINS, tr_sids, rich, sku_of)
    valid_df = _build_weekly_features(feat, weekly, VALID_ORIGINS, va_sids, rich, sku_of)
    print(f"训练 {len(train_df)} 行 / 验证 {len(valid_df)} 行 ({time.time() - t0:.0f}s)")

    vs = VocabStore.fit(train_df, RICH_CAT, frozen_on=str(TRAIN_ORIGINS[-1].date()))
    train_df, valid_df = vs.transform(train_df), vs.transform(valid_df)

    lean_cols = LEAN_FEATURES
    rich_cols = LEAN_FEATURES + RICH_NUM + RICH_CAT

    arms = {}

    # --- ERM-linear (Ban & Rudin) ---
    for tag, cols in [("erm-linear-lean", lean_cols), ("erm-linear-rich", rich_cols)]:
        Xtr = train_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
        ytr = train_df["y_week"].to_numpy(float)
        Xva = valid_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)

        try:
            q_pred = np.clip(_erm_linear(Xtr, ytr, Xva, alpha), 0, None)
            arms[tag] = q_pred
            print(f"{tag}: QuantReg solved ({time.time() - t0:.0f}s)")
        except Exception as e:
            print(f"{tag}: failed - {e}")

    # --- ERM-GBDT (nonlinear) ---
    for tag, cols in [("erm-gbdt-lean", lean_cols), ("erm-gbdt-rich", rich_cols)]:
        Xtr = train_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
        ytr = train_df["y_week"].to_numpy(float)
        Xva = valid_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)

        q_pred = np.clip(_erm_gbdt(Xtr, ytr, Xva, alpha), 0, None)
        arms[tag] = q_pred
        print(f"{tag}: GBDT fitted ({time.time() - t0:.0f}s)")

    # Also fit at τ=0.50 for each arm to get q50
    q50_map = {}
    for arm_name in arms:
        cols = rich_cols if "rich" in arm_name else lean_cols
        Xtr = train_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
        ytr = train_df["y_week"].to_numpy(float)
        Xva = valid_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)

        if "linear" in arm_name:
            try:
                q50_map[arm_name] = np.clip(_erm_linear(Xtr, ytr, Xva, 0.50), 0, None)
            except Exception:
                q50_map[arm_name] = np.zeros(len(Xva))
        else:
            q50_map[arm_name] = np.clip(_erm_gbdt(Xtr, ytr, Xva, 0.50), 0, None)

    rows = []
    for arm_name, q85_all in arms.items():
        q50_all = q50_map[arm_name]
        for o in VALID_ORIGINS:
            sel = (valid_df.origin == o).to_numpy()
            sub = valid_df[sel]
            if not len(sub):
                continue
            rows.append(pd.DataFrame({
                "variant": arm_name, "series_id": sub.series_id.to_numpy(),
                "month": o, "split": "validation",
                "y": sub.y_week.to_numpy(),
                "q50": q50_all[sel], "q85": q85_all[sel], "w": 1.0,
            }))

    if not rows:
        print("无预测结果")
        return 1

    pred = pd.concat(rows, ignore_index=True)
    pred.to_parquet(ART / "predictions_erm_validation.parquet", index=False)

    # Joint comparison
    base_path = ART / "predictions_validation.parquet"
    if base_path.exists():
        base = pd.read_parquet(base_path)
        combined = pd.concat([base, pred], ignore_index=True)
    else:
        combined = pred

    print(f"\n{'=' * 66}")
    print(f"{'model':<20} {'NPL':>8} {'cov50':>8} {'cov85':>8} {'n':>8}")
    order = ["chronos2-ft-full", "chronos2-zs", "tft",
             "erm-gbdt-rich", "erm-gbdt-lean", "erm-linear-rich", "erm-linear-lean",
             "gbdt-rich", "gbdt-lean", "emp-daily"]
    for v in order:
        s = combined[combined.variant == v]
        if not len(s):
            continue
        r = evaluate_slice(s.y, s.q50, s.q85, s.w, y_min)
        print(f"{v:<20} {r.npl:8.4f} {r.cov_50_pos:8.4f} {r.cov_85_pos:8.4f} {r.n:8d}")

    # Paired bootstrap
    if "emp-daily" in combined.variant.values:
        print(f"\n配对 bootstrap（基准 = emp-daily，限重叠行）")
        emp = combined[combined.variant == "emp-daily"]
        emp_keys = set(zip(emp.series_id, emp.month))
        for mn in arms:
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
    chk.note("alpha", alpha)
    chk.note("arms", list(arms.keys()))
    chk.finish(ART / "checks" / "erm_run.json")
    return chk.exit_code


if __name__ == "__main__":
    sys.exit(main())
