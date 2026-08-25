"""Hybrid pipeline: TSFM forecast features + supply-side rich features → decision model → S.

TSFM 在需求侧有优势（时序模式识别），ERM-rich 在供应侧有优势（端到端利用库存/成本）。
Hybrid 思路：保留 TSFM 的 pretrained representation 不做修改，在下游用轻量决策模型
组合 TSFM 的分位数预测 + supply-side 特征，用 newsvendor quantile loss 直接输出 S。

输入特征:
  - TSFM forecast features: PI 天卷积后的 q50, q85, mean, std (4 维)
  - Rich supply-side features: 与 ERM-rich 相同 (beginning_inventory, unit_cost, etc.)
  - Lean time-series features: 与 ERM-lean 相同 (lag1, roll7, etc.)

用法:  PYTHONPATH=src python -m f2d.run_zhao_layerc_hybrid [--device mps]
"""

from __future__ import annotations

import sys
import time

import lightgbm as lgb  # must import before torch to avoid segfault on macOS
import numpy as np
import pandas as pd
import torch

from . import config as cfgmod
from .aggregation import convolve_varying_pmf
from .conventions import SEED_BASE
from .datasets import zhao
from .decision import POLICIES, costs_from_alpha, order_up_to
from .models.chronos import (BASE_CHECKPOINT, NATIVE_LEVELS, QuantileRepair,
                             to_grid)
from .models.gbdt_grid import LEAN_FEATURES, make_lean_features
from .simulation import ReplayConfig, replay

ART = cfgmod.ARTIFACT_DIR / "zhao_layerc"
FT_CKPT = cfgmod.ARTIFACT_DIR / "zhao_finetune" / "chronos2-ft-full" / "ft"
VMAX = 60
LEAD_DAYS = 1
ALPHA_GRID = (0.85, 0.90, 0.95, 0.98)
KAPPA_H = 0.20
HORIZON = 7

VALID_MONTHS = [pd.Timestamp("2019-07-01"), pd.Timestamp("2019-08-01")]
TRAIN_ORIGINS = pd.date_range("2019-02-04", "2019-05-27", freq="7D")


def _tsfm_pi_features(pipe, sids, ctx, pi_days, batch_size=256):
    """Run TSFM inference and convolve to PI-day summary features.

    Returns dict: series_id -> (q50_pi, q85_pi, mean_pi, std_pi).
    """
    valid = [s for s in sids if s in ctx and len(ctx[s]) >= 30]
    if not valid:
        return {}

    tensors = [torch.tensor(ctx[s], dtype=torch.float32) for s in valid]
    n_days = min(pi_days, HORIZON)

    q, _ = pipe.predict_quantiles(
        tensors, prediction_length=n_days,
        quantile_levels=list(NATIVE_LEVELS), batch_size=batch_size)
    g, _ = QuantileRepair()(to_grid(q))
    g = g.reshape(len(valid), n_days, -1)

    per_step = [g[:, h, :] for h in range(n_days)]
    pmf = convolve_varying_pmf(NATIVE_LEVELS, per_step, vmax=VMAX)

    support = np.arange(pmf.shape[1], dtype=float)
    mean_pi = pmf @ support
    var_pi = pmf @ (support ** 2) - mean_pi ** 2
    std_pi = np.sqrt(np.maximum(var_pi, 0))

    from .aggregation import pmf_quantile
    qs = pmf_quantile(pmf, [0.50, 0.85])
    q50_pi = qs[0.50]
    q85_pi = qs[0.85]

    if pi_days > n_days:
        scale = pi_days / n_days
        q50_pi = np.round(q50_pi * scale)
        q85_pi = np.round(q85_pi * scale)
        mean_pi = mean_pi * scale
        std_pi = std_pi * np.sqrt(scale)

    result = {}
    for i, s in enumerate(valid):
        result[s] = (q50_pi[i], q85_pi[i], mean_pi[i], std_pi[i])
    return result


def _build_hybrid_training_data(pipe, daily, feat, sids, pi_days, batch_size):
    """Build training DataFrame with TSFM features + rich features + y_pi."""
    from .run_zhao_gbdt import RICH_CAT, RICH_NUM, build_rich_block
    from .encoding import VocabStore

    raw = zhao.load_raw()
    rich = build_rich_block(raw)
    ridx = rich.set_index(["sku_ID", "month"])
    sku_of = dict(zip(daily.series_id, daily.sku_ID))

    fidx = feat.set_index(["series_id", "d"])
    didx = daily.set_index(["series_id", "d"])
    sid_set = set(sids)
    hist_len = daily.groupby("series_id")["d"].min().to_dict()

    rows = []
    for o in TRAIN_ORIGINS:
        eligible = np.array([s for s in sid_set
                             if s in hist_len and (o - hist_len[s]).days >= 30])
        if not len(eligible):
            continue

        ctx = {s: g.sort_values("d").y.to_numpy()
               for s, g in daily[daily.d < o].groupby("series_id")
               if s in set(eligible)}

        tsfm_feats = _tsfm_pi_features(pipe, eligible, ctx, pi_days, batch_size)

        key = pd.MultiIndex.from_product([eligible, [o]])
        base = fidx.reindex(key)[LEAN_FEATURES].dropna(how="all")
        if not len(base):
            continue
        base_sids = base.index.get_level_values(0).to_numpy()

        m = pd.Timestamp(o).to_period("M").to_timestamp()
        skus = [sku_of[s] for s in base_sids]
        rb = ridx.reindex(pd.MultiIndex.from_arrays(
            [skus, np.repeat(m, len(skus))]))
        for c in RICH_NUM + RICH_CAT:
            base[c] = rb[c].to_numpy()

        tsfm_q50 = np.array([tsfm_feats.get(s, (0, 0, 0, 0))[0] for s in base_sids])
        tsfm_q85 = np.array([tsfm_feats.get(s, (0, 0, 0, 0))[1] for s in base_sids])
        tsfm_mean = np.array([tsfm_feats.get(s, (0, 0, 0, 0))[2] for s in base_sids])
        tsfm_std = np.array([tsfm_feats.get(s, (0, 0, 0, 0))[3] for s in base_sids])

        base["tsfm_q50_pi"] = tsfm_q50
        base["tsfm_q85_pi"] = tsfm_q85
        base["tsfm_mean_pi"] = tsfm_mean
        base["tsfm_std_pi"] = tsfm_std

        y_pi = []
        for s in base_sids:
            total = 0.0
            for d_off in range(pi_days):
                d = o + pd.Timedelta(days=d_off)
                try:
                    total += float(didx.loc[(s, d), "y"])
                except KeyError:
                    pass
            y_pi.append(total)
        base["y_pi"] = y_pi

        blk = base.reset_index(drop=True)
        blk["series_id"] = base_sids
        blk["origin"] = o
        rows.append(blk)

    train_df = pd.concat(rows, ignore_index=True)

    vs = VocabStore.fit(train_df, RICH_CAT,
                        frozen_on=str(TRAIN_ORIGINS[-1].date()))
    train_df = vs.transform(train_df)

    return train_df, vs


TSFM_FEATURES = ["tsfm_q50_pi", "tsfm_q85_pi", "tsfm_mean_pi", "tsfm_std_pi"]


def _predict_hybrid(pipe, train_df, vs, sids, daily, feat, alpha, pi_days,
                    batch_size, method="gbdt"):
    """Train hybrid decision model and predict S for validation.

    method: "gbdt" = LightGBM quantile, "linear" = statsmodels QuantReg.
    """
    from .run_zhao_gbdt import RICH_CAT, RICH_NUM, build_rich_block

    raw = zhao.load_raw()
    rich = build_rich_block(raw)
    ridx = rich.set_index(["sku_ID", "month"])
    sku_of = dict(zip(daily.series_id, daily.sku_ID))

    cols = LEAN_FEATURES + RICH_NUM + RICH_CAT + TSFM_FEATURES

    Xtr = train_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    ytr = train_df["y_pi"].to_numpy(float)

    if method == "gbdt":
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=float(alpha),
            n_estimators=300, learning_rate=0.05,
            num_leaves=31, min_child_samples=40,
            feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
            lambda_l1=0.1, lambda_l2=1.0,
            seed=SEED_BASE, deterministic=True, force_row_wise=True, verbose=-1)
        model.fit(Xtr, ytr)
    elif method == "linear":
        import statsmodels.api as sm
        Xtr_c = sm.add_constant(Xtr.to_numpy(float))
        qr = sm.QuantReg(ytr, Xtr_c)
        model = qr.fit(q=alpha, max_iter=1000)

    origin = VALID_MONTHS[0]
    fidx = feat.set_index(["series_id", "d"])
    key = pd.MultiIndex.from_product([list(sids), [origin]])
    base_p = fidx.reindex(key)[LEAN_FEATURES].dropna(how="all")
    pred_sids = base_p.index.get_level_values(0).to_numpy()

    mo = pd.Timestamp(origin).to_period("M").to_timestamp()
    skus = [sku_of[s] for s in pred_sids]
    rb = ridx.reindex(pd.MultiIndex.from_arrays(
        [skus, np.repeat(mo, len(skus))]))
    for c in RICH_NUM + RICH_CAT:
        base_p[c] = rb[c].to_numpy()
    base_p = vs.transform(base_p.reset_index(drop=True))

    ctx = {s: g.sort_values("d").y.to_numpy()
           for s, g in daily[daily.d < origin].groupby("series_id")
           if s in set(pred_sids)}
    tsfm_feats = _tsfm_pi_features(pipe, pred_sids, ctx, pi_days, batch_size)

    base_p["tsfm_q50_pi"] = [tsfm_feats.get(s, (0, 0, 0, 0))[0] for s in pred_sids]
    base_p["tsfm_q85_pi"] = [tsfm_feats.get(s, (0, 0, 0, 0))[1] for s in pred_sids]
    base_p["tsfm_mean_pi"] = [tsfm_feats.get(s, (0, 0, 0, 0))[2] for s in pred_sids]
    base_p["tsfm_std_pi"] = [tsfm_feats.get(s, (0, 0, 0, 0))[3] for s in pred_sids]

    Xva = base_p[cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    if method == "gbdt":
        q_pred = np.clip(model.predict(Xva), 0, None)
        importances = dict(zip(cols, model.feature_importances_))
    elif method == "linear":
        import statsmodels.api as sm
        Xva_c = sm.add_constant(Xva.to_numpy(float))
        q_pred = np.clip(Xva_c @ model.params, 0, None)
        importances = dict(zip(cols, np.abs(model.params[1:])))

    S = np.round(q_pred).clip(0)

    S_full = np.zeros(len(sids))
    sid_to_idx = {s: i for i, s in enumerate(sids)}
    for i, s in enumerate(pred_sids):
        if s in sid_to_idx:
            S_full[sid_to_idx[s]] = S[i]

    return S_full, importances


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=2000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args(argv)

    from chronos import BaseChronosPipeline

    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)

    pipe = BaseChronosPipeline.from_pretrained(
        FT_CKPT, device_map=args.device, torch_dtype=torch.float32)
    print(f"TSFM checkpoint loaded ({time.time() - t0:.0f}s)")

    raw = zhao.load_raw()
    daily, _ = zhao.build_daily_panel(raw)
    panel, _ = zhao.build_panel(raw)

    rng = np.random.default_rng(SEED_BASE)
    pool = np.asarray(sorted(set(daily.series_id)))
    keep = rng.choice(pool, size=min(args.n_series, len(pool)), replace=False)
    daily = daily[daily.series_id.isin(keep)]
    sku_of = dict(zip(daily.series_id, daily.sku_ID))
    feat = make_lean_features(daily)

    replay_start = VALID_MONTHS[0]
    replay_end = VALID_MONTHS[-1] + pd.DateOffset(months=1)
    replay_days_total = (replay_end - replay_start).days + LEAD_DAYS
    review_cadence_cross = 30
    pi_days = review_cadence_cross + LEAD_DAYS

    snap_jul = panel[panel.month == VALID_MONTHS[0]].set_index("sku_ID")
    snap_aug = panel[panel.month == VALID_MONTHS[1]].set_index("sku_ID")
    common_skus = sorted(set(snap_jul.index) & set(snap_aug.index))

    hist = daily[daily.d < replay_start]
    ctx = {s: g.y.to_numpy() for s, g in hist.groupby("series_id")}
    sids_cross = np.array([s for s in sorted(ctx)
                           if len(ctx[s]) >= 30
                           and sku_of[s] in common_skus])
    sk_cross = np.array([sku_of[s] for s in sids_cross])
    n_cross = len(sids_cross)
    print(f"跨月序列: {n_cross}")

    initial_inv_cross = snap_jul.loc[sk_cross, "beginning_inventory"].to_numpy(float)
    cost_cross = snap_jul.loc[sk_cross, "unit_cost_hist"].to_numpy(float)

    demand_cross = np.zeros((n_cross, replay_days_total))
    cross_daily = daily[(daily.d >= replay_start) &
                        (daily.d < replay_end + pd.Timedelta(days=LEAD_DAYS))]
    for i, s in enumerate(sids_cross):
        sd = cross_daily[cross_daily.series_id == s].set_index("d")
        for day_idx in range(replay_days_total):
            d = replay_start + pd.Timedelta(days=day_idx)
            if d in sd.index:
                demand_cross[i, day_idx] = float(sd.loc[d, "y"])

    review_days_list = list(range(0, replay_days_total, review_cadence_cross))
    n_reviews = len(review_days_list)

    print(f"Building hybrid training data with TSFM features... ({time.time()-t0:.0f}s)")
    train_df, vs = _build_hybrid_training_data(
        pipe, daily, feat, sids_cross, pi_days, args.batch_size)
    print(f"Training data: {len(train_df)} rows ({time.time()-t0:.0f}s)")

    HYBRID_ARMS = [
        ("hybrid-tsfm-gbdt", "gbdt"),
        ("hybrid-tsfm-linear", "linear"),
    ]

    cross_rows = []

    for arm, method in HYBRID_ARMS:
        print(f"\n--- {arm} ({method}) ---")
        for alpha in ALPHA_GRID:
            S_val, importances = _predict_hybrid(
                pipe, train_df, vs, sids_cross, daily, feat, alpha, pi_days,
                args.batch_size, method=method)

            S_matrix = np.tile(S_val[:, None], (1, n_reviews))

            rc = ReplayConfig(
                n_days=replay_days_total,
                lead_time_days=LEAD_DAYS,
                review_cadence_days=review_cadence_cross,
                shortage_mechanism="lost_sales")
            res = replay(demand_cross, S_matrix, initial_inv_cross, rc)

            has_cost = np.isfinite(cost_cross)
            avg_iend = res.i_end.mean(axis=1)
            total_short = res.lost.sum(axis=1)

            n_so = 0
            n_cyc = 0
            for ci, start in enumerate(review_days_list):
                end = review_days_list[ci + 1] if ci + 1 < n_reviews else replay_days_total
                cyc_lost = res.lost[:, start:end].sum(axis=1)
                n_so += int((cyc_lost > 0).sum())
                n_cyc += n_cross
            csr = 1.0 - n_so / max(n_cyc, 1)
            fr = 1.0 - float(res.lost.sum()) / max(res.demand.sum(), 1e-12)

            h_c, p_c = costs_from_alpha(cost_cross, alpha, KAPPA_H, 12)
            row_cost = h_c * avg_iend + p_c * total_short
            avg_cpsd = float(np.nanmean(row_cost[has_cost])) / replay_days_total

            cross_rows.append({
                "arm": arm, "policy": "P1", "alpha": alpha,
                "costing": "derived",
                "n_series": n_cross, "n_days": replay_days_total,
                "n_reviews": n_reviews,
                "CSR": round(csr, 4), "FR": round(fr, 4),
                "avg_cost_per_series_day": round(avg_cpsd, 6),
                "avg_inventory": round(float(avg_iend.mean()), 2),
                "total_lost": round(float(total_short.sum()), 1),
                "S_median": round(float(np.median(S_val)), 1),
                "n_violations": len(res.conservation_violations),
            })

            if alpha == 0.85:
                print(f"  cost={avg_cpsd:.4f} CSR={csr:.4f} FR={fr:.4f} S={np.median(S_val):.0f}")
                print(f"  Top features:")
                for k, v in sorted(importances.items(), key=lambda x: -x[1])[:10]:
                    print(f"    {k:<30} {v}")

    hybrid_df = pd.DataFrame(cross_rows)

    all_arm_names = [a for a, _ in HYBRID_ARMS]
    existing_path = ART / "layer_c_crossmonth.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        existing = existing[~existing.arm.isin(all_arm_names)]
        combined = pd.concat([existing, hybrid_df], ignore_index=True)
    else:
        combined = hybrid_df
    combined.to_csv(existing_path, index=False)

    print(f"\n=== Layer C 跨月结果 (α=0.85, derived, P1) ===")
    sel = combined[(combined.alpha == 0.85) & (combined.costing == "derived")
                   & (combined.policy == "P1")]
    for _, r in sel.sort_values("avg_cost_per_series_day").iterrows():
        print(f"  {r.arm:<24} cost={r.avg_cost_per_series_day:.4f} "
              f"CSR={r.CSR:.4f} FR={r.FR:.4f} S={r.S_median}")

    print(f"\n总耗时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
