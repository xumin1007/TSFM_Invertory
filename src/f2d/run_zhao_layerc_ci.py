"""Layer C 置信区间：全部 16 臂 × 验证窗 + 测试窗，paired bootstrap by series.

只在主情景 (α=0.85, P1, derived) 上跑 bootstrap，得到 ΔCost 相对 emp-daily 的 95% CI。
输出: artifacts/zhao_layerc/layer_c_bootstrap.csv

用法:  PYTHONPATH=src python -m f2d.run_zhao_layerc_ci [--device cpu] [--batch-size 64]
"""

from __future__ import annotations

import sys
import time

import lightgbm as lgb  # before torch to avoid macOS segfault
import numpy as np
import pandas as pd
import torch

from . import config as cfgmod
from .aggregation import convolve_varying_pmf, pmf_quantile
from .conventions import SEED_BASE
from .datasets import zhao
from .decision import POLICIES, costs_from_alpha, order_up_to
from .models.chronos import (BASE_CHECKPOINT, NATIVE_LEVELS, QuantileRepair,
                             to_grid)
from .models.gbdt_grid import LEAN_FEATURES, QuantileGridGBDT, make_lean_features
from .simulation import ReplayConfig, replay
from .uncertainty import paired_bootstrap_mean, report

ART = cfgmod.ARTIFACT_DIR / "zhao_layerc"
FT_CKPTS = {
    "chronos2-ft-full": cfgmod.ARTIFACT_DIR / "zhao_finetune" / "chronos2-ft-full" / "ft",
    "chronos2-ft-full-h32": cfgmod.ARTIFACT_DIR / "zhao_finetune" / "chronos2-ft-full-h32" / "ft",
    "chronos2-ft-full-short": cfgmod.ARTIFACT_DIR / "zhao_finetune" / "chronos2-ft-full-short" / "ft",
}

VMAX = 60
LEAD_DAYS = 1
ALPHA = 0.85
KAPPA_H = 0.20
HORIZON = 7
BASELINE = "emp-daily"
BOOTSTRAP_B = 10_000

# ---- Windows ----
VALID_MONTHS = [pd.Timestamp("2019-07-01"), pd.Timestamp("2019-08-01")]
TEST_MONTHS = [pd.Timestamp("2019-09-01"), pd.Timestamp("2019-10-01")]

GBDT_TRAIN_ORIGINS_VALID = pd.date_range("2019-02-01", "2019-05-01", freq="MS")
GBDT_TRAIN_ORIGINS_TEST = pd.date_range("2019-02-01", "2019-07-01", freq="MS")
ERM_TRAIN_ORIGINS_VALID = pd.date_range("2019-02-04", "2019-05-27", freq="7D")
ERM_TRAIN_ORIGINS_TEST = pd.date_range("2019-02-04", "2019-07-28", freq="7D")

BASE_ARMS = ("chronos2-zs", "chronos2-ft-full", "chronos2-ft-full-h32",
             "chronos2-ft-full-short", "emp-daily", "gbdt-lean", "always-zero")
EXTRA_ARMS = ("erm-gbdt-rich", "erm-gbdt-lean", "erm-linear-rich",
              "erm-linear-lean", "tft", "deepar", "gbdt-rich")
HYBRID_ARMS = (("hybrid-tsfm-gbdt", "gbdt"), ("hybrid-tsfm-linear", "linear"))

TSFM_FEATURES = ["tsfm_q50_pi", "tsfm_q85_pi", "tsfm_mean_pi", "tsfm_std_pi"]


# ==================================================================
# Reuse stateless helpers from test runner
# ==================================================================
from .run_zhao_layerc_test import (
    _daily_grids_base,
    _s_from_deepprob,
)


def _train_deepprob_grids_window(arm_name, sids, daily, window_start):
    """Train DeepAR/TFT for a given window."""
    hist = daily[daily.d < window_start]
    ctx = {s: g.sort_values("d").y.to_numpy() for s, g in hist.groupby("series_id")}
    input_sz = 30
    min_len = input_sz + 7 + 2
    valid_sids = [s for s in sids if s in ctx and len(ctx[s]) >= min_len]
    train_contexts = {str(s): ctx[s] for s in valid_sids}
    from .models.deep_prob import train_and_predict
    grids = train_and_predict(
        arm_name, train_contexts, train_contexts,
        horizon=7, max_steps=500, input_size=input_sz,
        seed=SEED_BASE, accelerator="auto")
    ordered_sids = [s for s in valid_sids if str(s) in grids]
    per_step = []
    for h in range(7):
        step_grid = np.array([grids[str(s)][h] for s in ordered_sids])
        per_step.append(step_grid)
    return per_step, ordered_sids


def _replay_per_series(arm, S_val, demand, initial_inv, cost_arr,
                       replay_days_total, review_days_list, n_reviews, n_cross):
    """Replay and return per-series cost / replay_days (for bootstrap)."""
    S_matrix = np.tile(S_val[:, None], (1, n_reviews))
    rc = ReplayConfig(
        n_days=replay_days_total,
        lead_time_days=LEAD_DAYS,
        review_cadence_days=review_days_list[1] if n_reviews > 1 else replay_days_total,
        shortage_mechanism="lost_sales")
    res = replay(demand, S_matrix, initial_inv, rc)

    avg_iend = res.i_end.mean(axis=1)
    total_short = res.lost.sum(axis=1)

    h_c, p_c = costs_from_alpha(cost_arr, ALPHA, KAPPA_H, 12)
    row_cost = (h_c * avg_iend + p_c * total_short) / replay_days_total

    # CSR / FR for summary
    n_so = 0
    n_cyc = 0
    for ci, start in enumerate(review_days_list):
        end = review_days_list[ci + 1] if ci + 1 < n_reviews else replay_days_total
        cyc_lost = res.lost[:, start:end].sum(axis=1)
        n_so += int((cyc_lost > 0).sum())
        n_cyc += n_cross
    csr = 1.0 - n_so / max(n_cyc, 1)
    fr = 1.0 - float(res.lost.sum()) / max(res.demand.sum(), 1e-12)

    has_cost = np.isfinite(cost_arr)
    avg_cpsd = float(np.nanmean(row_cost[has_cost]))

    return row_cost, avg_cpsd, csr, fr


def _train_erm_window(arm_name, sids, daily, feat, pi_days, erm_origins, window_start):
    """Train ERM for a given window (validation or test)."""
    from .run_zhao_gbdt import RICH_CAT, RICH_NUM, build_rich_block
    from .encoding import VocabStore

    raw = zhao.load_raw()
    rich = build_rich_block(raw) if "rich" in arm_name else None
    sku_of = dict(zip(daily.series_id, daily.sku_ID))

    fidx = feat.set_index(["series_id", "d"])
    didx = daily.set_index(["series_id", "d"])
    ridx = rich.set_index(["sku_ID", "month"]) if rich is not None else None

    sid_set = set(sids)
    hist_len = daily.groupby("series_id")["d"].min().to_dict()

    rows = []
    for o in erm_origins:
        eligible = [s for s in sid_set
                    if s in hist_len and (o - hist_len[s]).days >= 30]
        if not eligible:
            continue
        key = pd.MultiIndex.from_product([eligible, [o]])
        base = fidx.reindex(key)[LEAN_FEATURES].dropna(how="all")
        if not len(base):
            continue
        base_sids = base.index.get_level_values(0).to_numpy()

        if ridx is not None:
            m = pd.Timestamp(o).to_period("M").to_timestamp()
            skus = [sku_of[s] for s in base_sids]
            rb = ridx.reindex(pd.MultiIndex.from_arrays(
                [skus, np.repeat(m, len(skus))]))
            for c in RICH_NUM + RICH_CAT:
                base[c] = rb[c].to_numpy()

        blk = base.copy()
        blk["series_id"] = base_sids
        blk["origin"] = o

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
        blk["y_pi"] = y_pi
        rows.append(blk.reset_index(drop=True))

    train_df = pd.concat(rows, ignore_index=True)
    cols = LEAN_FEATURES + (RICH_NUM + RICH_CAT if rich is not None else [])

    vs = None
    if rich is not None:
        vs = VocabStore.fit(train_df, RICH_CAT,
                            frozen_on=str(erm_origins[-1].date()))
        train_df = vs.transform(train_df)

    Xtr = train_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
    ytr = train_df["y_pi"].to_numpy(float)

    origin = window_start
    key = pd.MultiIndex.from_product([list(sids), [origin]])
    base_p = fidx.reindex(key)[LEAN_FEATURES].dropna(how="all")
    pred_sids = base_p.index.get_level_values(0).to_numpy()

    if ridx is not None:
        m = pd.Timestamp(origin).to_period("M").to_timestamp()
        skus = [sku_of[s] for s in pred_sids]
        rb = ridx.reindex(pd.MultiIndex.from_arrays(
            [skus, np.repeat(m, len(skus))]))
        for c in RICH_NUM + RICH_CAT:
            base_p[c] = rb[c].to_numpy()
        base_p = vs.transform(base_p.reset_index(drop=True))
    else:
        base_p = base_p.reset_index(drop=True)

    Xva = base_p[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)

    if "linear" in arm_name:
        import statsmodels.api as sm
        Xc = sm.add_constant(Xtr, has_constant="add")
        model = sm.QuantReg(ytr, Xc)
        res = model.fit(q=ALPHA, max_iter=1000)
        q_pred = np.clip(sm.add_constant(Xva, has_constant="add") @ res.params, 0, None)
    else:
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=float(ALPHA),
            n_estimators=300, learning_rate=0.05,
            num_leaves=31, min_child_samples=40,
            feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
            lambda_l1=0.1, lambda_l2=1.0,
            seed=SEED_BASE, deterministic=True, force_row_wise=True, verbose=-1)
        m.fit(Xtr, ytr)
        q_pred = np.clip(m.predict(Xva), 0, None)

    S = np.round(q_pred).clip(0)
    S_full = np.zeros(len(sids))
    sid_to_idx = {s: i for i, s in enumerate(sids)}
    for i, s in enumerate(pred_sids):
        if s in sid_to_idx:
            S_full[sid_to_idx[s]] = S[i]
    return S_full


def _tsfm_pi_features(pipe, sids, ctx, pi_days, batch_size=64):
    """Extract TSFM quantile features for PI-day horizon."""
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
    qs = pmf_quantile(pmf, [0.50, 0.85])
    q50_pi, q85_pi = qs[0.50], qs[0.85]
    if pi_days > n_days:
        scale = pi_days / n_days
        q50_pi = np.round(q50_pi * scale)
        q85_pi = np.round(q85_pi * scale)
        mean_pi = mean_pi * scale
        std_pi = std_pi * np.sqrt(scale)
    return {s: (q50_pi[i], q85_pi[i], mean_pi[i], std_pi[i])
            for i, s in enumerate(valid)}


def _build_hybrid_train_window(pipe, daily, feat, sids, pi_days, batch_size,
                                erm_origins):
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
    for o in erm_origins:
        eligible = np.array([s for s in sid_set
                             if s in hist_len and (o - hist_len[s]).days >= 30])
        if not len(eligible):
            continue
        ctx_o = {s: g.sort_values("d").y.to_numpy()
                 for s, g in daily[daily.d < o].groupby("series_id")
                 if s in set(eligible)}
        tsfm_feats = _tsfm_pi_features(pipe, eligible, ctx_o, pi_days, batch_size)

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
        for feat_name in TSFM_FEATURES:
            idx_map = {"tsfm_q50_pi": 0, "tsfm_q85_pi": 1,
                       "tsfm_mean_pi": 2, "tsfm_std_pi": 3}
            base[feat_name] = [tsfm_feats.get(s, (0, 0, 0, 0))[idx_map[feat_name]]
                               for s in base_sids]
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
                        frozen_on=str(erm_origins[-1].date()))
    train_df = vs.transform(train_df)
    return train_df, vs


def _predict_hybrid_window(pipe, train_df, vs, sids, daily, feat, pi_days,
                           batch_size, window_start, method="gbdt"):
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
            objective="quantile", alpha=float(ALPHA),
            n_estimators=300, learning_rate=0.05,
            num_leaves=31, min_child_samples=40,
            feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
            lambda_l1=0.1, lambda_l2=1.0,
            seed=SEED_BASE, deterministic=True, force_row_wise=True, verbose=-1)
        model.fit(Xtr, ytr)
    else:
        import statsmodels.api as sm
        Xtr_c = sm.add_constant(Xtr.to_numpy(float), has_constant="add")
        qr = sm.QuantReg(ytr, Xtr_c)
        model = qr.fit(q=ALPHA, max_iter=1000)

    origin = window_start
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

    for feat_name in TSFM_FEATURES:
        idx_map = {"tsfm_q50_pi": 0, "tsfm_q85_pi": 1,
                   "tsfm_mean_pi": 2, "tsfm_std_pi": 3}
        base_p[feat_name] = [tsfm_feats.get(s, (0, 0, 0, 0))[idx_map[feat_name]]
                             for s in pred_sids]

    Xva = base_p[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    if method == "gbdt":
        q_pred = np.clip(model.predict(Xva), 0, None)
    else:
        import statsmodels.api as sm
        Xva_c = sm.add_constant(Xva.to_numpy(float), has_constant="add")
        q_pred = np.clip(Xva_c @ model.params, 0, None)

    S = np.round(q_pred).clip(0)
    S_full = np.zeros(len(sids))
    sid_to_idx = {s: i for i, s in enumerate(sids)}
    for i, s in enumerate(pred_sids):
        if s in sid_to_idx:
            S_full[sid_to_idx[s]] = S[i]
    return S_full


def _s_from_gbdt_rich_window(sids, ctx, feat, pi_days, daily, gbdt_origins,
                              window_start):
    from .run_zhao_gbdt import RICH_CAT, RICH_NUM, build_rich_block
    from .encoding import VocabStore

    raw = zhao.load_raw()
    rich = build_rich_block(raw)
    fidx = feat.set_index(["series_id", "d"])
    sku_of = dict(zip(daily.series_id, daily.sku_ID))

    tr = []
    for o in gbdt_origins:
        sids_tr = np.asarray(sorted(set(
            daily[daily.d < o - pd.Timedelta(days=30)].series_id)))
        b = fidx.reindex(pd.MultiIndex.from_product(
            [sids_tr, [o]]))[LEAN_FEATURES].dropna(how="all")
        bs = b.index.get_level_values(0).to_numpy()
        ridx = rich.set_index(["sku_ID", "month"])
        m = pd.Timestamp(o).to_period("M").to_timestamp()
        skus = [sku_of[s] for s in bs]
        rb = ridx.reindex(pd.MultiIndex.from_arrays(
            [skus, np.repeat(m, len(skus))]))
        for c in RICH_NUM + RICH_CAT:
            b[c] = rb[c].to_numpy()
        for h in range(pi_days):
            blk = b.copy()
            blk["h"] = h
            blk["y"] = fidx.reindex(pd.MultiIndex.from_arrays(
                [bs, np.repeat(o + pd.Timedelta(days=h), len(bs))]))["y"].to_numpy()
            tr.append(blk.reset_index(drop=True))
    train = pd.concat(tr, ignore_index=True).dropna(subset=["y"])

    vs = VocabStore.fit(train, RICH_CAT, frozen_on=str(gbdt_origins[-1].date()))
    train = vs.transform(train)

    all_cols = LEAN_FEATURES + RICH_NUM + RICH_CAT + ["h"]
    gbdt = QuantileGridGBDT(features=all_cols).fit(train)

    origin = window_start
    base = fidx.reindex(pd.MultiIndex.from_product(
        [sids, [origin]]))[LEAN_FEATURES].dropna(how="all")
    base_sids = base.index.get_level_values(0).to_numpy()

    ridx = rich.set_index(["sku_ID", "month"])
    m = pd.Timestamp(origin).to_period("M").to_timestamp()
    skus = [sku_of[s] for s in base_sids]
    rb = ridx.reindex(pd.MultiIndex.from_arrays(
        [skus, np.repeat(m, len(skus))]))
    for c in RICH_NUM + RICH_CAT:
        base[c] = rb[c].to_numpy()
    base = vs.transform(base.reset_index(drop=True))

    grids = []
    for h in range(pi_days):
        blk = base.copy()
        blk["h"] = h
        for c in all_cols:
            if c not in RICH_CAT:
                blk[c] = pd.to_numeric(blk[c], errors="coerce").fillna(0)
        g = np.round(np.clip(gbdt.predict_grid(blk[all_cols]), 0.0, None))
        grids.append(np.maximum.accumulate(g, axis=1))

    pmf_pi = convolve_varying_pmf(NATIVE_LEVELS, grids, vmax=VMAX)
    pmf_r = convolve_varying_pmf(NATIVE_LEVELS,
                                  grids[:pi_days - LEAD_DAYS], vmax=VMAX)
    m_ratio = pi_days / (pi_days - LEAD_DAYS)
    S_dict = order_up_to(pmf_r, pmf_pi, ALPHA, m_ratio)
    return S_dict, base_sids


def run_window(window_name, months, gbdt_origins, erm_origins,
               daily, panel, feat, pipe, ft_pipes, sids_cross,
               sk_cross, initial_inv, cost_arr, demand, ctx,
               replay_days_total, review_days_list, n_reviews,
               n_cross, batch_size, t0):
    """Run all 16 arms on one window and return per-series costs dict."""
    per_series = {}  # arm -> cost array (n_cross,)
    summary = {}     # arm -> (avg_cpsd, csr, fr)

    review_cadence_cross = review_days_list[1] if n_reviews > 1 else replay_days_total
    pi_days = review_cadence_cross + LEAD_DAYS

    # ---- Base arms ----
    print(f"\n--- {window_name} Base arms ({time.time()-t0:.0f}s) ---")

    # Train GBDT-lean
    fidx = feat.set_index(["series_id", "d"])
    max_h = max(m.days_in_month for m in months) + LEAD_DAYS
    tr = []
    for o in gbdt_origins:
        sids_tr = np.asarray(sorted(set(
            daily[daily.d < o - pd.Timedelta(days=30)].series_id)))
        b = fidx.reindex(pd.MultiIndex.from_product(
            [sids_tr, [o]]))[LEAN_FEATURES].dropna(how="all")
        bs = b.index.get_level_values(0).to_numpy()
        for h in range(max_h):
            blk = b.copy()
            blk["h"] = h
            blk["y"] = fidx.reindex(pd.MultiIndex.from_arrays(
                [bs, np.repeat(o + pd.Timedelta(days=h), len(bs))]))["y"].to_numpy()
            tr.append(blk.reset_index(drop=True))
    train_gbdt = pd.concat(tr, ignore_index=True).dropna(subset=["y"])
    gbdt = QuantileGridGBDT(features=LEAN_FEATURES + ["h"]).fit(train_gbdt)

    for arm in BASE_ARMS:
        if arm in FT_CKPTS and arm not in ft_pipes:
            continue
        grids = _daily_grids_base(arm, sids_cross, ctx, replay_days_total,
                                  pipe, gbdt, feat, months[0],
                                  batch_size, ft_pipes)
        pi_grids = grids[:pi_days] if pi_days <= len(grids) else grids
        if len(pi_grids) < pi_days:
            pi_grids = pi_grids + [pi_grids[-1]] * (pi_days - len(pi_grids))
        pmf_pi = convolve_varying_pmf(NATIVE_LEVELS, pi_grids, vmax=VMAX)
        pmf_r = convolve_varying_pmf(NATIVE_LEVELS,
                                      grids[:review_cadence_cross], vmax=VMAX)
        m_ratio = pi_days / review_cadence_cross
        S_dict = order_up_to(pmf_r, pmf_pi, ALPHA, m_ratio)
        S_val = S_dict["P1"]

        row_cost, avg_cpsd, csr, fr = _replay_per_series(
            arm, S_val, demand, initial_inv, cost_arr,
            replay_days_total, review_days_list, n_reviews, n_cross)
        per_series[arm] = row_cost
        summary[arm] = (avg_cpsd, csr, fr)
        print(f"  {arm:<24} cost={avg_cpsd:.4f} ({time.time()-t0:.0f}s)")

    # ---- Extra arms ----
    print(f"\n--- {window_name} Extra arms ({time.time()-t0:.0f}s) ---")
    dp_cache = {}
    for arm in EXTRA_ARMS:
        if arm in ("tft", "deepar") and arm not in dp_cache:
            print(f"  训练 {arm}...")
            dp_cache[arm] = _train_deepprob_grids_window(arm, sids_cross, daily, months[0])

        if arm.startswith("erm-"):
            S_val = _train_erm_window(arm, sids_cross, daily, feat, pi_days,
                                      erm_origins, months[0])
        elif arm in ("tft", "deepar"):
            per_step, ordered_sids = dp_cache[arm]
            S_val = _s_from_deepprob(
                per_step, ordered_sids, sids_cross, ALPHA, pi_days)
        elif arm == "gbdt-rich":
            S_policies, gbdt_sids = _s_from_gbdt_rich_window(
                sids_cross, ctx, feat, pi_days, daily, gbdt_origins,
                months[0])
            sid2idx = {s: i for i, s in enumerate(gbdt_sids)}
            S_full = S_policies["P1"]
            S_val = np.array([S_full[sid2idx[s]] if s in sid2idx else 0.0
                              for s in sids_cross])
        else:
            continue

        row_cost, avg_cpsd, csr, fr = _replay_per_series(
            arm, S_val, demand, initial_inv, cost_arr,
            replay_days_total, review_days_list, n_reviews, n_cross)
        per_series[arm] = row_cost
        summary[arm] = (avg_cpsd, csr, fr)
        print(f"  {arm:<24} cost={avg_cpsd:.4f} ({time.time()-t0:.0f}s)")

    # ---- Hybrid arms ----
    print(f"\n--- {window_name} Hybrid arms ({time.time()-t0:.0f}s) ---")
    hybrid_pipe = ft_pipes.get("chronos2-ft-full", pipe)
    hybrid_train, hybrid_vs = _build_hybrid_train_window(
        hybrid_pipe, daily, feat, sids_cross, pi_days, batch_size, erm_origins)
    print(f"  Training data: {len(hybrid_train)} rows")

    for arm, method in HYBRID_ARMS:
        S_val = _predict_hybrid_window(
            hybrid_pipe, hybrid_train, hybrid_vs, sids_cross, daily, feat,
            pi_days, batch_size, months[0], method=method)

        row_cost, avg_cpsd, csr, fr = _replay_per_series(
            arm, S_val, demand, initial_inv, cost_arr,
            replay_days_total, review_days_list, n_reviews, n_cross)
        per_series[arm] = row_cost
        summary[arm] = (avg_cpsd, csr, fr)
        print(f"  {arm:<24} cost={avg_cpsd:.4f} ({time.time()-t0:.0f}s)")

    return per_series, summary


def _setup_window(daily, panel, months, lead_days=1, review_cadence=30):
    """Build replay arrays for a window."""
    sku_of = dict(zip(daily.series_id, daily.sku_ID))
    replay_start = months[0]
    replay_end = months[-1] + pd.DateOffset(months=1)
    replay_days_total = (replay_end - replay_start).days + lead_days

    snaps = [panel[panel.month == m].set_index("sku_ID") for m in months]
    common_skus = sorted(set.intersection(*[set(s.index) for s in snaps]))

    hist = daily[daily.d < replay_start]
    ctx = {s: g.y.to_numpy() for s, g in hist.groupby("series_id")}
    sids_cross = np.array([s for s in sorted(ctx)
                           if len(ctx[s]) >= 30
                           and sku_of[s] in common_skus])
    sk_cross = np.array([sku_of[s] for s in sids_cross])
    n_cross = len(sids_cross)

    initial_inv = snaps[0].loc[sk_cross, "beginning_inventory"].to_numpy(float)
    cost_arr = snaps[0].loc[sk_cross, "unit_cost_hist"].to_numpy(float)

    demand = np.zeros((n_cross, replay_days_total))
    cross_daily = daily[(daily.d >= replay_start) &
                        (daily.d < replay_end + pd.Timedelta(days=lead_days))]
    for i, s in enumerate(sids_cross):
        sd = cross_daily[cross_daily.series_id == s].set_index("d")
        for day_idx in range(replay_days_total):
            d = replay_start + pd.Timedelta(days=day_idx)
            if d in sd.index:
                demand[i, day_idx] = float(sd.loc[d, "y"])

    review_days_list = list(range(0, replay_days_total, review_cadence))
    n_reviews = len(review_days_list)

    return (sids_cross, sk_cross, initial_inv, cost_arr, demand, ctx,
            replay_days_total, review_days_list, n_reviews, n_cross)


def bootstrap_ci(per_series_costs, sids, baseline="emp-daily", b=10_000):
    """Run paired bootstrap by series, return ΔCost CI for each arm."""
    arms = sorted(per_series_costs.keys())
    if baseline not in arms:
        raise ValueError(f"Baseline {baseline} not in arms")

    variants = [a for a in arms if a != baseline]
    # Build a DataFrame for paired_bootstrap_mean
    rows = []
    for arm in arms:
        costs = per_series_costs[arm]
        for i, s in enumerate(sids):
            rows.append({"variant": arm, "series_id": s, "cost": costs[i]})

    df = pd.DataFrame(rows)
    df["month"] = 0  # dummy — single cross-month replay, no month dimension
    # Drop series with NaN cost in ANY arm (keeps pairing intact)
    nan_series = df[df["cost"].isna()]["series_id"].unique()
    if len(nan_series):
        df = df[~df["series_id"].isin(nan_series)]
    cis = paired_bootstrap_mean(
        df, value_col="cost", baseline=baseline, variants=variants,
        variant_col="variant", series_col="series_id",
        b=b, ci=0.95, seed=SEED_BASE)
    return cis


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)

    from chronos import BaseChronosPipeline

    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)

    raw = zhao.load_raw()
    daily, _ = zhao.build_daily_panel(raw)
    panel, _ = zhao.build_panel(raw)

    rng = np.random.default_rng(SEED_BASE)
    pool = np.asarray(sorted(set(daily.series_id)))
    keep = rng.choice(pool, size=min(args.n_series, len(pool)), replace=False)
    daily = daily[daily.series_id.isin(keep)]
    feat = make_lean_features(daily)

    # Load Chronos pipelines once
    pipe = BaseChronosPipeline.from_pretrained(
        BASE_CHECKPOINT, device_map=args.device, torch_dtype=torch.float32)
    ft_pipes = {}
    for a, c in FT_CKPTS.items():
        if c.exists():
            ft_pipes[a] = BaseChronosPipeline.from_pretrained(
                str(c), device_map=args.device, torch_dtype=torch.float32)
    print(f"Chronos pipelines loaded ({time.time()-t0:.0f}s)")

    all_ci_rows = []

    for window_name, months, gbdt_origins, erm_origins in [
        ("validation", VALID_MONTHS, GBDT_TRAIN_ORIGINS_VALID, ERM_TRAIN_ORIGINS_VALID),
        ("test", TEST_MONTHS, GBDT_TRAIN_ORIGINS_TEST, ERM_TRAIN_ORIGINS_TEST),
    ]:
        print(f"\n{'='*60}")
        print(f"  Window: {window_name} ({months[0]:%Y-%m} ~ {months[-1]:%Y-%m})")
        print(f"{'='*60}")

        (sids_cross, sk_cross, initial_inv, cost_arr, demand, ctx,
         replay_days_total, review_days_list, n_reviews, n_cross
         ) = _setup_window(daily, panel, months)

        print(f"  序列数: {n_cross}, 回放天数: {replay_days_total}")

        per_series, summary = run_window(
            window_name, months, gbdt_origins, erm_origins,
            daily, panel, feat, pipe, ft_pipes, sids_cross,
            sk_cross, initial_inv, cost_arr, demand, ctx,
            replay_days_total, review_days_list, n_reviews,
            n_cross, args.batch_size, t0)

        # Run bootstrap
        print(f"\n--- {window_name} Bootstrap (B={BOOTSTRAP_B}) ---")
        cis = bootstrap_ci(per_series, sids_cross, baseline=BASELINE,
                           b=BOOTSTRAP_B)

        for ci in sorted(cis, key=lambda c: c.delta):
            sig = "***" if ci.significant else "n.s."
            print(f"  {ci.variant:<24} Δ={ci.delta:+.4f}  "
                  f"[{ci.lo:+.4f}, {ci.hi:+.4f}]  {sig}")
            avg_cpsd, csr, fr = summary[ci.variant]
            all_ci_rows.append({
                "split": window_name,
                "arm": ci.variant,
                "cost": round(avg_cpsd, 4),
                "CSR": round(csr, 4),
                "FR": round(fr, 4),
                "delta_cost": round(ci.delta, 6),
                "ci_lo": round(ci.lo, 6),
                "ci_hi": round(ci.hi, 6),
                "significant": ci.significant,
                "n_series": ci.n_series,
                "B": ci.b,
            })

        # Add baseline row
        avg_cpsd, csr, fr = summary[BASELINE]
        all_ci_rows.append({
            "split": window_name,
            "arm": BASELINE,
            "cost": round(avg_cpsd, 4),
            "CSR": round(csr, 4),
            "FR": round(fr, 4),
            "delta_cost": 0.0,
            "ci_lo": 0.0,
            "ci_hi": 0.0,
            "significant": False,
            "n_series": n_cross,
            "B": BOOTSTRAP_B,
        })

    ci_df = pd.DataFrame(all_ci_rows)
    out_path = ART / "layer_c_bootstrap.csv"
    ci_df.to_csv(out_path, index=False)
    print(f"\n结果已保存: {out_path}")

    # Print summary tables
    for split in ["validation", "test"]:
        print(f"\n=== {split} window (α={ALPHA}, P1, derived) ===")
        sub = ci_df[ci_df.split == split].sort_values("cost")
        for _, r in sub.iterrows():
            sig = "***" if r.significant else "n.s."
            if r.arm == BASELINE:
                print(f"  {r.arm:<24} cost={r.cost:.4f}  (baseline)")
            else:
                print(f"  {r.arm:<24} cost={r.cost:.4f}  "
                      f"Δ={r.delta_cost:+.4f} [{r.ci_lo:+.4f},{r.ci_hi:+.4f}] {sig}")

    print(f"\n总耗时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
