"""Layer C 补充：ERM / TFT / DeepAR / GBDT-rich 的跨月连续回放。

复用 run_zhao_layerc 的回放框架（62 天，L=1，review=30 天），
但 S 的来源不同：
  - ERM: 周级 q_α 预测 × (PI/7) 缩放到月级
  - TFT / DeepAR: 日级 7 步网格 → 卷积得周级 PMF → 提取 q_α → × (PI/7)
  - GBDT-rich: 日级 21 点网格 → 卷积到 PI 天 PMF → order_up_to

用法:  PYTHONPATH=src python -m f2d.run_zhao_layerc_extra
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from . import config as cfgmod
from .aggregation import convolve_varying_pmf
from .conventions import SEED_BASE
from .datasets import zhao
from .decision import POLICIES, costs_from_alpha, order_up_to
from .models.chronos import NATIVE_LEVELS, QuantileRepair
from .models.gbdt_grid import LEAN_FEATURES, QuantileGridGBDT, make_lean_features
from .simulation import ReplayConfig, replay

ART = cfgmod.ARTIFACT_DIR / "zhao_layerc"
VMAX = 60
LEAD_DAYS = 1
ALPHA_GRID = (0.85, 0.90, 0.95, 0.98)
KAPPA_H = 0.20

VALID_MONTHS = [pd.Timestamp("2019-07-01"), pd.Timestamp("2019-08-01")]
GBDT_TRAIN_ORIGINS = pd.date_range("2019-02-01", "2019-05-01", freq="MS")


def _train_erm_for_sids(arm_name, sids, daily, alpha, pi_days):
    """在 Layer C 的序列集上训练 ERM，目标 = PI 天总需求，直接输出 S。"""
    import lightgbm as lgb
    from .run_zhao_gbdt import RICH_CAT, RICH_NUM, build_rich_block
    from .encoding import VocabStore

    raw = zhao.load_raw()
    feat = make_lean_features(daily)
    sku_of = dict(zip(daily.series_id, daily.sku_ID))

    TRAIN_ORIGINS = pd.date_range("2019-02-04", "2019-05-27", freq="7D")
    rich = build_rich_block(raw) if "rich" in arm_name else None

    fidx = feat.set_index(["series_id", "d"])
    didx = daily.set_index(["series_id", "d"])
    ridx = rich.set_index(["sku_ID", "month"]) if rich is not None else None

    sid_set = set(sids)
    hist_len = daily.groupby("series_id")["d"].min().to_dict()

    rows = []
    for o in TRAIN_ORIGINS:
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

    if rich is not None:
        vs = VocabStore.fit(train_df, RICH_CAT,
                            frozen_on=str(TRAIN_ORIGINS[-1].date()))
        train_df = vs.transform(train_df)

    Xtr = train_df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
    ytr = train_df["y_pi"].to_numpy(float)

    origin = VALID_MONTHS[0]
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
        Xc = sm.add_constant(Xtr)
        model = sm.QuantReg(ytr, Xc)
        res = model.fit(q=alpha, max_iter=1000)
        q_pred = np.clip(sm.add_constant(Xva) @ res.params, 0, None)
    else:
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=float(alpha),
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


def _train_deepprob_grids(arm_name, sids, daily):
    """Train once, return (per_step, ordered_sids)."""
    hist = daily[daily.d < VALID_MONTHS[0]]
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


def _s_from_deepprob_cached(per_step, ordered_sids, sids, alpha, pi_days):
    """Extract S at a given alpha from pre-computed grids."""
    from .aggregation import convolve_varying
    from .models.chronos import NATIVE_LEVELS as NL

    res = convolve_varying(NL, per_step, taus=(alpha,), vmax=300)
    q_weekly = res[alpha]
    S_monthly = np.round(q_weekly * (pi_days / 7.0)).clip(0)

    S_full = np.zeros(len(sids))
    sid_to_idx = {s: i for i, s in enumerate(sids)}
    for i, s in enumerate(ordered_sids):
        if s in sid_to_idx:
            S_full[sid_to_idx[s]] = S_monthly[i]
    return S_full


def _s_from_gbdt_rich(sids, ctx, feat, alpha, pi_days, daily):
    """GBDT-rich: 训练带结构化特征的 GBDT，生成日级网格 → 卷积 → S。"""
    from .run_zhao_gbdt import RICH_CAT, RICH_NUM, build_rich_block
    from .encoding import VocabStore

    raw = zhao.load_raw()
    rich = build_rich_block(raw)

    fidx = feat.set_index(["series_id", "d"])
    sku_of = dict(zip(daily.series_id, daily.sku_ID))

    tr = []
    for o in GBDT_TRAIN_ORIGINS:
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

    vs = VocabStore.fit(train, RICH_CAT, frozen_on=str(GBDT_TRAIN_ORIGINS[-1].date()))
    train = vs.transform(train)

    all_cols = LEAN_FEATURES + RICH_NUM + RICH_CAT + ["h"]
    gbdt = QuantileGridGBDT(features=all_cols).fit(train)

    origin = VALID_MONTHS[0]
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
    S_dict = order_up_to(pmf_r, pmf_pi, alpha, m_ratio)
    return S_dict, base_sids


def main(argv=None) -> int:
    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)

    raw = zhao.load_raw()
    daily, _ = zhao.build_daily_panel(raw)
    panel, _ = zhao.build_panel(raw)

    rng = np.random.default_rng(SEED_BASE)
    pool = np.asarray(sorted(set(daily.series_id)))
    keep = rng.choice(pool, size=min(2000, len(pool)), replace=False)
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

    EXTRA_ARMS = [
        "erm-gbdt-rich", "erm-gbdt-lean", "erm-linear-rich", "erm-linear-lean",
        "tft", "deepar", "gbdt-rich",
    ]

    cross_rows = []

    dp_cache = {}

    for arm in EXTRA_ARMS:
        if arm in ("tft", "deepar") and arm not in dp_cache:
            print(f"  训练 {arm}...")
            dp_cache[arm] = _train_deepprob_grids(arm, sids_cross, daily)

        for alpha in ALPHA_GRID:
            if arm.startswith("erm-"):
                S_val = _train_erm_for_sids(arm, sids_cross, daily, alpha, pi_days)
            elif arm in ("tft", "deepar"):
                per_step, ordered_sids = dp_cache[arm]
                S_val = _s_from_deepprob_cached(
                    per_step, ordered_sids, sids_cross, alpha, pi_days)
            elif arm == "gbdt-rich":
                S_policies, gbdt_sids = _s_from_gbdt_rich(
                    sids_cross, ctx, feat, alpha, pi_days, daily)
                sid2idx = {s: i for i, s in enumerate(gbdt_sids)}
                S_full = S_policies["P1"]
                S_val = np.array([S_full[sid2idx[s]] if s in sid2idx else 0.0
                                  for s in sids_cross])
            else:
                continue

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

        print(f"  {arm:<24} ({time.time() - t0:.0f}s)")

    extra_df = pd.DataFrame(cross_rows)

    existing_path = ART / "layer_c_crossmonth.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        existing = existing[~existing.arm.isin(EXTRA_ARMS)]
        combined = pd.concat([existing, extra_df], ignore_index=True)
    else:
        combined = extra_df
    combined.to_csv(existing_path, index=False)

    print("\n=== Layer C 跨月结果 (α=0.85, derived, P1) ===")
    sel = combined[(combined.alpha == 0.85) & (combined.costing == "derived")
                   & (combined.policy == "P1")]
    for _, r in sel.sort_values("avg_cost_per_series_day").iterrows():
        print(f"  {r.arm:<24} cost={r.avg_cost_per_series_day:.4f} "
              f"CSR={r.CSR:.4f} FR={r.FR:.4f} S={r.S_median}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
