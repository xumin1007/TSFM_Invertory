"""连续 carry-state / burn-in 回放（v2：统一 estimand）。

修正 v1 中的三个问题：
  1. missing-month bug：per-cell eligibility，不因 1 个 SKU 缺失删除整月
  2. CI/p 一致性：全部报告双侧 bootstrap CI，exact two-sided p
  3. cadence slope 符号：outcome = (C_ZS − C_Emp)/C_Emp，负值 = TSFM 更好

设计：
  - 使用 SEED_BASE=42 抽样 n_series（默认 2000），与 transmission audit 一致
  - 先生成 analysis_manifest.csv，冻结 series/origin/alpha/baseline 选择
  - 分析 eligibility 只由 common-series manifest 与有限 unit_cost_hist 决定
  - burn-in = 62 天（统一），计分月份 = origins 3-7（Jun-Oct）
  - 三条线同时计算：R0 (static), R3_switch (monthly reset), R3_longrun (carry-state)
  - Joint cluster bootstrap，三条线配对
  - Cadence sensitivity: R ∈ {1, 7, 30}

用法:  PYTHONPATH=src python -m f2d.run_continuous_replay [--n-series 2000]
"""

from __future__ import annotations

import argparse
import hashlib
import time

import numpy as np
import pandas as pd

from . import config as cfgmod
from .aggregation import convolve_varying_pmf
from .checks import CheckResult
from .conventions import SEED_BASE
from .datasets import zhao
from .decision import costs_from_alpha, order_up_to
from .models.chronos import BASE_CHECKPOINT, NATIVE_LEVELS, QuantileRepair, to_grid
from .models.gbdt_grid import make_lean_features
from .simulation import ReplayConfig, replay
from .run_rolling_origins import (ALPHAS, EMP_SCHEMES, HIGH_ALPHAS, ORIGINS,
                                  _emp_grid)

ART = cfgmod.ARTIFACT_DIR / "zhao_continuous"
VMAX = 60
LEAD_DAYS = 1
KAPPA_H = 0.20
REVIEW_CADENCES = (1, 7, 30)
BURN_IN_DAYS = 2 * (max(REVIEW_CADENCES) + LEAD_DAYS)  # 62
SCORED_ORIGINS = list(range(3, 8))  # Jun-Oct
B = 10_000


def _bootstrap_p2(draws: np.ndarray) -> float:
    """Plus-one corrected, two-sided bootstrap sign p-value."""
    x = np.asarray(draws)
    b = len(x)
    p_hi = (1 + np.count_nonzero(x >= 0)) / (b + 1)
    p_lo = (1 + np.count_nonzero(x <= 0)) / (b + 1)
    return min(2.0 * min(p_hi, p_lo), 1.0)


def _build_manifest(daily: pd.DataFrame, panel: pd.DataFrame,
                    sids: np.ndarray, sku_of: dict,
                    sel: pd.DataFrame) -> pd.DataFrame:
    """生成 analysis_manifest.csv：每个 (series, origin, alpha) cell 的 eligibility。"""
    base_of = {(r.alpha, r.origin_idx): r.retuned
               for _, r in sel.iterrows()}
    rows = []
    for oi in SCORED_ORIGINS:
        month = ORIGINS[oi]
        snap = panel[panel.month == month].set_index("sku_ID")
        snap_skus = set(snap.index)
        for s in sids:
            sk = sku_of.get(s)
            has_panel = sk in snap_skus
            for alpha in ALPHAS:
                ba = base_of.get((alpha, oi), "emp-daily")
                rows.append(dict(
                    series_id=s, sku_ID=sk,
                    origin_idx=oi, month=str(month.date()),
                    alpha=alpha, emp_baseline=ba,
                    has_panel=has_panel))
    mf = pd.DataFrame(rows)
    return mf


def _build_demand(daily: pd.DataFrame, sids: np.ndarray,
                  start: pd.Timestamp, total_days: int) -> np.ndarray:
    dm = np.zeros((len(sids), total_days))
    sid_idx = {s: i for i, s in enumerate(sids)}
    end = start + pd.Timedelta(days=total_days - 1)
    sub = daily[(daily.d >= start) & (daily.d <= end)].copy()
    sub = sub[sub.series_id.isin(sid_idx)]
    rows_i = sub.series_id.map(sid_idx).to_numpy()
    cols_i = ((sub.d - start).dt.days).to_numpy()
    vals = sub.y.to_numpy(float)
    valid = (cols_i >= 0) & (cols_i < total_days)
    dm[rows_i[valid], cols_i[valid]] = vals[valid]
    return dm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=2000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--b", type=int, default=B)
    args = ap.parse_args(argv)

    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    chk = CheckResult(step_id="zhao_continuous_v2", dataset="zhao",
                      seed=SEED_BASE)

    raw = zhao.load_raw()
    daily, _ = zhao.build_daily_panel(raw)
    panel, _ = zhao.build_panel(raw)

    rng = np.random.default_rng(SEED_BASE)
    pool = np.asarray(sorted(set(daily.series_id)))
    keep = rng.choice(pool, size=min(args.n_series, len(pool)), replace=False)
    daily = daily[daily.series_id.isin(keep)]
    sku_of = dict(zip(daily.series_id, daily.sku_ID))
    make_lean_features(daily)

    # Series eligible across ALL scored origins (common support)
    hist_all = {oi: {} for oi in range(len(ORIGINS))}
    for oi, month in enumerate(ORIGINS):
        h = daily[daily.d < month]
        hist_all[oi] = {s: g.y.to_numpy()
                        for s, g in h.groupby("series_id")}

    snap_march = panel[panel.month == ORIGINS[0]].set_index("sku_ID")
    eligible_per_origin = {}
    for oi in SCORED_ORIGINS:
        month = ORIGINS[oi]
        snap = panel[panel.month == month].set_index("sku_ID")
        ctx = hist_all[oi]
        eligible_per_origin[oi] = set(
            s for s in ctx
            if len(ctx[s]) >= 30
            and sku_of.get(s) in snap.index
            and sku_of.get(s) in snap_march.index)

    # Common support: present in ALL scored origins
    common_sids = eligible_per_origin[SCORED_ORIGINS[0]]
    for oi in SCORED_ORIGINS[1:]:
        common_sids = common_sids & eligible_per_origin[oi]
    common_sids = np.array(sorted(common_sids))
    common_sk = np.array([sku_of[s] for s in common_sids])
    n_common = len(common_sids)

    # Save series list with hash
    sid_hash = hashlib.sha256(
        ",".join(str(s) for s in common_sids).encode()).hexdigest()[:12]
    pd.DataFrame({"series_id": common_sids}).to_csv(
        ART / "common_series.csv", index=False)
    print(f"Common-support manifest: {n_common} series (hash={sid_hash})")
    print(f"Scored origins: {SCORED_ORIGINS} (Jun-Oct)")
    print(f"burn-in = {BURN_IN_DAYS} days  ({time.time()-t0:.0f}s)")

    # Load baseline selection
    sel_file = (cfgmod.ARTIFACT_DIR / "zhao_rolling_origins"
                / "baseline_selection.csv")
    if not sel_file.exists():
        print("ERROR: baseline_selection.csv not found")
        return 1
    sel = pd.read_csv(sel_file)
    base_of = {(r.alpha, r.origin_idx): r.retuned
               for _, r in sel.iterrows()}

    # Generate and save manifest
    manifest = _build_manifest(daily, panel, common_sids, sku_of, sel)

    # Cost eligibility is fixed before replay and does not depend on outcomes.
    common_inv0 = snap_march.loc[
        common_sk, "beginning_inventory"].to_numpy(float)
    common_cost = snap_march.loc[
        common_sk, "unit_cost_hist"].to_numpy(float)
    finite_cost = np.isfinite(common_cost)
    finite_cost_of = dict(zip(common_sids, finite_cost))
    manifest["finite_unit_cost"] = manifest.series_id.map(finite_cost_of)
    manifest["eligible"] = manifest.has_panel & manifest.finite_unit_cost
    manifest.to_csv(ART / "analysis_manifest.csv", index=False)

    sids = common_sids[finite_cost]
    sk = common_sk[finite_cost]
    inv0 = common_inv0[finite_cost]
    cost_i = common_cost[finite_cost]
    n_ser = len(sids)
    eligible_hash = hashlib.sha256(
        ",".join(str(s) for s in sids).encode()).hexdigest()[:12]
    pd.DataFrame({"series_id": sids}).to_csv(
        ART / "eligible_series.csv", index=False)
    print(f"Manifest: {len(manifest)} cells "
          f"(panel coverage = {manifest.has_panel.mean():.1%})")
    print(f"Cost-eligible series: {n_ser}/{n_common} "
          f"(excluded nonfinite unit_cost={n_common - n_ser}, "
          f"hash={eligible_hash})")

    if not np.isfinite(inv0).all():
        raise RuntimeError("Nonfinite initial inventory on the fixed eligible set")

    # Per-month beginning inventory (for switch replay)
    monthly_inv = {}
    for oi in SCORED_ORIGINS:
        month = ORIGINS[oi]
        snap = panel[panel.month == month].set_index("sku_ID")
        inv = np.full(n_ser, np.nan)
        for i, s in enumerate(sk):
            if s in snap.index:
                inv[i] = float(snap.loc[s, "beginning_inventory"])
        monthly_inv[oi] = inv

    # Demand matrix
    start_date = ORIGINS[0]
    end_date = (ORIGINS[-1] + pd.DateOffset(months=1)
                + pd.Timedelta(days=LEAD_DAYS - 1))
    total_days = (end_date - start_date).days + 1
    dm_full = _build_demand(daily, sids, start_date, total_days)
    print(f"Demand matrix: {n_ser} × {total_days} days  "
          f"({time.time()-t0:.0f}s)")

    # Load Chronos
    from chronos import BaseChronosPipeline
    import torch
    print(f"Loading Chronos-2 …")
    pipe = BaseChronosPipeline.from_pretrained(
        BASE_CHECKPOINT, device_map=args.device)

    # ---- Precompute monthly grids ----
    monthly_grids: dict[int, dict[str, list[np.ndarray]]] = {}
    for oi, month in enumerate(ORIGINS):
        ctx = hist_all[oi]
        ctx_sids = {s: ctx[s] for s in sids if s in ctx and len(ctx[s]) >= 30}
        n_missing = n_ser - len(ctx_sids)
        if n_missing > 0:
            print(f"  Origin {oi}: {len(ctx_sids)}/{n_ser} have history")
            fallback = np.zeros(30)
            for s in sids:
                if s not in ctx_sids:
                    ctx_sids[s] = fallback

        n_days_m = month.days_in_month + LEAD_DAYS
        grids: dict[str, list[np.ndarray]] = {}
        for sch in EMP_SCHEMES:
            g_emp = _emp_grid(sch, sids, ctx_sids)
            grids[sch] = [g_emp] * n_days_m

        q, _ = pipe.predict_quantiles(
            [torch.tensor(ctx_sids.get(s, np.zeros(30)),
                          dtype=torch.float32) for s in sids],
            prediction_length=n_days_m,
            quantile_levels=list(NATIVE_LEVELS),
            batch_size=args.batch_size)
        g, _ = QuantileRepair()(to_grid(q))
        g = g.reshape(n_ser, n_days_m, -1)
        grids["chronos2-zs"] = [g[:, d, :] for d in range(n_days_m)]

        monthly_grids[oi] = grids
        print(f"  Origin {oi} ({month:%Y-%m}) grids done  "
              f"({time.time()-t0:.0f}s)")

    # ---- Main replay loop ----
    arms = list(EMP_SCHEMES) + ["chronos2-zs"]
    long_rows = []

    for R in REVIEW_CADENCES:
        PI = R + LEAD_DAYS
        print(f"\n{'='*74}")
        print(f"R={R}  PI={PI}")
        print(f"{'='*74}")

        for arm in arms:
            for alpha in ALPHAS:
                h_c, p_c = costs_from_alpha(cost_i, alpha, KAPPA_H, 12)

                # ---- Long-run continuous replay ----
                review_days_list = []
                S_list = []
                for oi in range(len(ORIGINS)):
                    if oi not in monthly_grids:
                        continue
                    month = ORIGINS[oi]
                    mo = (month - start_date).days
                    gr = monthly_grids[oi][arm]
                    for dim in range(0, month.days_in_month, R):
                        abs_day = mo + dim
                        if abs_day >= total_days:
                            break
                        review_days_list.append(abs_day)
                        pi_grids = [gr[min(dim + k, len(gr) - 1)]
                                    for k in range(PI)]
                        pmf_pi = convolve_varying_pmf(
                            NATIVE_LEVELS, pi_grids, vmax=VMAX)
                        pmf_r = convolve_varying_pmf(
                            NATIVE_LEVELS, pi_grids[:R], vmax=VMAX)
                        S_val = order_up_to(
                            pmf_r, pmf_pi, alpha, PI / R)["P3"]
                        S_list.append(S_val)

                if not review_days_list:
                    continue
                S_arr = np.column_stack(S_list)
                rc = ReplayConfig(
                    n_days=total_days, lead_time_days=LEAD_DAYS,
                    review_cadence_days=R,
                    shortage_mechanism="lost_sales",
                    review_days=tuple(review_days_list))
                res_lr = replay(dm_full, S_arr, inv0, rc)

                # ---- Switch replay (R=30 only) ----
                res_sw_by_oi = {}
                if R == 30:
                    for oi in SCORED_ORIGINS:
                        if oi not in monthly_grids:
                            continue
                        month = ORIGINS[oi]
                        n_dm = month.days_in_month + LEAD_DAYS
                        mo = (month - start_date).days
                        if mo + n_dm > total_days:
                            break
                        dm_m = dm_full[:, mo:mo + n_dm]
                        gr = monthly_grids[oi][arm]
                        pmf_pi = convolve_varying_pmf(
                            NATIVE_LEVELS, gr[:n_dm], vmax=VMAX)
                        pmf_r = convolve_varying_pmf(
                            NATIVE_LEVELS, gr[:month.days_in_month],
                            vmax=VMAX)
                        m_ratio = n_dm / month.days_in_month
                        S_sw = order_up_to(
                            pmf_r, pmf_pi, alpha, m_ratio)["P3"]

                        inv_sw = monthly_inv[oi].copy()
                        nan_mask = np.isnan(inv_sw)
                        inv_sw[nan_mask] = 0.0

                        rc_sw = ReplayConfig(
                            n_days=n_dm, lead_time_days=LEAD_DAYS,
                            review_cadence_days=n_dm,
                            shortage_mechanism="lost_sales")
                        res_sw = replay(dm_m, S_sw[:, None], inv_sw, rc_sw)
                        res_sw_by_oi[oi] = (res_sw, S_sw, nan_mask)

                # ---- Score per scored origin ----
                for oi in SCORED_ORIGINS:
                    month = ORIGINS[oi]
                    mo = (month - start_date).days
                    me = mo + month.days_in_month
                    if me > total_days or mo < BURN_IN_DAYS:
                        continue

                    # Long-run cost
                    cost_lr = (h_c * res_lr.i_end[:, mo:me].mean(axis=1)
                               + p_c * res_lr.lost[:, mo:me].sum(axis=1))

                    # Static L0 (using full-month PI S)
                    if oi in monthly_grids:
                        n_dm = month.days_in_month + LEAD_DAYS
                        gr = monthly_grids[oi][arm]
                        pmf_pi_s = convolve_varying_pmf(
                            NATIVE_LEVELS, gr[:n_dm], vmax=VMAX)
                        pmf_r_s = convolve_varying_pmf(
                            NATIVE_LEVELS,
                            gr[:month.days_in_month], vmax=VMAX)
                        m_r = n_dm / month.days_in_month
                        S_static = order_up_to(
                            pmf_r_s, pmf_pi_s, alpha, m_r)["P3"]
                        pi_end = min(mo + n_dm, total_days)
                        y_pi = dm_full[:, mo:pi_end].sum(axis=1)
                        l0 = (h_c * np.maximum(S_static - y_pi, 0)
                              + p_c * np.maximum(y_pi - S_static, 0))
                    else:
                        l0 = np.full(n_ser, np.nan)

                    rec = dict(
                        series_id=sids, origin_idx=oi,
                        month=str(month.date()), alpha=alpha,
                        arm=arm, R=R,
                        L0_static=l0, L3_longrun=cost_lr)

                    # Switch cost
                    if oi in res_sw_by_oi:
                        res_sw, _, nan_mask = res_sw_by_oi[oi]
                        cost_sw = (h_c * res_sw.i_end.mean(axis=1)
                                   + p_c * res_sw.lost.sum(axis=1))
                        cost_sw[nan_mask] = np.nan
                        rec["L3_switch"] = cost_sw

                    metric_names = ["L0_static", "L3_longrun"]
                    if "L3_switch" in rec:
                        metric_names.append("L3_switch")
                    bad = {
                        name: int(np.count_nonzero(~np.isfinite(rec[name])))
                        for name in metric_names
                        if not np.isfinite(rec[name]).all()
                    }
                    if bad:
                        raise RuntimeError(
                            "Nonfinite replay outcomes on the predeclared "
                            f"eligible support (arm={arm}, alpha={alpha}, "
                            f"origin={oi}, R={R}): {bad}")
                    long_rows.append(pd.DataFrame(rec))

            print(f"  {arm}  ({time.time()-t0:.0f}s)")

    df = pd.concat(long_rows, ignore_index=True)
    df.to_csv(ART / "continuous_long_v2.csv", index=False)
    print(f"\nSaved {len(df)} rows  ({time.time()-t0:.0f}s)")

    # ======== JOINT ANALYSIS ========
    print(f"\n{'='*78}")
    print("JOINT ANALYSIS (common support, ratio-of-sums, joint cluster bootstrap)")
    print(f"{'='*78}")

    # Build joint table: (series, origin, alpha) with L0, L3_sw, L3_lr
    # for chronos vs retuned baseline, R=30
    df30 = df[df.R == 30]
    joint_recs = []
    expected_sids = pd.Index(sids, name="series_id")
    for alpha in HIGH_ALPHAS:
        for oi in SCORED_ORIGINS:
            ba = base_of.get((alpha, oi))
            if ba is None:
                raise RuntimeError(
                    f"Missing frozen empirical baseline for alpha={alpha}, "
                    f"origin={oi}")
            zs = df30[(df30.alpha == alpha) & (df30.origin_idx == oi)
                      & (df30.arm == "chronos2-zs")]
            emp = df30[(df30.alpha == alpha) & (df30.origin_idx == oi)
                       & (df30.arm == ba)]
            if zs.series_id.duplicated().any() or emp.series_id.duplicated().any():
                raise RuntimeError(
                    f"Duplicate R=30 policy rows for alpha={alpha}, origin={oi}")
            zs_i = zs.set_index("series_id").reindex(expected_sids)
            emp_i = emp.set_index("series_id").reindex(expected_sids)
            vals = np.column_stack([
                zs_i["L0_static"], emp_i["L0_static"],
                zs_i["L3_longrun"], emp_i["L3_longrun"],
                zs_i["L3_switch"], emp_i["L3_switch"]])
            if not np.isfinite(vals).all():
                n_bad = int(np.count_nonzero(~np.isfinite(vals)))
                raise RuntimeError(
                    "Missing/nonfinite R=30 cells on the predeclared support "
                    f"(alpha={alpha}, origin={oi}, bad_values={n_bad})")
            for sid, v in zip(expected_sids, vals):
                joint_recs.append((
                    sid, oi, alpha,
                    v[0] - v[1], v[1],  # d0, b0
                    v[4] - v[5], v[5],  # d_sw, b_sw
                    v[2] - v[3], v[3])) # d_lr, b_lr

    if not joint_recs:
        print("  No joint cells!")
        chk.n_rows = len(df)
        chk.finish(ART / "checks")
        return 0

    jdf = pd.DataFrame(joint_recs, columns=[
        "series_id", "origin_idx", "alpha",
        "d0", "b0", "d_sw", "b_sw", "d_lr", "b_lr"])
    expected_cells = n_ser * len(SCORED_ORIGINS) * len(HIGH_ALPHAS)
    if len(jdf) != expected_cells:
        raise RuntimeError(
            f"Expected {expected_cells} eligible joint cells, got {len(jdf)}")
    print(f"  Joint cells: {len(jdf)}, clusters: {jdf.series_id.nunique()}")

    # Cluster bootstrap setup
    codes, uniq = pd.factorize(jdf.series_id)
    n_clu = len(uniq)
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(n_clu))
    ends = np.searchsorted(codes[order], np.arange(n_clu), side="right")

    d0, b0 = jdf.d0.to_numpy(), jdf.b0.to_numpy()
    d_sw, b_sw = jdf.d_sw.to_numpy(), jdf.b_sw.to_numpy()
    d_lr, b_lr = jdf.d_lr.to_numpy(), jdf.b_lr.to_numpy()

    def _ratio_of_sums(d, b, rows):
        denom = b[rows].sum()
        if not np.isfinite(denom) or denom <= 0:
            raise RuntimeError(
                f"Nonpositive/nonfinite aggregate denominator: {denom}")
        return d[rows].sum() / denom * 100

    def _joint(rows):
        r0 = _ratio_of_sums(d0, b0, rows)
        r_sw = _ratio_of_sums(d_sw, b_sw, rows)
        r_lr = _ratio_of_sums(d_lr, b_lr, rows)
        return (r0, r_sw, r_lr,
                r_sw - r0, r_lr - r0, r_lr - r_sw)

    pt = _joint(np.arange(len(jdf)))
    boot_rng = np.random.default_rng(SEED_BASE)
    draws = np.empty((args.b, 6))
    for k in range(args.b):
        picks = boot_rng.integers(0, n_clu, size=n_clu)
        rows = np.concatenate([order[starts[c]:ends[c]] for c in picks])
        draws[k] = _joint(rows)

    lo = np.quantile(draws, .025, axis=0)
    hi = np.quantile(draws, .975, axis=0)

    labels = ["R0 (static)", "R3_switch", "R3_longrun",
              "A_SW = R3_sw−R0", "A_LR = R3_lr−R0", "D_reset = R3_lr−R3_sw"]
    units = ["%", "%", "%", " pp", " pp", " pp"]

    print(f"\n  {'Metric':<24} {'Point':>8} {'95% CI':>22} "
          f"{'p (2-sided)':>12}")
    print(f"  {'-'*68}")
    for i, (nm, u) in enumerate(zip(labels, units)):
        p2 = _bootstrap_p2(draws[:, i])
        sig = "***" if p2 < 0.001 else "**" if p2 < 0.01 else "*" if p2 < 0.05 else ""
        print(f"  {nm:<24} {pt[i]:>+7.2f}{u}  "
              f"[{lo[i]:>+7.2f}, {hi[i]:>+7.2f}]  "
              f"p={p2:.4f} {sig}")
        chk.note(f"joint_{nm.split('(')[0].strip().replace(' ','_').lower()}",
                 round(float(pt[i]), 3))

    # ---- 4-cell decomposition ----
    print(f"\n{'='*78}")
    print("4-CELL DECOMPOSITION (sample × origin effects)")
    print(f"{'='*78}")

    # Load transmission audit for full-sample comparison
    trans_file = cfgmod.ARTIFACT_DIR / "zhao_audit" / "audit_long_v2.csv"
    if trans_file.exists():
        trans = pd.read_csv(trans_file)
        piv_t = trans.pivot_table(
            index=["series_id", "origin_idx", "alpha"],
            columns="arm", values=["L0", "L3"])

        def _r0r3(piv, sids_f=None, origins_f=None):
            recs = []
            for (sid, oi, a), row in piv.iterrows():
                if a not in HIGH_ALPHAS:
                    continue
                if sids_f is not None and sid not in sids_f:
                    continue
                if origins_f is not None and oi not in origins_f:
                    continue
                ba = base_of.get((a, oi))
                if ba is None:
                    continue
                cols = [("L0", "chronos2-zs"), ("L0", ba),
                        ("L3", "chronos2-zs"), ("L3", ba)]
                if not all(c in piv.columns for c in cols):
                    continue
                v = [row[c] for c in cols]
                if not all(np.isfinite(x) for x in v):
                    continue
                recs.append((v[0]-v[1], v[1], v[2]-v[3], v[3]))
            if not recs:
                return "n/a", "n/a"
            dd = np.array(recs)
            r0 = dd[:, 0].sum() / dd[:, 1].sum() * 100
            r3 = dd[:, 2].sum() / dd[:, 3].sum() * 100
            return f"{r0:+.2f}%", f"{r3:+.2f}%"

        common_set = set(sids)
        print(f"\n  {'Sample':<25} {'Origins':<12} {'R0':>10} {'R3_sw':>10}")
        print(f"  {'-'*60}")
        for label, sf, of in [
            ("Full (1790)", None, None),
            ("Full (1790)", None, set(SCORED_ORIGINS)),
            ("Common (this run)", common_set, None),
            ("Common (this run)", common_set, set(SCORED_ORIGINS)),
        ]:
            olab = "0-7" if of is None else f"{min(of)}-{max(of)}"
            r0s, r3s = _r0r3(piv_t, sf, of)
            print(f"  {label:<25} {olab:<12} {r0s:>10} {r3s:>10}")

    # ---- CADENCE INTERACTION TEST ----
    print(f"\n{'='*78}")
    print("CADENCE INTERACTION TEST (common-support, paired cluster bootstrap)")
    print(f"{'='*78}")

    cad_recs = []
    for R in REVIEW_CADENCES:
        sub = df[df.R == R]
        for alpha in HIGH_ALPHAS:
            for oi in SCORED_ORIGINS:
                ba = base_of.get((alpha, oi))
                if ba is None:
                    raise RuntimeError(
                        f"Missing frozen baseline for alpha={alpha}, origin={oi}")
                zs = sub[(sub.alpha == alpha) & (sub.origin_idx == oi)
                         & (sub.arm == "chronos2-zs")]
                emp = sub[(sub.alpha == alpha) & (sub.origin_idx == oi)
                          & (sub.arm == ba)]
                if (zs.series_id.duplicated().any()
                        or emp.series_id.duplicated().any()):
                    raise RuntimeError(
                        f"Duplicate cadence rows for R={R}, alpha={alpha}, "
                        f"origin={oi}")
                zs_i = zs.set_index("series_id")["L3_longrun"].reindex(
                    expected_sids)
                emp_i = emp.set_index("series_id")["L3_longrun"].reindex(
                    expected_sids)
                pair = np.column_stack([zs_i, emp_i])
                if not np.isfinite(pair).all():
                    raise RuntimeError(
                        "Missing/nonfinite cadence outcomes on predeclared "
                        f"support (R={R}, alpha={alpha}, origin={oi})")
                for sid in expected_sids:
                    cad_recs.append((sid, oi, alpha, R,
                                    float(zs_i.loc[sid] - emp_i.loc[sid]),
                                    float(emp_i.loc[sid])))

    cdf = pd.DataFrame(cad_recs, columns=[
        "series_id", "origin_idx", "alpha", "R", "d3", "b3"])

    # Every cadence must use the same predeclared cells.
    cdf["key"] = list(zip(cdf.series_id, cdf.origin_idx, cdf.alpha))
    keys_per_R = {R: set(cdf[cdf.R == R].key) for R in REVIEW_CADENCES}
    expected_keys = {
        (sid, oi, alpha)
        for sid in sids
        for oi in SCORED_ORIGINS
        for alpha in HIGH_ALPHAS}
    for R in REVIEW_CADENCES:
        if keys_per_R[R] != expected_keys:
            raise RuntimeError(
                f"Cadence R={R} support differs from fixed eligibility "
                f"(missing={len(expected_keys - keys_per_R[R])}, "
                f"extra={len(keys_per_R[R] - expected_keys)})")
    common_keys = expected_keys
    cdf_c = cdf[cdf.key.isin(common_keys)].copy()
    cdf_c.drop(columns="key", inplace=True)
    print(f"  Common cells: {len(common_keys)} × 3 = {len(cdf_c)}")

    # Per-R point estimates on common support
    for R in REVIEW_CADENCES:
        s = cdf_c[cdf_c.R == R]
        r3 = _ratio_of_sums(
            s.d3.to_numpy(), s.b3.to_numpy(), np.arange(len(s)))
        print(f"  R={R:>2}  R3_longrun = {r3:>+7.2f}%")

    # Pairwise paired bootstrap
    codes_c, uniq_c = pd.factorize(cdf_c.series_id)
    n_clu_c = len(uniq_c)
    order_c = np.argsort(codes_c, kind="stable")
    starts_c = np.searchsorted(codes_c[order_c], np.arange(n_clu_c))
    ends_c = np.searchsorted(codes_c[order_c], np.arange(n_clu_c),
                              side="right")

    d3_by_r, b3_by_r = {}, {}
    for R in REVIEW_CADENCES:
        mask = cdf_c.R.to_numpy() == R
        d = np.zeros(len(cdf_c))
        b = np.zeros(len(cdf_c))
        d[mask] = cdf_c.d3.to_numpy()[mask]
        b[mask] = cdf_c.b3.to_numpy()[mask]
        d3_by_r[R] = d
        b3_by_r[R] = b

    rng_c = np.random.default_rng(SEED_BASE + 7777)
    print(f"\n  {'Pair':<10} {'Point':>8} {'95% CI':>22} {'p (2-sided)':>12}")
    print(f"  {'-'*55}")
    def _make_diff(ra, rb):
        def _diff(rows):
            r_a = _ratio_of_sums(d3_by_r[ra], b3_by_r[ra], rows)
            r_b = _ratio_of_sums(d3_by_r[rb], b3_by_r[rb], rows)
            return r_a - r_b
        return _diff

    for ra, rb in [(1, 7), (1, 30), (7, 30)]:
        _diff = _make_diff(ra, rb)
        pt_d = _diff(np.arange(len(cdf_c)))
        boot_d = np.empty(args.b)
        for k in range(args.b):
            picks = rng_c.integers(0, n_clu_c, size=n_clu_c)
            rows = np.concatenate([order_c[starts_c[c]:ends_c[c]]
                                   for c in picks])
            boot_d[k] = _diff(rows)
        lo_d = float(np.quantile(boot_d, .025))
        hi_d = float(np.quantile(boot_d, .975))
        p2 = _bootstrap_p2(boot_d)
        print(f"  R{ra}-R{rb}{'':<5} {pt_d:>+7.2f} pp  "
              f"[{lo_d:>+7.2f}, {hi_d:>+7.2f}]  p={p2:.4f}")

    # log(R) slope
    from scipy import stats as sp_stats
    logR = np.log(cdf_c.R.to_numpy().astype(float))
    # outcome = d3/b3 = (C_ZS - C_Emp)/C_Emp, negative = TSFM better
    rel = cdf_c.d3.to_numpy() / np.clip(cdf_c.b3.to_numpy(), 1e-9, None)
    oh = pd.get_dummies(cdf_c.origin_idx, drop_first=True).to_numpy(float)
    oa = pd.get_dummies(cdf_c.alpha, drop_first=True).to_numpy(float)
    X = np.column_stack([np.ones(len(rel)), logR, oh, oa])
    beta = np.linalg.lstsq(X, rel, rcond=None)[0]
    beta1 = beta[1]

    # Bootstrap β1 via sufficient statistics
    k_dim = X.shape[1]
    XtX_cl = np.zeros((n_clu_c, k_dim, k_dim))
    Xty_cl = np.zeros((n_clu_c, k_dim))
    for ci in range(n_clu_c):
        m = codes_c == ci
        Xc, yc = X[m], rel[m]
        XtX_cl[ci] = Xc.T @ Xc
        Xty_cl[ci] = Xc.T @ yc

    rng_b1 = np.random.default_rng(SEED_BASE + 8888)
    boot_b1 = np.empty(args.b)
    for bk in range(args.b):
        picks = rng_b1.integers(0, n_clu_c, size=n_clu_c)
        A = XtX_cl[picks].sum(0)
        bv = Xty_cl[picks].sum(0)
        try:
            boot_b1[bk] = np.linalg.solve(A, bv)[1]
        except np.linalg.LinAlgError:
            boot_b1[bk] = np.linalg.lstsq(A, bv, rcond=None)[0][1]

    p_b1 = _bootstrap_p2(boot_b1)
    print(f"\n  log(R) slope β1 = {beta1:+.6f}  "
          f"[{np.quantile(boot_b1, .025):+.6f}, "
          f"{np.quantile(boot_b1, .975):+.6f}]  p={p_b1:.4f}")
    print(f"  (outcome = (C_ZS−C_Emp)/C_Emp; negative = TSFM better)")
    print(f"  (β1 > 0 → TSFM advantage shrinks with lower review frequency)")
    print(f"  (β1 < 0 → TSFM advantage grows with lower review frequency)")

    chk.n_rows = len(df)
    chk.finish(ART / "checks")
    print(f"\nAll results saved to {ART}")
    print(f"Total time: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
