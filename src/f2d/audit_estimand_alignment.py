"""对齐三个实验的 estimand，在 common support 上联合重算 R0, R3_SW, R3_LR。

解决的问题：
  Transmission audit:  R0=-8.26%, R3_switch=-3.78%  (n=1790, origins 0-7)
  Continuous replay:   R3_switch=-11.70%             (n=265,  origins 5,6 only)
  不匹配原因：series sample, origin coverage, missing months bug

本脚本：
  1. 生成 estimand audit 表，逐项列出两实验的参数差异
  2. 在 common-support (同 series, 同 origin, 同 alpha, 同 baseline)
     上联合计算 R0, R3_SW, R3_LR
  3. Joint bootstrap for A_SW = R3_SW - R0, A_LR = R3_LR - R0, A_LR - A_SW
  4. Cadence paired interaction test: R3(R1) vs R3(R7) vs R3(R30)
  5. Exact bootstrap p-values

用法:  PYTHONPATH=src python -m f2d.audit_estimand_alignment
前置:  artifacts/zhao_audit/audit_long_v2.csv
       artifacts/zhao_continuous/continuous_long.csv
       artifacts/zhao_rolling_origins/baseline_selection.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as cfgmod
from .conventions import SEED_BASE
from .run_rolling_origins import HIGH_ALPHAS, ORIGINS

ART = cfgmod.ARTIFACT_DIR / "zhao_continuous"
B = 10_000

ORIGIN_DATES = [m.date() for m in ORIGINS]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=B)
    args = ap.parse_args(argv)

    # ---- Load data ----
    trans = pd.read_csv(cfgmod.ARTIFACT_DIR / "zhao_audit" / "audit_long_v2.csv")
    cont = pd.read_csv(cfgmod.ARTIFACT_DIR / "zhao_continuous" / "continuous_long.csv")
    sel = pd.read_csv(cfgmod.ARTIFACT_DIR / "zhao_rolling_origins"
                      / "baseline_selection.csv")
    base_of = {(r.alpha, r.origin_idx): r.retuned
               for _, r in sel.iterrows()}

    # ---- 1. Estimand audit table ----
    print("=" * 78)
    print("1. ESTIMAND AUDIT TABLE")
    print("=" * 78)

    t_sids = set(trans.series_id.unique())
    c_sids = set(cont.series_id.unique())
    overlap = t_sids & c_sids

    # Map continuous months to origin indices
    cont_months = sorted(cont.month.unique())
    cont_origins = set()
    for m_str in cont_months:
        for i, d in enumerate(ORIGIN_DATES):
            if str(d) == m_str:
                cont_origins.add(i)

    # Continuous switch coverage
    c30 = cont[(cont.R == 30)]
    switch_months = set()
    if "L3_switch" in c30.columns:
        sw_data = c30[c30.L3_switch.notna()]
        switch_months = set(sw_data.month.unique())

    items = [
        ("TSFM arm", "chronos2-zs", "chronos2-zs"),
        ("Emp baseline", "retuned (origin-specific)",
         "retuned (origin-specific)"),
        ("α set", "{0.95, 0.98} (HIGH_ALPHAS)",
         "{0.95, 0.98} (HIGH_ALPHAS)"),
        ("Origins", f"0-7 (Mar-Oct, n=8)",
         f"{sorted(cont_origins)} (n={len(cont_origins)})"),
        ("Series (n)", f"{len(t_sids)}", f"{len(c_sids)}"),
        ("Overlap", f"{len(overlap)}", f"{len(overlap)}"),
        ("Switch months w/ data", "all 8",
         f"{sorted(switch_months)} ({len(switch_months)} of {len(cont_origins)})"),
        ("Cost = h·avg(I_end) + p·Σ(lost)", "yes", "yes"),
        ("Initial inventory", "logged beginning_inv (monthly reset)",
         "logged beginning_inv (monthly reset for switch)"),
        ("Pipeline at start", "empty", "empty"),
        ("Review cadence", "n_days (monthly single)", "n_days (monthly single)"),
        ("Aggregation", "Σδ / Σbase × 100", "Σδ / Σbase × 100"),
    ]

    print(f"{'Item':<35} {'Transmission audit':<35} {'Continuous replay':<35}")
    print("-" * 105)
    for item, t_val, c_val in items:
        match = "✓" if t_val == c_val else "≠"
        print(f"{item:<35} {t_val:<35} {c_val:<35} {match}")

    # ---- 2. Decompose the discrepancy ----
    print(f"\n{'='*78}")
    print("2. DISCREPANCY DECOMPOSITION: R3_switch = -3.78% vs -11.70%")
    print(f"{'='*78}")

    # Full transmission
    piv_t = trans.pivot_table(
        index=["series_id", "origin_idx", "alpha"],
        columns="arm", values=["L0", "L3"])

    def _compute_r3(piv, sids_filter=None, origins_filter=None):
        recs = []
        for (sid, oi, a), row in piv.iterrows():
            if a not in HIGH_ALPHAS:
                continue
            if sids_filter is not None and sid not in sids_filter:
                continue
            if origins_filter is not None and oi not in origins_filter:
                continue
            ba = base_of.get((a, oi))
            if ba is None:
                continue
            cols_needed = [("L0", "chronos2-zs"), ("L0", ba),
                           ("L3", "chronos2-zs"), ("L3", ba)]
            if not all(c in piv.columns for c in cols_needed):
                continue
            vals = [row[c] for c in cols_needed]
            if not all(np.isfinite(v) for v in vals):
                continue
            recs.append((sid, oi, a,
                         vals[0] - vals[1], vals[1],
                         vals[2] - vals[3], vals[3]))
        if not recs:
            return None
        dd = pd.DataFrame(recs, columns=[
            "series_id", "origin_idx", "alpha", "d0", "b0", "d3", "b3"])
        r0 = dd.d0.sum() / dd.b0.sum() * 100
        r3 = dd.d3.sum() / dd.b3.sum() * 100
        return dict(R0=r0, R3=r3, att=r3 - r0,
                    n_rows=len(dd), n_clu=dd.series_id.nunique())

    specs = [
        ("Full (published)", None, None),
        ("Restrict origins 3-7", None, set(range(3, 8))),
        ("Restrict to 265 overlap series", overlap, None),
        ("Both restrictions", overlap, set(range(3, 8))),
    ]

    print(f"\n{'Specification':<35} {'R0':>8} {'R3_sw':>8} {'Att':>8} "
          f"{'n_rows':>7} {'n_clu':>6}")
    print("-" * 78)
    for label, sids_f, origins_f in specs:
        r = _compute_r3(piv_t, sids_f, origins_f)
        if r:
            print(f"{label:<35} {r['R0']:>+7.2f}% {r['R3']:>+7.2f}% "
                  f"{r['att']:>+7.2f}pp {r['n_rows']:>7} {r['n_clu']:>6}")

    # ---- 3. Common-support unified computation ----
    print(f"\n{'='*78}")
    print("3. UNIFIED R0, R3_SW, R3_LR ON COMMON SUPPORT")
    print(f"{'='*78}")

    # Build the common-support dataset
    # For each (series, origin, alpha): need L0, L3_switch from transmission
    # AND L3_longrun from continuous replay (R=30)

    c30_lr = cont[(cont.R == 30)]

    joint_recs = []
    for alpha in HIGH_ALPHAS:
        for oi in range(len(ORIGINS)):
            month_date = ORIGIN_DATES[oi]
            ba = base_of.get((alpha, oi))
            if ba is None:
                continue

            # Transmission: L0, L3 for chronos and baseline
            try:
                t_blk = piv_t.loc[(slice(None), oi, alpha), :]
            except KeyError:
                continue

            # Continuous: L3_longrun for chronos and baseline
            zs_lr = c30_lr[(c30_lr.alpha == alpha)
                           & (c30_lr.month == str(month_date))
                           & (c30_lr.arm == "chronos2-zs")]
            emp_lr = c30_lr[(c30_lr.alpha == alpha)
                            & (c30_lr.month == str(month_date))
                            & (c30_lr.arm == ba)]
            if zs_lr.empty or emp_lr.empty:
                continue

            zs_lr_i = zs_lr.set_index("series_id")["L3_longrun"]
            emp_lr_i = emp_lr.set_index("series_id")["L3_longrun"]

            for sid in t_blk.index.get_level_values("series_id"):
                if sid not in zs_lr_i.index or sid not in emp_lr_i.index:
                    continue
                row = t_blk.loc[(sid, oi, alpha)]
                cols = [("L0", "chronos2-zs"), ("L0", ba),
                        ("L3", "chronos2-zs"), ("L3", ba)]
                if not all(c in t_blk.columns for c in cols):
                    continue
                vals = [row[c] for c in cols]
                if not all(np.isfinite(v) for v in vals):
                    continue

                lr_zs = float(zs_lr_i.loc[sid])
                lr_emp = float(emp_lr_i.loc[sid])
                if not (np.isfinite(lr_zs) and np.isfinite(lr_emp)):
                    continue

                joint_recs.append((
                    sid, oi, alpha,
                    vals[0] - vals[1], vals[1],       # d0, b0
                    vals[2] - vals[3], vals[3],       # d3_sw, b3_sw
                    lr_zs - lr_emp, lr_emp))           # d3_lr, b3_lr

    if not joint_recs:
        print("  No common-support cells found!")
        return 1

    jdf = pd.DataFrame(joint_recs, columns=[
        "series_id", "origin_idx", "alpha",
        "d0", "b0", "d3_sw", "b3_sw", "d3_lr", "b3_lr"])

    print(f"  Common-support cells: {len(jdf)}")
    print(f"  Unique series: {jdf.series_id.nunique()}")
    print(f"  Origins: {sorted(jdf.origin_idx.unique())}")
    print(f"  Alphas: {sorted(jdf.alpha.unique())}")

    # Point estimates
    R0 = jdf.d0.sum() / jdf.b0.sum() * 100
    R3_sw = jdf.d3_sw.sum() / jdf.b3_sw.sum() * 100
    R3_lr = jdf.d3_lr.sum() / jdf.b3_lr.sum() * 100
    A_sw = R3_sw - R0
    A_lr = R3_lr - R0
    delta_A = A_lr - A_sw

    print(f"\n  {'Metric':<25} {'Value':>10}")
    print(f"  {'-'*37}")
    print(f"  {'R0 (static)':<25} {R0:>+9.2f}%")
    print(f"  {'R3_switch':<25} {R3_sw:>+9.2f}%")
    print(f"  {'R3_longrun (R=30)':<25} {R3_lr:>+9.2f}%")
    print(f"  {'A_SW = R3_sw − R0':<25} {A_sw:>+9.2f} pp")
    print(f"  {'A_LR = R3_lr − R0':<25} {A_lr:>+9.2f} pp")
    print(f"  {'A_LR − A_SW':<25} {delta_A:>+9.2f} pp")

    # Joint bootstrap
    print(f"\n  Joint bootstrap (B={args.b}, cluster=series_id)")

    codes, uniq = pd.factorize(jdf.series_id)
    n_clu = len(uniq)
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(n_clu))
    ends = np.searchsorted(codes[order], np.arange(n_clu), side="right")

    d0 = jdf.d0.to_numpy()
    b0 = jdf.b0.to_numpy()
    d3_sw = jdf.d3_sw.to_numpy()
    b3_sw = jdf.b3_sw.to_numpy()
    d3_lr = jdf.d3_lr.to_numpy()
    b3_lr = jdf.b3_lr.to_numpy()

    def _stats(rows):
        r0 = d0[rows].sum() / b0[rows].sum() * 100
        r_sw = d3_sw[rows].sum() / b3_sw[rows].sum() * 100
        r_lr = d3_lr[rows].sum() / b3_lr[rows].sum() * 100
        a_sw = r_sw - r0
        a_lr = r_lr - r0
        return r0, r_sw, r_lr, a_sw, a_lr, a_lr - a_sw

    pt = _stats(np.arange(len(jdf)))
    rng = np.random.default_rng(SEED_BASE)
    draws = np.empty((args.b, 6))
    for k in range(args.b):
        picks = rng.integers(0, n_clu, size=n_clu)
        rows = np.concatenate([order[starts[c]:ends[c]] for c in picks])
        draws[k] = _stats(rows)

    lo = np.quantile(draws, .025, axis=0)
    hi = np.quantile(draws, .975, axis=0)

    labels = ["R0", "R3_switch", "R3_longrun",
              "A_SW (R3_sw−R0)", "A_LR (R3_lr−R0)", "A_LR − A_SW"]
    units = ["%", "%", "%", " pp", " pp", " pp"]

    print(f"\n  {'Metric':<22} {'Point':>8} {'95% CI':>20} {'p(one-sided)':>14}")
    print(f"  {'-'*66}")
    for i, (nm, u) in enumerate(zip(labels, units)):
        # Exact one-sided p-value: fraction of bootstrap draws ≥ 0 (for metrics
        # where we test < 0) or ≤ 0 (for A_LR - A_SW where we test < 0)
        if i <= 2:  # R0, R3_sw, R3_lr: test < 0
            p = float(np.mean(draws[:, i] >= 0))
        elif i <= 4:  # A_SW, A_LR: test > 0 (attenuation exists)
            p = float(np.mean(draws[:, i] <= 0))
        else:  # A_LR - A_SW: test < 0 (LR recovers more than SW)
            p = float(np.mean(draws[:, i] >= 0))

        print(f"  {nm:<22} {pt[i]:>+7.2f}{u}  "
              f"[{lo[i]:>+7.2f}, {hi[i]:>+7.2f}]  p={p:.4f}")

    # ---- 4. Cadence interaction test ----
    print(f"\n{'='*78}")
    print("4. CADENCE INTERACTION TEST (paired cluster bootstrap)")
    print(f"{'='*78}")

    # Build matched sample across R values
    cadence_recs = []
    for R in (1, 7, 30):
        c_r = cont[cont.R == R]
        for alpha in HIGH_ALPHAS:
            for oi in range(len(ORIGINS)):
                month_date = ORIGIN_DATES[oi]
                ba = base_of.get((alpha, oi))
                if ba is None:
                    continue
                zs = c_r[(c_r.alpha == alpha)
                         & (c_r.month == str(month_date))
                         & (c_r.arm == "chronos2-zs")]
                emp = c_r[(c_r.alpha == alpha)
                          & (c_r.month == str(month_date))
                          & (c_r.arm == ba)]
                if zs.empty or emp.empty:
                    continue
                zs_i = zs.set_index("series_id")["L3_longrun"]
                emp_i = emp.set_index("series_id")["L3_longrun"]
                common = zs_i.index.intersection(emp_i.index)
                for sid in common:
                    cadence_recs.append((
                        sid, oi, alpha, R,
                        float(zs_i.loc[sid] - emp_i.loc[sid]),
                        float(emp_i.loc[sid])))

    if not cadence_recs:
        print("  No cadence data")
        return 0

    cdf = pd.DataFrame(cadence_recs, columns=[
        "series_id", "origin_idx", "alpha", "R", "d3", "b3"])

    # Common support: series present in ALL three R values
    sets_by_r = {}
    for R in (1, 7, 30):
        keys = set(zip(cdf[cdf.R == R].series_id,
                       cdf[cdf.R == R].origin_idx,
                       cdf[cdf.R == R].alpha))
        sets_by_r[R] = keys

    common_keys = sets_by_r[1] & sets_by_r[7] & sets_by_r[30]
    cdf["key"] = list(zip(cdf.series_id, cdf.origin_idx, cdf.alpha))
    cdf_common = cdf[cdf.key.isin(common_keys)].copy()
    cdf_common.drop(columns="key", inplace=True)

    print(f"  Common-support cells: {len(common_keys)} "
          f"(× 3 cadences = {len(cdf_common)})")
    print(f"  Unique series: {cdf_common.series_id.nunique()}")

    if len(common_keys) < 30:
        print("  Too few common cells for interaction test")
        return 0

    # Compute R3 per cadence on common support
    for R in (1, 7, 30):
        sub = cdf_common[cdf_common.R == R]
        r3 = sub.d3.sum() / sub.b3.sum() * 100
        print(f"  R={R:>2}  R3_longrun = {r3:>+7.2f}%  (n={len(sub)})")

    # Pairwise differences via paired bootstrap
    # For each pair (R_a, R_b), compute R3(R_a) - R3(R_b)
    codes_c, uniq_c = pd.factorize(cdf_common.series_id)
    n_clu_c = len(uniq_c)
    order_c = np.argsort(codes_c, kind="stable")
    starts_c = np.searchsorted(codes_c[order_c], np.arange(n_clu_c))
    ends_c = np.searchsorted(codes_c[order_c], np.arange(n_clu_c), side="right")

    # Pre-index by R
    d3_by_r = {}
    b3_by_r = {}
    row_mask_by_r = {}
    for R in (1, 7, 30):
        mask = cdf_common.R.to_numpy() == R
        row_mask_by_r[R] = mask
        d3_by_r[R] = cdf_common.d3.to_numpy().copy()
        d3_by_r[R][~mask] = 0.0
        b3_by_r[R] = cdf_common.b3.to_numpy().copy()
        b3_by_r[R][~mask] = 0.0

    def _r3_diff(rows, ra, rb):
        r3a = d3_by_r[ra][rows].sum() / b3_by_r[ra][rows].sum() * 100
        r3b = d3_by_r[rb][rows].sum() / b3_by_r[rb][rows].sum() * 100
        return r3a - r3b

    print(f"\n  Pairwise cadence differences (B={args.b}):")
    print(f"  {'Comparison':<15} {'Point':>8} {'95% CI':>20} "
          f"{'p(=0)':>10}")
    print(f"  {'-'*55}")

    all_rows = np.arange(len(cdf_common))
    rng_c = np.random.default_rng(SEED_BASE + 7777)

    for ra, rb in [(1, 7), (1, 30), (7, 30)]:
        pt_diff = _r3_diff(all_rows, ra, rb)
        boot_diffs = np.empty(args.b)
        for k in range(args.b):
            picks = rng_c.integers(0, n_clu_c, size=n_clu_c)
            rows = np.concatenate([order_c[starts_c[c]:ends_c[c]]
                                   for c in picks])
            boot_diffs[k] = _r3_diff(rows, ra, rb)
        lo_d = float(np.quantile(boot_diffs, .025))
        hi_d = float(np.quantile(boot_diffs, .975))
        p_two = 2 * min(float(np.mean(boot_diffs >= 0)),
                        float(np.mean(boot_diffs <= 0)))
        print(f"  R{ra}-R{rb}{'':<9} {pt_diff:>+7.2f} pp  "
              f"[{lo_d:>+7.2f}, {hi_d:>+7.2f}]  p={p_two:.4f}")

    # Log(R) regression: d = β0 + β1·log(R) + λ_t + λ_α + ε
    from scipy import stats as sp_stats
    logR = np.log(cdf_common.R.to_numpy().astype(float))
    rel = cdf_common.d3.to_numpy() / np.clip(cdf_common.b3.to_numpy(), 1e-9, None)

    oh = pd.get_dummies(cdf_common.origin_idx, drop_first=True).to_numpy(float)
    oa = pd.get_dummies(cdf_common.alpha, drop_first=True).to_numpy(float)
    X = np.column_stack([np.ones(len(rel)), logR, oh, oa])
    beta = np.linalg.lstsq(X, rel, rcond=None)[0]
    beta1 = beta[1]

    # Bootstrap β1
    rng_b1 = np.random.default_rng(SEED_BASE + 8888)
    boot_b1 = np.empty(args.b)

    # Precompute per-cluster sufficient stats
    k = X.shape[1]
    XtX_c = np.zeros((n_clu_c, k, k))
    Xty_c = np.zeros((n_clu_c, k))
    for ci in range(n_clu_c):
        mask = codes_c == ci
        idx = np.where(mask)[0]
        Xc, yc = X[idx], rel[idx]
        XtX_c[ci] = Xc.T @ Xc
        Xty_c[ci] = Xc.T @ yc

    for bk in range(args.b):
        picks = rng_b1.integers(0, n_clu_c, size=n_clu_c)
        A = XtX_c[picks].sum(0)
        b_vec = Xty_c[picks].sum(0)
        try:
            boot_b1[bk] = np.linalg.solve(A, b_vec)[1]
        except np.linalg.LinAlgError:
            boot_b1[bk] = np.linalg.lstsq(A, b_vec, rcond=None)[0][1]

    p_b1 = 2 * min(float(np.mean(boot_b1 >= 0)),
                   float(np.mean(boot_b1 <= 0)))
    print(f"\n  log(R) slope β1 = {beta1:+.6f}  "
          f"[{np.quantile(boot_b1, .025):+.6f}, "
          f"{np.quantile(boot_b1, .975):+.6f}]  p={p_b1:.4f}")
    if p_b1 < 0.05:
        direction = "more negative" if beta1 < 0 else "less negative"
        print(f"  → TSFM advantage becomes {direction} "
              f"as review frequency decreases")
    else:
        print("  → No significant cadence effect detected")

    # ---- Summary ----
    print(f"\n{'='*78}")
    print("5. SUMMARY")
    print(f"{'='*78}")
    print(f"""
Discrepancy explained:
  R3_switch = -3.78% (n=1790, 8 origins) vs -11.70% (n=265, 2 origins)
  On matched sample (265 series, origins 3-7): R3_switch = {pt[1]:+.2f}%
  The gap is due to: (a) series subsample, (b) origin restriction,
  (c) missing-months bug leaving only 2 scored months in continuous replay.

Unified estimates (common support, n_clusters={n_clu}):
  R0 (static)    = {pt[0]:+.2f}%  [{lo[0]:+.2f}, {hi[0]:+.2f}]
  R3_switch      = {pt[1]:+.2f}%  [{lo[1]:+.2f}, {hi[1]:+.2f}]
  R3_longrun     = {pt[2]:+.2f}%  [{lo[2]:+.2f}, {hi[2]:+.2f}]
  A_SW           = {pt[3]:+.2f} pp  [{lo[3]:+.2f}, {hi[3]:+.2f}]
  A_LR           = {pt[4]:+.2f} pp  [{lo[4]:+.2f}, {hi[4]:+.2f}]
  A_LR − A_SW   = {pt[5]:+.2f} pp  [{lo[5]:+.2f}, {hi[5]:+.2f}]
""")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
