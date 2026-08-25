"""Pareto 前沿（§9.3）与冷启动切片分析。

  1. Dense Pareto sweep：对每个臂在 15 个 alpha 工作点提取订至水平、跑 layer_b，
     输出 (CSR, cost) 用于前沿绘图。PMF 只算一次/臂/月，alpha 扫描在 PMF 上做。
  2. 冷启动切片：context < 90 天的序列（~8%），在预测层和决策层分别比较。
  3. Bootstrap 检验：冷启动 vs warm，NPL 和成本差。

用法:  PYTHONPATH=src python -m f2d.run_zhao_analysis [--n-series 2000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from . import config as cfgmod
from .aggregation import convolve_varying_pmf, pmf_quantile
from .checks import CheckResult
from .conventions import SEED_BASE
from .datasets import zhao
from .decision import costs_from_alpha, implied_alpha, layer_b, costs_from_margin
from .metrics import evaluate_slice
from .models.chronos import BASE_CHECKPOINT, NATIVE_LEVELS, QuantileRepair, to_grid
from .models.gbdt_grid import LEAN_FEATURES, QuantileGridGBDT, make_lean_features
from .uncertainty import paired_bootstrap, paired_bootstrap_mean, report

ART = cfgmod.ARTIFACT_DIR / "zhao_analysis"
DECISION_ART = cfgmod.ARTIFACT_DIR / "zhao_decision"
PRED_ART = cfgmod.ARTIFACT_DIR / "zhao_daily"
VMAX = 60
LEAD_DAYS = 1
KAPPA_H = 0.20
COLD_THRESHOLD = 90

ALPHA_DENSE = [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95,
               0.96, 0.97, 0.98, 0.99, 0.995, 0.999]

ARMS = ("chronos2-zs", "chronos2-ft-full", "gbdt-lean", "emp-daily", "always-zero")
FT_CKPTS = {a: cfgmod.ARTIFACT_DIR / "zhao_finetune" / a / "ft"
            for a in ("chronos2-ft-full",)}

VALID_MONTHS = [pd.Timestamp("2019-07-01"), pd.Timestamp("2019-08-01")]
GBDT_TRAIN_ORIGINS = pd.date_range("2019-02-01", "2019-05-01", freq="MS")
VALID_ORIGINS_WEEKLY = pd.date_range("2019-07-01", "2019-08-26", freq="7D")


def _daily_grids(arm, sids, ctx, n_days, pipe, gbdt, feat, origin, batch_size,
                 ft_pipes=None):
    """日级分位网格。与 run_zhao_decision._daily_grids 同逻辑。"""
    if arm == "always-zero":
        return [np.zeros((len(sids), NATIVE_LEVELS.size)) for _ in range(n_days)]

    if arm == "emp-daily":
        emp = np.array([np.quantile(ctx[s], NATIVE_LEVELS, method="inverted_cdf")
                        for s in sids], float)
        emp, _ = QuantileRepair()(emp)
        return [emp] * n_days

    if arm == "chronos2-zs" or arm in FT_CKPTS:
        import torch
        p = pipe if arm == "chronos2-zs" else ft_pipes[arm]
        q, _ = p.predict_quantiles(
            [torch.tensor(ctx[s], dtype=torch.float32) for s in sids],
            prediction_length=n_days, quantile_levels=list(NATIVE_LEVELS),
            batch_size=batch_size)
        g, _ = QuantileRepair()(to_grid(q))
        g = g.reshape(len(sids), n_days, -1)
        return [g[:, i, :] for i in range(n_days)]

    if arm == "gbdt-lean":
        base = feat.set_index(["series_id", "d"]).reindex(
            pd.MultiIndex.from_product([sids, [origin]]))[LEAN_FEATURES]
        out = []
        for i in range(n_days):
            blk = base.copy()
            blk["h"] = i
            g = np.round(np.clip(gbdt.predict_grid(blk), 0.0, None))
            out.append(np.maximum.accumulate(g, axis=1))
        return out

    raise ValueError(arm)


def _pareto_sweep(daily, panel, raw, keep, chk, args):
    """Part 1: dense alpha sweep for Pareto frontier."""
    t0 = time.time()
    sku_of = dict(zip(daily.series_id, daily.sku_ID))
    feat = make_lean_features(daily)

    lt = raw["orders"].groupby("sku_ID")["normal_lead_time"].median()
    lt_exceeds = set(lt[lt > 7].index)

    max_h = max(m.days_in_month for m in VALID_MONTHS) + LEAD_DAYS
    fidx = feat.set_index(["series_id", "d"])
    tr = []
    for o in GBDT_TRAIN_ORIGINS:
        sids_tr = np.asarray(sorted(set(
            daily[daily.d < o - pd.Timedelta(days=30)].series_id)))
        b = fidx.reindex(pd.MultiIndex.from_product(
            [sids_tr, [o]]))[LEAN_FEATURES].dropna(how="all")
        bs = b.index.get_level_values(0).to_numpy()
        for h in range(max_h):
            blk = b.copy()
            blk["h"] = h
            blk["y"] = fidx.reindex(pd.MultiIndex.from_arrays(
                [bs, np.repeat(o + pd.Timedelta(days=h), len(bs))]
            ))["y"].to_numpy()
            tr.append(blk.reset_index(drop=True))
    train = pd.concat(tr, ignore_index=True).dropna(subset=["y"])
    gbdt = QuantileGridGBDT(features=LEAN_FEATURES + ["h"]).fit(train)
    print(f"GBDT 就绪 ({time.time() - t0:.0f}s)")

    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(
        BASE_CHECKPOINT, device_map=args.device, torch_dtype=torch.float32)
    ft_pipes = {a: BaseChronosPipeline.from_pretrained(
        str(c), device_map=args.device, torch_dtype=torch.float32)
        for a, c in FT_CKPTS.items()}

    margin = zhao.build_margin_block(raw, VALID_MONTHS).set_index(["sku_ID", "month"])
    ctx_lens = {}
    for month in VALID_MONTHS:
        lens = daily[daily.d < month].groupby("series_id").size()
        for s, n in lens.items():
            ctx_lens[(s, month)] = n

    rows = []
    for month in VALID_MONTHS:
        n_days = month.days_in_month + LEAD_DAYS
        snap = panel[panel.month == month].set_index("sku_ID")
        hist = daily[daily.d < month]
        ctx = {s: g.y.to_numpy() for s, g in hist.groupby("series_id")}
        sids = np.array([s for s in sorted(ctx)
                         if len(ctx[s]) >= 30 and sku_of[s] in snap.index])
        sk = [sku_of[s] for s in sids]
        cur = snap.loc[sk]

        ip = (cur["beginning_inventory"].to_numpy(float)
              + cur["on_order_inventory"].to_numpy(float))
        y = cur["observed_sales_next_month"].to_numpy(float)
        cost_i = cur["unit_cost_hist"].to_numpy(float)
        mg = margin.reindex(pd.MultiIndex.from_arrays(
            [sk, np.repeat(month, len(sk))]))
        margin_i = mg["margin_unit"].to_numpy(float)
        has_c = np.isfinite(cost_i) & np.array([k not in lt_exceeds for k in sk])
        cl = np.array([ctx_lens.get((s, month), 0) for s in sids])

        for arm in ARMS:
            grids = _daily_grids(arm, sids, ctx, n_days, pipe, gbdt, feat,
                                 month, args.batch_size, ft_pipes)
            pmf_pi = convolve_varying_pmf(NATIVE_LEVELS, grids, vmax=VMAX)

            for alpha in ALPHA_DENSE:
                S = pmf_quantile(pmf_pi, [alpha])[alpha]
                for sl_name, sl_mask in [("all", np.ones(len(sids), bool)),
                                          ("cold", cl < COLD_THRESHOLD),
                                          ("warm", cl >= COLD_THRESHOLD)]:
                    m = has_c & sl_mask
                    if m.sum() == 0:
                        continue
                    h, p = costs_from_alpha(cost_i[m], alpha, KAPPA_H, 12)
                    r = layer_b(S[m], ip[m], y[m], h, p)
                    rows.append(dict(
                        arm=arm, alpha=alpha, slice=sl_name,
                        month=str(month.date()), n=int(m.sum()),
                        CSR_ub=r.csr_upper_bound,
                        FR_ub=r.fill_rate_upper_bound,
                        cost_mean=float(r.cost.mean()),
                        lost_lb=r.lost_units_lower_bound,
                        S_med=float(np.median(S[m])),
                        avg_inv=r.avg_inventory))
            print(f"  {month:%Y-%m} {arm:<20s} ({time.time() - t0:.0f}s)")

    pareto = pd.DataFrame(rows)
    pareto.to_csv(ART / "pareto_sweep.csv", index=False)
    chk.note("pareto_n_points", len(pareto))
    return pareto


def _coldstart_prediction(daily, chk):
    """Part 2: cold-start vs warm on prediction layer (from existing predictions)."""
    pred = pd.read_parquet(PRED_ART / "predictions_validation.parquet")
    y_min = 1.0

    origins = sorted(pred.month.unique())
    ctx_len = {}
    for origin in origins:
        lens = daily[daily.d < origin].groupby("series_id").size()
        for s, n in lens.items():
            ctx_len[(s, origin)] = n
    pred["ctx_len"] = [ctx_len.get((s, o), 0)
                       for s, o in zip(pred.series_id, pred.month)]
    pred["cold"] = pred.ctx_len < COLD_THRESHOLD
    pred["split"] = "validation"

    main_arms = ["chronos2-zs", "chronos2-ft-full", "gbdt-lean", "gbdt-rich", "emp-daily"]
    results = []

    for sl_name, sl_col_val in [("cold", True), ("warm", False)]:
        sub = pred[pred.cold == sl_col_val]
        n = len(sub) // sub.variant.nunique()
        print(f"\n=== Prediction {sl_name} (n={n}) ===")
        for v in main_arms:
            sv = sub[sub.variant == v]
            r = evaluate_slice(sv.y, sv.q50, sv.q85, sv.w, y_min)
            results.append(dict(slice=sl_name, arm=v, NPL=r.npl,
                                cov50=r.cov_50_pos, cov85=r.cov_85_pos, n=r.n))
            print(f"  {v:22s}  NPL={r.npl:.5f}  cov50={r.cov_50_pos:.4f}"
                  f"  cov85={r.cov_85_pos:.4f}")

        print(f"\n  Bootstrap vs emp-daily:")
        others = [a for a in main_arms if a != "emp-daily"]
        cis = paired_bootstrap(sub, "emp-daily", others, y_min=y_min,
                               split="validation")
        rr = report(cis)
        print(rr[["variant", "delta", "lo", "hi", "verdict"]]
              .round(5).to_string(index=False))
        for c in cis:
            chk.note(f"pred_{sl_name}_{c.variant}",
                     [round(c.delta, 5), round(c.lo, 5), round(c.hi, 5),
                      c.significant])
            results.append(dict(slice=sl_name, arm=c.variant,
                                delta_npl=c.delta, lo=c.lo, hi=c.hi,
                                significant=c.significant))

    return pd.DataFrame(results)


def _coldstart_decision(daily, chk):
    """Part 3: cold-start decision layer from existing layer_b data."""
    df = pd.read_parquet(DECISION_ART / "layer_b_validation.parquet")
    has_c = df.unit_cost.notna() & df.lt_exceeds.eq(False)

    ctx_lens = {}
    for month in VALID_MONTHS:
        lens = daily[daily.d < month].groupby("series_id").size()
        for s, n in lens.items():
            ctx_lens[(s, month)] = n
    df["ctx_len"] = [ctx_lens.get((s, m), 0)
                     for s, m in zip(df.series_id, df.month)]
    df["cold"] = df.ctx_len < COLD_THRESHOLD

    main_arms = ["chronos2-zs", "chronos2-ft-full", "emp-daily", "gbdt-lean"]

    print("\n=== Decision layer cold-start (P3, alpha=0.95) ===")
    for sl_name, cold_val in [("cold", True), ("warm", False)]:
        sub = df[(df.policy == "P3") & (df.alpha == 0.95)
                 & (df.cold == cold_val) & has_c.loc[df.index]]
        n_each = len(sub) // len(main_arms)
        print(f"\n  {sl_name} (n={n_each}):")
        for arm in main_arms:
            g = sub[sub.arm == arm]
            if not len(g):
                continue
            h, p = costs_from_alpha(g.unit_cost.to_numpy(), 0.95, KAPPA_H, 12)
            r = layer_b(g.S.to_numpy(), g.ip.to_numpy(), g.y.to_numpy(), h, p)
            print(f"    {arm:25s} CSR={r.csr_upper_bound:.4f}"
                  f"  cost={r.cost.mean():.2f}  n={len(g)}")
            chk.note(f"decision_{sl_name}_{arm}",
                     {"CSR_ub": round(r.csr_upper_bound, 4),
                      "cost": round(float(r.cost.mean()), 2),
                      "n": len(g)})

    # Bootstrap cost difference on cold-start
    print("\n  Cost bootstrap on cold-start slice (P3, α=0.95):")
    cb = df[(df.policy == "P3") & (df.alpha == 0.95) & df.cold & has_c.loc[df.index]].copy()
    pos = np.maximum(cb.ip.to_numpy(), cb.S.to_numpy())
    h, p = costs_from_alpha(cb.unit_cost.to_numpy(), 0.95, KAPPA_H, 12)
    cb["row_cost"] = h * np.clip(pos - cb.y, 0, None) + p * np.clip(cb.y - pos, 0, None)
    others = [a for a in main_arms if a != "emp-daily"]
    cis = paired_bootstrap_mean(cb, "row_cost", "emp-daily", others, variant_col="arm")
    print(report(cis)[["variant", "delta", "lo", "hi", "verdict"]]
          .round(4).to_string(index=False))
    for c in cis:
        chk.note(f"decision_cold_cost_{c.variant}",
                 [round(c.delta, 4), round(c.lo, 4), round(c.hi, 4), c.significant])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=2000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--skip-pareto", action="store_true",
                    help="跳过 Pareto（需要模型推理），只跑冷启动切片")
    args = ap.parse_args(argv)

    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    chk = CheckResult(step_id="zhao_analysis", dataset="zhao", seed=SEED_BASE)

    raw = zhao.load_raw()
    daily, _ = zhao.build_daily_panel(raw)
    panel, _ = zhao.build_panel(raw)

    rng = np.random.default_rng(SEED_BASE)
    pool = np.asarray(sorted(set(daily.series_id)))
    keep = rng.choice(pool, size=min(args.n_series, len(pool)), replace=False)
    daily_sub = daily[daily.series_id.isin(keep)]

    # Part 1: Pareto
    if not args.skip_pareto:
        pareto = _pareto_sweep(daily_sub, panel, raw, keep, chk, args)
        print("\n=== Pareto 前沿 (P3, derived, all) ===")
        p_all = pareto[(pareto.slice == "all")]
        print(p_all.pivot_table(index="arm", columns="alpha",
                                values="CSR_ub", aggfunc="mean")
              .to_string(float_format="{:.4f}".format))
        print()
        print(p_all.pivot_table(index="arm", columns="alpha",
                                values="cost_mean", aggfunc="mean")
              .to_string(float_format="{:.2f}".format))

    # Part 2: cold-start prediction
    pred_results = _coldstart_prediction(daily, chk)
    pred_results.to_csv(ART / "coldstart_prediction.csv", index=False)

    # Part 3: cold-start decision
    _coldstart_decision(daily, chk)

    chk.note("cold_threshold_days", COLD_THRESHOLD)
    chk.note("wall_clock_sec", round(time.time() - t0, 1))
    chk.finish(ART / "checks" / "analysis.json")
    return chk.exit_code


if __name__ == "__main__":
    sys.exit(main())
