"""WS-3 Decision-aware fine-tuning。

三个微调臂对比：
  chronos2-ft-pinball     标准 pinball loss（对照，与 run_zhao_finetune 等价）
  chronos2-ft-newsvendor  只在 α=0.95 最近的分位数上训练
  chronos2-ft-focused     α-focused 加权 pinball（高斯核 σ=0.15）

评价：预测层（NPL）+ 决策层（Layer B cost, P3, α=0.95）。
如果 newsvendor/focused 在决策层显著优于 pinball，则打破了
"微调用服务换成本"的僵局。

用法:  PYTHONPATH=src python -m f2d.run_zhao_finetune_decision [--device mps]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from . import config as cfgmod
from .aggregation import convolve_varying, convolve_varying_pmf
from .checks import CheckResult
from .conventions import SEED_BASE
from .datasets import zhao
from .decision import POLICIES, costs_from_alpha, layer_b, order_up_to
from .losses import patched_loss
from .metrics import evaluate_slice
from .models.chronos import BASE_CHECKPOINT, NATIVE_LEVELS, QuantileRepair, to_grid
from .uncertainty import paired_bootstrap, report

ART = cfgmod.ARTIFACT_DIR / "zhao_finetune_decision"
FT_DIR = cfgmod.ARTIFACT_DIR / "zhao_finetune"
HORIZON = 7
VMAX_PRED = 300
VMAX_DEC = 60
CONTEXT_LENGTH = 400
MIN_PAST = max(2 * HORIZON, 28)
NUM_STEPS = 1000
BATCH_SIZE = 32
ALPHA_PRIMARY = 0.95
KAPPA_H = 0.20
LEAD_DAYS = 1

TRAIN_END = pd.Timestamp("2019-06-30")
VALID_ORIGINS = pd.date_range("2019-07-01", "2019-08-26", freq="7D")
VALID_MONTHS = [pd.Timestamp("2019-07-01"), pd.Timestamp("2019-08-01")]

LOSS_CONFIGS = [
    ("chronos2-ft-pinball", "pinball", {}),
    ("chronos2-ft-newsvendor", "newsvendor", {}),
    ("chronos2-ft-focused", "alpha_focused", {"focus_width": 0.15}),
]


def _windows(ft_horizon):
    ivt = TRAIN_END - pd.Timedelta(days=ft_horizon - 1)
    return ivt - pd.Timedelta(days=1), ivt


def _series_dict(daily, upto):
    h = daily[daily.d <= upto]
    return {s: g.sort_values("d").y.to_numpy(float)
            for s, g in h.groupby("series_id")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=2000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args(argv)

    import torch
    from chronos import BaseChronosPipeline

    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    chk = CheckResult(step_id="ws3_decision_ft", dataset="zhao", seed=SEED_BASE)
    cfg = cfgmod.load("zhao")
    y_min = float(cfg.metric["y_min"])

    raw = zhao.load_raw()
    daily, _ = zhao.build_daily_panel(raw)
    weekly = zhao.aggregate_to_period(daily, "W")
    panel, _ = zhao.build_panel(raw)
    target = weekly[weekly.origin.isin(VALID_ORIGINS)][["series_id", "origin", "y"]]

    rng = np.random.default_rng(SEED_BASE)
    pool = np.asarray(sorted(set(target.series_id)))
    keep = rng.choice(pool, size=min(args.n_series, len(pool)), replace=False)
    target = target[target.series_id.isin(keep)].reset_index(drop=True)
    daily = daily[daily.series_id.isin(keep)]
    sku_of = dict(zip(daily.series_id, daily.sku_ID))

    train_cutoff, inner_start = _windows(HORIZON)
    tr = _series_dict(daily, train_cutoff)
    iv = _series_dict(daily, TRAIN_END)
    tr = {s: v for s, v in tr.items() if len(v) >= MIN_PAST + HORIZON}
    iv = {s: v for s, v in iv.items() if len(v) >= MIN_PAST + HORIZON}
    chk.assert_true("微调输入不越过内部验证目标段",
                    bool(train_cutoff < inner_start))
    chk.assert_true("内部验证不越过验证窗",
                    bool(TRAIN_END < VALID_ORIGINS[0]))
    chk.note("n_train_series", len(tr))

    train_inputs = [torch.tensor(v, dtype=torch.float32) for v in tr.values()]
    val_inputs = [torch.tensor(v, dtype=torch.float32) for v in iv.values()]

    # ================================================================
    # Phase 1: Fine-tune with three different losses
    # ================================================================
    ft_pipes = {}
    for arm_name, loss_name, loss_kwargs in LOSS_CONFIGS:
        print(f"\n=== {arm_name}: loss={loss_name} ===")
        with patched_loss(loss_name, alpha=ALPHA_PRIMARY, **loss_kwargs):
            pipe = BaseChronosPipeline.from_pretrained(
                BASE_CHECKPOINT, device_map=args.device, torch_dtype=torch.float32)
            torch.manual_seed(SEED_BASE)
            ft = pipe.fit(
                inputs=train_inputs, validation_inputs=val_inputs,
                prediction_length=HORIZON, context_length=CONTEXT_LENGTH,
                num_steps=NUM_STEPS, batch_size=BATCH_SIZE, min_past=MIN_PAST,
                output_dir=FT_DIR / arm_name, finetuned_ckpt_name="ft",
                remove_printer_callback=True, disable_data_parallel=True,
                finetune_mode="full", learning_rate=1e-6)
        ft_pipes[arm_name] = ft
        print(f"  微调完毕 ({time.time() - t0:.0f}s)")

    # Also load zero-shot for comparison
    zs_pipe = BaseChronosPipeline.from_pretrained(
        BASE_CHECKPOINT, device_map=args.device, torch_dtype=torch.float32)

    # ================================================================
    # Phase 2: Evaluate prediction layer (NPL)
    # ================================================================
    print("\n=== 预测层评价 ===")
    all_arms = {"chronos2-zs": zs_pipe, **ft_pipes}
    pred_rows = []

    for arm_name, p in all_arms.items():
        for origin in VALID_ORIGINS:
            cur = target[target.origin == origin]
            sids = cur.series_id.to_numpy()
            hist = daily[(daily.d < origin) & (daily.series_id.isin(sids))]
            ctx = {s: g.y.to_numpy() for s, g in hist.groupby("series_id")}
            sids = np.array([s for s in sids if len(ctx.get(s, [])) >= 30])
            if not len(sids):
                continue
            cur = cur[cur.series_id.isin(sids)]
            q, _ = p.predict_quantiles(
                [torch.tensor(ctx[s], dtype=torch.float32) for s in sids],
                prediction_length=HORIZON, quantile_levels=list(NATIVE_LEVELS),
                batch_size=args.batch_size)
            g, _ = QuantileRepair()(to_grid(q))
            g = g.reshape(len(sids), HORIZON, -1)
            r = convolve_varying(NATIVE_LEVELS, [g[:, h, :] for h in range(HORIZON)],
                                 taus=(.5, .85), vmax=VMAX_PRED)
            pred_rows.append(pd.DataFrame({
                "variant": arm_name, "series_id": sids, "month": origin,
                "split": "validation",
                "y": cur.set_index("series_id").loc[sids, "y"].to_numpy(),
                "q50": r[.5], "q85": r[.85], "w": 1.0}))

    pred = pd.concat(pred_rows, ignore_index=True)

    # Align paired rows
    keys = {v: set(map(tuple, pred[pred.variant == v][["series_id", "month"]].to_numpy()))
            for v in pred.variant.unique()}
    common = set.intersection(*keys.values())
    pred = pred[[k in common for k in
                 map(tuple, pred[["series_id", "month"]].to_numpy())]]
    pred = pred.sort_values(["variant", "month", "series_id"]).reset_index(drop=True)

    print("\n" + "=" * 66)
    print("%-24s %-10s %-11s %-11s %-8s" % ("model", "NPL", "cov50_pos", "cov85_pos", "n"))
    for v in all_arms:
        s = pred[pred.variant == v]
        r = evaluate_slice(s.y, s.q50, s.q85, s.w, y_min)
        print("%-24s %-10.5f %-11.4f %-11.4f %-8d" % (
            v, r.npl, r.cov_50_pos, r.cov_85_pos, r.n))
        chk.note(f"npl_{v}", round(r.npl, 5))

    print("\n配对 bootstrap NPL（基准 = chronos2-zs）")
    cis = paired_bootstrap(pred, "chronos2-zs",
                           [v for v in all_arms if v != "chronos2-zs"],
                           y_min=y_min, split="validation")
    print(report(cis)[["variant", "delta", "lo", "hi", "verdict"]]
          .round(5).to_string(index=False))
    for c in cis:
        chk.note(f"npl_delta_{c.variant}",
                 [round(c.delta, 5), round(c.lo, 5), round(c.hi, 5), c.significant])

    # ================================================================
    # Phase 3: Evaluate decision layer (Layer B, P3, α=0.95)
    # ================================================================
    print("\n=== 决策层评价 (Layer B, P3, α=0.95) ===")

    margin_block = zhao.build_margin_block(raw, VALID_MONTHS).set_index(
        ["sku_ID", "month"])
    decision_rows = []

    for month in VALID_MONTHS:
        n_days = month.days_in_month + LEAD_DAYS
        snap = panel[panel.month == month].set_index("sku_ID")
        hist = daily[daily.d < month]
        ctx = {s: g.y.to_numpy() for s, g in hist.groupby("series_id")}
        sids = np.array([s for s in sorted(ctx)
                         if len(ctx[s]) >= 30 and sku_of[s] in snap.index])
        sk = np.array([sku_of[s] for s in sids])
        cur = snap.loc[sk]

        ip = cur["beginning_inventory"].to_numpy(float) + cur["on_order_inventory"].to_numpy(float)
        y = cur["observed_sales_next_month"].to_numpy(float)
        cost_i = cur["unit_cost_hist"].to_numpy(float)
        has_c = np.isfinite(cost_i)

        for arm_name, p in all_arms.items():
            q, _ = p.predict_quantiles(
                [torch.tensor(ctx[s], dtype=torch.float32) for s in sids],
                prediction_length=n_days, quantile_levels=list(NATIVE_LEVELS),
                batch_size=args.batch_size)
            g, _ = QuantileRepair()(to_grid(q))
            g = g.reshape(len(sids), n_days, -1)
            grids = [g[:, i, :] for i in range(n_days)]

            pmf_pi = convolve_varying_pmf(NATIVE_LEVELS, grids, vmax=VMAX_DEC)
            pmf_r = convolve_varying_pmf(NATIVE_LEVELS,
                                          grids[:month.days_in_month], vmax=VMAX_DEC)
            m_ratio = n_days / month.days_in_month
            S = order_up_to(pmf_r, pmf_pi, ALPHA_PRIMARY, m_ratio)["P3"]

            h_c, p_c = costs_from_alpha(cost_i, ALPHA_PRIMARY, KAPPA_H, 12)
            r = layer_b(S, ip, y, h_c, p_c)

            decision_rows.append(dict(
                month=month, arm=arm_name,
                n=len(sids), n_costed=int(has_c.sum()),
                CSR_ub=r.csr_upper_bound,
                FR_ub=r.fill_rate_upper_bound,
                cost=float(r.cost[has_c].mean()) if has_c.any() else float("nan"),
                hold=float((h_c * r.i_end)[has_c].mean()) if has_c.any() else float("nan"),
                short=float((p_c * r.observed_shortage_lower_bound)[has_c].mean()) if has_c.any() else float("nan"),
                S_med=float(np.median(S)),
            ))
            print(f"  {month:%Y-%m} {arm_name:<24} CSR={r.csr_upper_bound:.4f} "
                  f"cost={decision_rows[-1]['cost']:.2f} ({time.time() - t0:.0f}s)")

    dec_df = pd.DataFrame(decision_rows)
    dec_df.to_csv(ART / "decision_results.csv", index=False)

    # Aggregate across months
    print("\n--- 决策层两月平均 ---")
    dec_agg = dec_df.groupby("arm")[["CSR_ub", "FR_ub", "cost", "hold", "short", "S_med"]].mean()
    print(dec_agg.round(4).to_string())

    for arm in all_arms:
        chk.note(f"decision_cost_{arm}", round(float(dec_agg.loc[arm, "cost"]), 4))
        chk.note(f"decision_csr_{arm}", round(float(dec_agg.loc[arm, "CSR_ub"]), 4))

    # Cost delta bootstrap (paired by series)
    print(f"\n成本差 bootstrap（基准 = chronos2-zs, P3, α={ALPHA_PRIMARY}）")
    # Build per-row costs for bootstrap
    boot_frames = []
    for month in VALID_MONTHS:
        snap = panel[panel.month == month].set_index("sku_ID")
        hist = daily[daily.d < month]
        ctx = {s: g.y.to_numpy() for s, g in hist.groupby("series_id")}
        sids = np.array([s for s in sorted(ctx)
                         if len(ctx[s]) >= 30 and sku_of[s] in snap.index])
        sk = np.array([sku_of[s] for s in sids])
        cur = snap.loc[sk]
        n_days = month.days_in_month + LEAD_DAYS
        ip = cur["beginning_inventory"].to_numpy(float) + cur["on_order_inventory"].to_numpy(float)
        y = cur["observed_sales_next_month"].to_numpy(float)
        cost_i = cur["unit_cost_hist"].to_numpy(float)
        has_c = np.isfinite(cost_i)
        h_c, p_c = costs_from_alpha(cost_i, ALPHA_PRIMARY, KAPPA_H, 12)

        for arm_name, p in all_arms.items():
            q, _ = p.predict_quantiles(
                [torch.tensor(ctx[s], dtype=torch.float32) for s in sids],
                prediction_length=n_days, quantile_levels=list(NATIVE_LEVELS),
                batch_size=args.batch_size)
            g, _ = QuantileRepair()(to_grid(q))
            g = g.reshape(len(sids), n_days, -1)
            grids = [g[:, i, :] for i in range(n_days)]
            pmf_pi = convolve_varying_pmf(NATIVE_LEVELS, grids, vmax=VMAX_DEC)
            pmf_r = convolve_varying_pmf(NATIVE_LEVELS,
                                          grids[:month.days_in_month], vmax=VMAX_DEC)
            m_ratio = n_days / month.days_in_month
            S = order_up_to(pmf_r, pmf_pi, ALPHA_PRIMARY, m_ratio)["P3"]
            pos = np.maximum(ip, S)
            row_cost = h_c * np.clip(pos - y, 0, None) + p_c * np.clip(y - pos, 0, None)

            boot_frames.append(pd.DataFrame({
                "arm": arm_name, "series_id": sids, "month": month,
                "row_cost": row_cost, "has_cost": has_c,
            }))

    from .uncertainty import paired_bootstrap_mean
    boot_df = pd.concat(boot_frames, ignore_index=True)
    boot_df = boot_df[boot_df.has_cost]
    cost_cis = paired_bootstrap_mean(
        boot_df, "row_cost", "chronos2-zs",
        [a for a in all_arms if a != "chronos2-zs"],
        variant_col="arm")
    print(report(cost_cis)[["variant", "delta", "lo", "hi", "verdict"]]
          .round(4).to_string(index=False))
    for c in cost_cis:
        chk.note(f"cost_delta_{c.variant}",
                 [round(c.delta, 4), round(c.lo, 4), round(c.hi, 4), c.significant])

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("WS-3 Decision-aware fine-tuning 汇总")
    print("=" * 70)
    for arm in all_arms:
        npl = chk.notes.get(f"npl_{arm}", "?")
        cost = chk.notes.get(f"decision_cost_{arm}", "?")
        csr = chk.notes.get(f"decision_csr_{arm}", "?")
        print(f"  {arm:<24}  NPL={npl}  Cost={cost}  CSR={csr}")

    chk.n_rows = len(pred) + len(dec_df)
    chk.note("wall_clock_sec", round(time.time() - t0, 1))
    chk.finish(ART / "checks" / "ws3.json")
    print(f"\n总耗时 {time.time() - t0:.0f}s, 状态: {chk.status}")
    return chk.exit_code


if __name__ == "__main__":
    sys.exit(main())
