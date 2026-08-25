"""WS-3 on OSA: Decision-aware fine-tuning with L=7/14.

OSA 的长 lead time 使策略差异显著化，可能揭示 decision-aware loss 的决策层增量。
评价通过 Layer C 连续回放 (test window 2021-01-04, 84天)。

三个微调臂:
  chronos2-ft-pinball     标准 pinball loss
  chronos2-ft-newsvendor  newsvendor loss (α-focused single quantile)
  chronos2-ft-focused     α-focused 加权 pinball (σ=0.15)

对照:
  chronos2-zs             零样本
  emp-daily               经验法

用法:  PYTHONPATH=src python -m f2d.run_osa_finetune_decision [--device mps]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch

from . import config as cfgmod
from .aggregation import convolve_varying, convolve_varying_pmf
from .checks import CheckResult
from .conventions import SEED_BASE
from .datasets import osa
from .decision import costs_from_alpha, order_up_to
from .losses import patched_loss
from .metrics import evaluate_slice
from .models.chronos import BASE_CHECKPOINT, NATIVE_LEVELS, QuantileRepair, to_grid
from .simulation import ReplayConfig, replay, replay_metrics

ART = cfgmod.ARTIFACT_DIR / "osa_finetune_decision"
FT_DIR = cfgmod.ARTIFACT_DIR / "osa_finetune"

HORIZON = 7
CONTEXT_LENGTH = 400
MIN_PAST = 28
NUM_STEPS = 1000
BATCH_SIZE = 32
VMAX_PRED = 300
VMAX_DEC = 300
MIN_CONTEXT = 30

ALPHA_PRIMARY = 0.95
KAPPA_H = 0.20
UNIT_COST = 1.0  # OSA 无成本数据，用名义单位成本
LEAD_TIMES = [7, 14]
REVIEW_CADENCE = 7

LOSS_CONFIGS = [
    ("chronos2-ft-pinball", "pinball", {}),
    ("chronos2-ft-newsvendor", "newsvendor", {}),
    ("chronos2-ft-focused", "alpha_focused", {"focus_width": 0.15}),
]


def _daily_grids_from_pipe(pipe, sids, ctx, n_days, batch_size):
    q, _ = pipe.predict_quantiles(
        [torch.tensor(ctx[s], dtype=torch.float32) for s in sids],
        prediction_length=n_days, quantile_levels=list(NATIVE_LEVELS),
        batch_size=batch_size)
    g, _ = QuantileRepair()(to_grid(q))
    g = g.reshape(len(sids), n_days, -1)
    return [g[:, i, :] for i in range(n_days)]


def _emp_grids(sids, ctx, n_days):
    emp = np.array([np.quantile(ctx[s], NATIVE_LEVELS, method="inverted_cdf")
                    for s in sids], float)
    emp, _ = QuantileRepair()(emp)
    return [emp] * n_days


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)

    from chronos import BaseChronosPipeline

    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    chk = CheckResult(step_id="ws3_osa_decision_ft", dataset="osa", seed=SEED_BASE)
    cfg = cfgmod.load("osa")

    raw = osa.load_raw()
    daily, audit = osa.build_daily_panel(raw)
    print(f"面板: {audit['n_rows']} 行, {audit['n_series']} 序列")

    # ================================================================
    # Phase 0: Fine-tune on OSA training data
    # ================================================================
    train_end = pd.Timestamp(cfg.splits["train"]["origins"][1])
    train_cutoff = train_end - pd.Timedelta(days=HORIZON - 1)
    val_end = pd.Timestamp(cfg.splits["validation"]["origins"][1])

    hist_train = daily[daily.d <= train_cutoff]
    hist_val = daily[daily.d <= val_end]

    def _series_tensors(df, min_len):
        out = []
        for s, g in df.groupby("series_id"):
            v = g.sort_values("d").y.to_numpy(float)
            if len(v) >= min_len:
                out.append(torch.tensor(v, dtype=torch.float32))
        return out

    train_inputs = _series_tensors(hist_train, MIN_PAST + HORIZON)
    val_inputs = _series_tensors(hist_val, MIN_PAST + HORIZON)
    print(f"微调: {len(train_inputs)} 训练序列, {len(val_inputs)} 验证序列")

    ft_pipes = {}
    for arm_name, loss_name, loss_kwargs in LOSS_CONFIGS:
        ckpt_dir = FT_DIR / arm_name / "ft"
        if ckpt_dir.exists():
            print(f"\n=== 加载已有 checkpoint {arm_name} ===")
            ft_pipes[arm_name] = BaseChronosPipeline.from_pretrained(
                ckpt_dir, device_map=args.device, torch_dtype=torch.float32)
        else:
            print(f"\n=== 微调 {arm_name}: loss={loss_name} ===")
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
        print(f"  就绪 ({time.time() - t0:.0f}s)")

    zs_pipe = BaseChronosPipeline.from_pretrained(
        BASE_CHECKPOINT, device_map=args.device, torch_dtype=torch.float32)

    all_arms = {"chronos2-zs": zs_pipe, **ft_pipes}

    # ================================================================
    # Phase 1: Prediction layer (NPL on test window)
    # ================================================================
    print("\n=== Phase 1: 预测层 NPL (test window) ===")
    weekly = osa.aggregate_to_weekly(daily)
    test_start = pd.Timestamp(cfg.splits["test"]["origins"][0])
    test_end = pd.Timestamp(cfg.splits["test"]["origins"][1])
    test_origins = pd.date_range(test_start, test_end, freq="7D")
    y_min = float(cfg.metric["y_min"])

    target = weekly[weekly.origin.isin(test_origins)][["series_id", "origin", "y"]].copy()
    print(f"测试窗: {len(test_origins)} origins, {len(target)} 完整周行")

    pred_rows = []
    for arm_name, p in all_arms.items():
        for origin in test_origins:
            cur = target[target.origin == origin]
            sids = cur.series_id.to_numpy()
            hist = daily[daily.d < origin]
            ctx = {s: g.sort_values("d").y.to_numpy(float)
                   for s, g in hist[hist.series_id.isin(sids)].groupby("series_id")}
            sids = np.array([s for s in sids if len(ctx.get(s, [])) >= MIN_CONTEXT])
            if not len(sids):
                continue
            cur = cur[cur.series_id.isin(sids)].set_index("series_id")

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
                "split": "test", "y": cur.loc[sids, "y"].to_numpy(),
                "q50": r[.5], "q85": r[.85], "w": 1.0}))

    # emp-daily
    for origin in test_origins:
        cur = target[target.origin == origin]
        sids = cur.series_id.to_numpy()
        hist = daily[daily.d < origin]
        ctx = {s: g.sort_values("d").y.to_numpy(float)
               for s, g in hist[hist.series_id.isin(sids)].groupby("series_id")}
        sids = np.array([s for s in sids if len(ctx.get(s, [])) >= MIN_CONTEXT])
        if not len(sids):
            continue
        cur = cur[cur.series_id.isin(sids)].set_index("series_id")
        emp_q50 = np.array([np.median(ctx[s]) * HORIZON for s in sids])
        emp_q85 = np.array([np.quantile(ctx[s], 0.85) * HORIZON for s in sids])
        pred_rows.append(pd.DataFrame({
            "variant": "emp-daily", "series_id": sids, "month": origin,
            "split": "test", "y": cur.loc[sids, "y"].to_numpy(),
            "q50": emp_q50, "q85": emp_q85, "w": 1.0}))

    pred = pd.concat(pred_rows, ignore_index=True)
    keys_all = {v: set(map(tuple, pred[pred.variant == v][["series_id", "month"]].to_numpy()))
                for v in pred.variant.unique()}
    common = set.intersection(*keys_all.values())
    pred = pred[[k in common for k in
                 map(tuple, pred[["series_id", "month"]].to_numpy())]]
    pred = pred.sort_values(["variant", "month", "series_id"]).reset_index(drop=True)

    print("\n%-24s %-10s %-8s" % ("model", "NPL", "n"))
    for v in sorted(pred.variant.unique()):
        s = pred[pred.variant == v]
        r = evaluate_slice(s.y, s.q50, s.q85, s.w, y_min)
        print("%-24s %-10.5f %-8d" % (v, r.npl, r.n))
        chk.note(f"npl_{v}", round(r.npl, 5))

    # ================================================================
    # Phase 2: Decision layer via Layer C replay, L=7 and L=14
    # ================================================================
    print("\n=== Phase 2: 决策层 Layer C replay (L=7, L=14) ===")

    replay_start = pd.Timestamp(cfg["replay"]["window"]["start"])
    replay_days = int(cfg["replay"]["window"]["days"])

    # Build demand matrix
    window_data = daily[(daily.d >= replay_start) &
                        (daily.d < replay_start + pd.Timedelta(days=replay_days))]
    obs_per_series = window_data.groupby("series_id").size()
    min_obs = int(replay_days * 0.8)
    valid_series = obs_per_series[obs_per_series >= min_obs].index

    demand_matrix = np.zeros((len(valid_series), replay_days))
    sid_list = sorted(valid_series)
    sid_to_idx = {s: i for i, s in enumerate(sid_list)}
    for _, row in window_data[window_data.series_id.isin(valid_series)].iterrows():
        day_idx = (row.d - replay_start).days
        if 0 <= day_idx < replay_days:
            demand_matrix[sid_to_idx[row.series_id], day_idx] = row.y
    sids = np.array(sid_list)
    n_ser = len(sids)

    # Initial inventory
    init_day = replay_start - pd.Timedelta(days=1)
    init_data = daily[daily.d == init_day].set_index("series_id")
    initial_inv = np.array([
        float(init_data.loc[s, "on_hand_inventory_units"])
        if s in init_data.index else 0.0
        for s in sids])

    # Context
    hist = daily[daily.d < replay_start]
    ctx = {}
    for s, g in hist[hist.series_id.isin(sids)].groupby("series_id"):
        ctx[s] = g.sort_values("d").y.to_numpy(float)
    sids_with_ctx = np.array([s for s in sids if len(ctx.get(s, [])) >= MIN_CONTEXT])
    keep_mask = np.isin(sids, sids_with_ctx)
    sids = sids_with_ctx
    demand_matrix = demand_matrix[keep_mask]
    initial_inv = initial_inv[keep_mask]
    n_ser = len(sids)
    print(f"回放: {n_ser} 序列, {replay_days} 天")

    review_days_list = list(range(0, replay_days, REVIEW_CADENCE))
    n_reviews = len(review_days_list)

    # Cost params (nominal)
    h_c = UNIT_COST * KAPPA_H / 12
    p_c = UNIT_COST * ALPHA_PRIMARY / (1 - ALPHA_PRIMARY) * KAPPA_H / 12

    all_replay_arms = {**all_arms, "emp-daily": None}
    dec_results = []

    for L in LEAD_TIMES:
        print(f"\n--- L={L} ---")
        for arm_name in all_replay_arms:
            max_pi = replay_days
            if arm_name == "emp-daily":
                grids = _emp_grids(sids, ctx, max_pi)
            else:
                grids = _daily_grids_from_pipe(
                    all_replay_arms[arm_name], sids, ctx, max_pi, args.batch_size)

            # Order-up-to at each review point
            S_matrix = np.zeros((n_ser, n_reviews))
            for ri, rd in enumerate(review_days_list):
                pi_len = REVIEW_CADENCE + L
                end = min(rd + pi_len, len(grids))
                if end <= rd:
                    S_matrix[:, ri] = 0
                    continue
                pi_grids = grids[rd:end]
                r_grids = grids[rd:min(rd + REVIEW_CADENCE, len(grids))]
                pmf_pi = convolve_varying_pmf(NATIVE_LEVELS, pi_grids, vmax=VMAX_DEC)
                pmf_r = convolve_varying_pmf(NATIVE_LEVELS, r_grids, vmax=VMAX_DEC)
                m_ratio = len(pi_grids) / max(len(r_grids), 1)
                S = order_up_to(pmf_r, pmf_pi, ALPHA_PRIMARY, m_ratio)["P3"]
                S_matrix[:, ri] = S

            rc = ReplayConfig(
                n_days=replay_days, lead_time_days=L,
                review_cadence_days=REVIEW_CADENCE,
                shortage_mechanism="lost_sales")
            rr = replay(demand_matrix, S_matrix, initial_inv, rc)

            h_arr = np.full(n_ser, h_c)
            p_arr = np.full(n_ser, p_c)
            m = replay_metrics(rr, h_arr, p_arr, REVIEW_CADENCE)

            dec_results.append(dict(
                L=L, arm=arm_name,
                CSR=m["CSR"], FR=m["FR"],
                cost=m["avg_cost_per_series_day"],
                hold=m["hold_cost"],
                short=m["short_cost"],
                n=n_ser))
            print(f"  {arm_name:<24} CSR={m['CSR']:.4f} FR={m['FR']:.4f} "
                  f"cost={m['avg_cost_per_series_day']:.4f} ({time.time() - t0:.0f}s)")

    dec_df = pd.DataFrame(dec_results)
    dec_df.to_csv(ART / "decision_results.csv", index=False)

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("WS-3 OSA Decision-aware FT 汇总")
    print("=" * 70)

    print("\n预测层 NPL:")
    for v in sorted(pred.variant.unique()):
        npl = chk.notes.get(f"npl_{v}", "?")
        print(f"  {v:<24} NPL={npl}")

    print("\n决策层 (Layer C replay):")
    for _, row in dec_df.iterrows():
        print(f"  L={row['L']:>2}  {row['arm']:<24} CSR={row['CSR']:.4f} "
              f"cost={row['cost']:.4f}")

    # Key comparison: decision-aware vs pinball at each L
    for L in LEAD_TIMES:
        sub = dec_df[dec_df.L == L].set_index("arm")
        if "chronos2-ft-pinball" in sub.index:
            pb_cost = sub.loc["chronos2-ft-pinball", "cost"]
            for arm in ["chronos2-ft-newsvendor", "chronos2-ft-focused"]:
                if arm in sub.index:
                    delta = sub.loc[arm, "cost"] - pb_cost
                    pct = delta / pb_cost * 100
                    csr_delta = sub.loc[arm, "CSR"] - sub.loc["chronos2-ft-pinball", "CSR"]
                    chk.note(f"cost_delta_{arm}_vs_pinball_L{L}",
                             round(delta, 6))
                    chk.note(f"csr_delta_{arm}_vs_pinball_L{L}",
                             round(csr_delta, 4))
                    print(f"\n  L={L} {arm} vs pinball: "
                          f"Δcost={delta:+.4f} ({pct:+.1f}%) "
                          f"ΔCSR={csr_delta:+.4f}")

    chk.n_rows = len(pred) + len(dec_df)
    chk.note("wall_clock_sec", round(time.time() - t0, 1))
    chk.finish(ART / "checks" / "ws3_osa.json")
    print(f"\n总耗时 {time.time() - t0:.0f}s, 状态: {chk.status}")
    return chk.exit_code


if __name__ == "__main__":
    sys.exit(main())
