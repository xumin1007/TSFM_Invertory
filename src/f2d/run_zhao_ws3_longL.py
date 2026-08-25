"""WS-3 补充: 在 Zhao 上用半合成 L=7/14 评价 decision-aware FT 的决策层。

微调模型来自 run_zhao_finetune_decision.py (已训练好)，本脚本只做评价。
Layer B 评价，P3 策略，α=0.95。配对 bootstrap 检验成本差。

用法:  PYTHONPATH=src python -m f2d.run_zhao_ws3_longL [--device mps]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd
import torch

from . import config as cfgmod
from .aggregation import convolve_varying_pmf
from .checks import CheckResult
from .conventions import SEED_BASE
from .datasets import zhao
from .decision import costs_from_alpha, layer_b, order_up_to
from .models.chronos import BASE_CHECKPOINT, NATIVE_LEVELS, QuantileRepair, to_grid

ART = cfgmod.ARTIFACT_DIR / "zhao_ws3_longL"
FT_DIR = cfgmod.ARTIFACT_DIR / "zhao_finetune"
VMAX_DEC = 60
ALPHA_PRIMARY = 0.95
KAPPA_H = 0.20
LEAD_TIMES = [1, 7, 14]
MIN_CONTEXT = 30

VALID_MONTHS = [pd.Timestamp("2019-07-01"), pd.Timestamp("2019-08-01")]

ARMS = [
    ("chronos2-zs", None),
    ("chronos2-ft-pinball", FT_DIR / "chronos2-ft-pinball" / "ft"),
    ("chronos2-ft-newsvendor", FT_DIR / "chronos2-ft-newsvendor" / "ft"),
    ("chronos2-ft-focused", FT_DIR / "chronos2-ft-focused" / "ft"),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-series", type=int, default=2000)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args(argv)

    from chronos import BaseChronosPipeline

    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    chk = CheckResult(step_id="ws3_longL", dataset="zhao", seed=SEED_BASE)

    raw = zhao.load_raw()
    daily, _ = zhao.build_daily_panel(raw)
    panel, _ = zhao.build_panel(raw)

    rng = np.random.default_rng(SEED_BASE)
    all_sids = sorted(daily.series_id.unique())
    keep = rng.choice(all_sids, size=min(args.n_series, len(all_sids)), replace=False)
    daily = daily[daily.series_id.isin(keep)]
    sku_of = dict(zip(daily.series_id, daily.sku_ID))

    # Load all pipes
    pipes = {}
    for arm_name, ckpt in ARMS:
        if ckpt is None:
            pipes[arm_name] = BaseChronosPipeline.from_pretrained(
                BASE_CHECKPOINT, device_map=args.device, torch_dtype=torch.float32)
        else:
            pipes[arm_name] = BaseChronosPipeline.from_pretrained(
                ckpt, device_map=args.device, torch_dtype=torch.float32)
        print(f"  加载 {arm_name} ({time.time() - t0:.0f}s)")

    # emp-daily baseline (no pipe needed)
    dec_rows = []
    boot_rows = []

    for month in VALID_MONTHS:
        snap = panel[panel.month == month].set_index("sku_ID")
        hist = daily[daily.d < month]
        ctx = {s: g.sort_values("d").y.to_numpy(float)
               for s, g in hist.groupby("series_id")}
        sids = np.array(sorted([s for s in ctx
                                if len(ctx[s]) >= MIN_CONTEXT
                                and sku_of.get(s) in snap.index]))
        sk = np.array([sku_of[s] for s in sids])
        cur = snap.loc[sk]
        ip = cur["beginning_inventory"].to_numpy(float) + cur["on_order_inventory"].to_numpy(float)
        y = cur["observed_sales_next_month"].to_numpy(float)
        cost_i = cur["unit_cost_hist"].to_numpy(float)
        has_c = np.isfinite(cost_i)

        for L in LEAD_TIMES:
            n_days = month.days_in_month + L

            for arm_name in list(pipes) + ["emp-daily"]:
                if arm_name == "emp-daily":
                    emp = np.array([np.quantile(ctx[s], NATIVE_LEVELS, method="inverted_cdf")
                                    for s in sids], float)
                    emp, _ = QuantileRepair()(emp)
                    grids = [emp] * n_days
                else:
                    q, _ = pipes[arm_name].predict_quantiles(
                        [torch.tensor(ctx[s], dtype=torch.float32) for s in sids],
                        prediction_length=n_days,
                        quantile_levels=list(NATIVE_LEVELS),
                        batch_size=args.batch_size)
                    g, _ = QuantileRepair()(to_grid(q))
                    g = g.reshape(len(sids), n_days, -1)
                    grids = [g[:, i, :] for i in range(n_days)]

                pmf_pi = convolve_varying_pmf(NATIVE_LEVELS, grids, vmax=VMAX_DEC)
                pmf_r = convolve_varying_pmf(NATIVE_LEVELS,
                                              grids[:month.days_in_month], vmax=VMAX_DEC)
                m_ratio = n_days / month.days_in_month
                S = order_up_to(pmf_r, pmf_pi, ALPHA_PRIMARY, m_ratio)["P3"]
                r = layer_b(S, ip, y, *costs_from_alpha(cost_i, ALPHA_PRIMARY, KAPPA_H, 12))

                cost_mean = float(r.cost[has_c].mean()) if has_c.any() else float("nan")
                dec_rows.append(dict(
                    month=month, L=L, arm=arm_name,
                    n=len(sids), n_costed=int(has_c.sum()),
                    CSR_ub=r.csr_upper_bound,
                    FR_ub=r.fill_rate_upper_bound,
                    cost=cost_mean,
                    S_med=float(np.median(S))))

                # Per-series cost for bootstrap
                h_c, p_c = costs_from_alpha(cost_i, ALPHA_PRIMARY, KAPPA_H, 12)
                pos = np.maximum(ip, S)
                row_cost = h_c * np.clip(pos - y, 0, None) + p_c * np.clip(y - pos, 0, None)
                boot_rows.append(pd.DataFrame({
                    "arm": arm_name, "series_id": sids, "month": month,
                    "L": L, "row_cost": row_cost, "has_cost": has_c}))

                print(f"  {month:%Y-%m} L={L:>2} {arm_name:<24} CSR={r.csr_upper_bound:.4f} "
                      f"cost={cost_mean:.2f} ({time.time() - t0:.0f}s)")

    dec_df = pd.DataFrame(dec_rows)
    dec_df.to_csv(ART / "decision_longL.csv", index=False)

    # Summary by L
    print("\n" + "=" * 70)
    for L in LEAD_TIMES:
        sub = dec_df[dec_df.L == L].groupby("arm")[["CSR_ub", "FR_ub", "cost", "S_med"]].mean()
        print(f"\n--- L={L} 两月平均 ---")
        print(sub.round(4).to_string())

    # Paired bootstrap cost comparison at each L
    from .uncertainty import paired_bootstrap_mean, report
    boot_df = pd.concat(boot_rows, ignore_index=True)
    boot_df = boot_df[boot_df.has_cost]

    print("\n" + "=" * 70)
    print("配对 bootstrap 成本差 (基准=chronos2-zs)")
    for L in LEAD_TIMES:
        sub = boot_df[boot_df.L == L].copy()
        variants = [a for a in sub.arm.unique() if a != "chronos2-zs"]
        cis = paired_bootstrap_mean(sub, "row_cost", "chronos2-zs", sorted(variants),
                                    variant_col="arm")
        print(f"\n  L={L}:")
        print("  " + report(cis)[["variant", "delta", "lo", "hi", "verdict"]]
              .round(4).to_string(index=False).replace("\n", "\n  "))
        for c in cis:
            chk.note(f"L{L}_{c.variant}_delta",
                     [round(c.delta, 4), round(c.lo, 4), round(c.hi, 4), c.significant])

    # Also compare decision-aware vs pinball specifically
    print("\n" + "=" * 70)
    print("配对 bootstrap 成本差 (基准=chronos2-ft-pinball)")
    for L in LEAD_TIMES:
        sub = boot_df[boot_df.L == L].copy()
        da_arms = [a for a in ["chronos2-ft-newsvendor", "chronos2-ft-focused"]
                   if a in sub.arm.unique()]
        if not da_arms:
            continue
        cis = paired_bootstrap_mean(sub, "row_cost", "chronos2-ft-pinball", da_arms,
                                    variant_col="arm")
        print(f"\n  L={L}:")
        print("  " + report(cis)[["variant", "delta", "lo", "hi", "verdict"]]
              .round(4).to_string(index=False).replace("\n", "\n  "))
        for c in cis:
            chk.note(f"L{L}_{c.variant}_vs_pinball",
                     [round(c.delta, 4), round(c.lo, 4), round(c.hi, 4), c.significant])

    chk.n_rows = len(dec_df)
    chk.note("wall_clock_sec", round(time.time() - t0, 1))
    chk.finish(ART / "checks" / "ws3_longL.json")
    print(f"\n总耗时 {time.time() - t0:.0f}s, 状态: {chk.status}")
    return chk.exit_code


if __name__ == "__main__":
    sys.exit(main())
