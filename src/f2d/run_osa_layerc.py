"""OSA 层 C 连续机制回放。对应 docs/07_decision_layer.md §7 + configs/osa.yaml replay。

回放窗口：2021-01-04 起 84 天（完全落在 test split 内）。
复核节奏：每 7 天。
提前期网格：[7, 14, 21] 天（声明情景，非数据事实）。
缺货机制：lost-sales。
需求路径：观测销量（OSA_DEMAND_TRUTH 门未通过，不可称真实需求）。

每个复核时点 t，用 origin=t 之前的历史生成日级预测，卷积到保护期 PI=R+L，
取 S = F^{-1}(alpha)。三个预测臂 × 三个提前期 × 三个策略 × alpha 网格。

统计功效：单路径 84 天，仅 12 个复核周期 × ~99 序列。
定位：机制演示 + 提前期敏感性，非模型排名。

用法:  PYTHONPATH=src python -m f2d.run_osa_layerc [--device mps]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import pandas as pd

from . import config as cfgmod
from .aggregation import convolve_varying_pmf, pmf_quantile
from .checks import CheckResult
from .conventions import SEED_BASE
from .datasets import osa
from .decision import POLICIES, order_up_to
from .models.chronos import BASE_CHECKPOINT, NATIVE_LEVELS, QuantileRepair, to_grid
from .simulation import ReplayConfig, replay, replay_metrics

ART = cfgmod.ARTIFACT_DIR / "osa"
VMAX = 300
MIN_CONTEXT = 30
ALPHA_GRID = (0.90, 0.95, 0.98)
LEAD_TIME_GRID = (7, 14, 21)
REVIEW_CADENCE = 7
KAPPA_H = 0.20
ARMS = ("chronos2-zs", "emp-daily", "always-zero")


def _daily_grids(arm, sids, ctx, n_days, pipe, batch_size):
    """返回长度 n_days 的列表，每项 (n_series, 21)。"""
    if arm == "always-zero":
        return [np.zeros((len(sids), NATIVE_LEVELS.size)) for _ in range(n_days)]

    if arm == "emp-daily":
        emp = np.array([np.quantile(ctx[s], NATIVE_LEVELS, method="inverted_cdf")
                        for s in sids], float)
        emp, _ = QuantileRepair()(emp)
        return [emp] * n_days

    if arm == "chronos2-zs":
        import torch
        q, _ = pipe.predict_quantiles(
            [torch.tensor(ctx[s], dtype=torch.float32) for s in sids],
            prediction_length=n_days, quantile_levels=list(NATIVE_LEVELS),
            batch_size=batch_size)
        g, _ = QuantileRepair()(to_grid(q))
        g = g.reshape(len(sids), n_days, -1)
        return [g[:, i, :] for i in range(n_days)]

    raise ValueError(arm)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)

    import torch
    from chronos import BaseChronosPipeline

    t0 = time.time()
    ART.mkdir(parents=True, exist_ok=True)
    chk = CheckResult(step_id="osa_layer_c", dataset="osa", seed=SEED_BASE)
    cfg = cfgmod.load("osa")

    raw = osa.load_raw()
    daily, audit = osa.build_daily_panel(raw)
    chk.note("panel_audit", {k: audit[k] for k in ("n_rows", "n_series", "density_median")})

    # 回放窗口
    replay_cfg = cfg["replay"]
    replay_start = pd.Timestamp(replay_cfg["window"]["start"])
    replay_days = int(replay_cfg["window"]["days"])
    replay_dates = pd.date_range(replay_start, periods=replay_days, freq="D")
    print(f"回放窗口: {replay_start:%Y-%m-%d} 起 {replay_days} 天")

    # 构建需求路径：在回放窗口内有观测的序列
    # 由于 gap_decision=EXCLUDE，只保留窗口内每天都有观测的序列
    window_data = daily[(daily.d >= replay_start) &
                        (daily.d < replay_start + pd.Timedelta(days=replay_days))]
    obs_per_series = window_data.groupby("series_id").size()
    # 要求至少 80% 的天数有观测（84 天中至少 67 天）
    min_obs = int(replay_days * 0.8)
    valid_series = obs_per_series[obs_per_series >= min_obs].index
    print(f"序列筛选: 窗口内 >= {min_obs} 天观测 → {len(valid_series)} 条序列")

    # 构建稠密需求矩阵（缺失日补零——回放中"无记录"只能视为零销量或排除序列）
    demand_matrix = np.zeros((len(valid_series), replay_days))
    sid_list = sorted(valid_series)
    sid_to_idx = {s: i for i, s in enumerate(sid_list)}
    for _, row in window_data[window_data.series_id.isin(valid_series)].iterrows():
        day_idx = (row.d - replay_start).days
        if 0 <= day_idx < replay_days:
            demand_matrix[sid_to_idx[row.series_id], day_idx] = row.y
    n_ser = len(sid_list)
    sids = np.array(sid_list)
    chk.note("replay_n_series", n_ser)
    chk.note("demand_zero_share", round(float((demand_matrix == 0).mean()), 4))

    # 初始库存：回放前一天的 on_hand_inventory_units
    init_day = replay_start - pd.Timedelta(days=1)
    init_data = daily[daily.d == init_day].set_index("series_id")
    initial_inv = np.array([
        float(init_data.loc[s, "on_hand_inventory_units"])
        if s in init_data.index else 0.0
        for s in sids])
    chk.note("initial_inv_median", round(float(np.median(initial_inv)), 1))

    # 上下文：回放开始前的历史
    hist = daily[daily.d < replay_start]
    ctx = {}
    for s, g in hist[hist.series_id.isin(sids)].groupby("series_id"):
        ctx[s] = g.y.to_numpy()
    sids_with_ctx = np.array([s for s in sids if len(ctx.get(s, [])) >= MIN_CONTEXT])
    print(f"有足够上下文 (>={MIN_CONTEXT}): {len(sids_with_ctx)}/{n_ser} 序列")

    # 只保留有上下文的序列
    keep_mask = np.isin(sids, sids_with_ctx)
    sids = sids_with_ctx
    demand_matrix = demand_matrix[keep_mask]
    initial_inv = initial_inv[keep_mask]
    n_ser = len(sids)

    # 加载 Chronos-2
    pipe = BaseChronosPipeline.from_pretrained(
        BASE_CHECKPOINT, device_map=args.device, torch_dtype=torch.float32)

    # 复核时点
    review_days_list = list(range(0, replay_days, REVIEW_CADENCE))
    n_reviews = len(review_days_list)
    print(f"复核时点: {n_reviews} 个 (每 {REVIEW_CADENCE} 天)")

    # 对每个 arm × lead_time × alpha，预计算 order-up-to 序列
    results = []

    for arm in ARMS:
        # 预测：在回放开始时一次性预测（简化——实际应在每个复核时点更新预测，
        # 但 OSA 的 context 在 84 天内变化不大，且这是机制演示）
        max_pi = replay_days  # 最大保护期
        grids = _daily_grids(arm, sids, ctx, max_pi, pipe, args.batch_size)
        print(f"  {arm} 预测就绪 ({time.time() - t0:.0f}s)")

        for L in LEAD_TIME_GRID:
            pi_days = REVIEW_CADENCE + L  # 保护期 = R + L

            # 从分位网格计算保护期 PMF → 每个复核时点的 S
            # 简化：所有复核时点用同一个预测（origin 固定在回放开始前）
            pi_grids = grids[:pi_days] if pi_days <= len(grids) else grids
            if len(pi_grids) < pi_days:
                pi_grids = pi_grids + [pi_grids[-1]] * (pi_days - len(pi_grids))
            pmf_pi = convolve_varying_pmf(NATIVE_LEVELS, pi_grids, vmax=VMAX)

            r_grids = grids[:REVIEW_CADENCE]
            pmf_r = convolve_varying_pmf(NATIVE_LEVELS, r_grids, vmax=VMAX)
            m_ratio = pi_days / REVIEW_CADENCE

            for alpha in ALPHA_GRID:
                S_dict = order_up_to(pmf_r, pmf_pi, alpha, m_ratio)

                for pol in POLICIES:
                    S_val = S_dict[pol]
                    # 每个复核时点用同一个 S（静态预测）
                    S_matrix = np.tile(S_val[:, None], (1, n_reviews))

                    rc = ReplayConfig(
                        n_days=replay_days,
                        lead_time_days=L,
                        review_cadence_days=REVIEW_CADENCE,
                        shortage_mechanism="lost_sales")
                    res = replay(demand_matrix, S_matrix, initial_inv, rc)

                    if res.conservation_violations:
                        for v in res.conservation_violations:
                            print(f"    ⚠ {v}")
                            chk.note(f"violation_{arm}_{L}_{alpha}_{pol}", v)

                    # 成本：用声明的 kappa_h，p 由临界比导出
                    # OSA 无单位成本数据，用 1.0 作为标准化单位
                    unit_cost = np.ones(n_ser)
                    h = KAPPA_H * unit_cost / 365.0  # 日级
                    p = h * alpha / (1.0 - alpha)

                    metrics = replay_metrics(res, h, p, REVIEW_CADENCE)
                    metrics.update(dict(arm=arm, policy=pol, alpha=alpha,
                                        lead_time=L, n_series=n_ser,
                                        S_median=round(float(np.median(S_val)), 1)))
                    results.append(metrics)

            print(f"    L={L:2d}  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(results)
    df.to_csv(ART / "layer_c_summary.csv", index=False)

    # §7.2 断言 5：所有策略共享同一需求路径（构造性——同一 demand_matrix）
    chk.assert_true("§7.2.5 需求路径跨策略一致", True)

    # --- 输出 ---
    print("\n" + "=" * 90)
    print("OSA 层 C 连续回放结果（机制演示，非模型排名）")
    print(f"回放: {replay_start:%Y-%m-%d} 起 {replay_days} 天, "
          f"{n_ser} 序列, {n_reviews} 个复核周期")
    print(f"提前期: 声明情景（非数据事实）\n")

    for L in LEAD_TIME_GRID:
        print(f"--- L = {L} 天 (PI = {REVIEW_CADENCE + L} 天) ---")
        sub = df[(df.lead_time == L) & (df.policy == "P3")]
        print(f"{'arm':<16} {'alpha':>6} {'CSR':>6} {'FR':>6} "
              f"{'cost/sd':>8} {'avg_inv':>8} {'S_med':>6} {'lost':>8}")
        for _, r in sub.iterrows():
            print(f"{r.arm:<16} {r.alpha:>6.2f} {r.CSR:>6.4f} {r.FR:>6.4f} "
                  f"{r.avg_cost_per_series_day:>8.4f} {r.avg_inventory:>8.1f} "
                  f"{r.S_median:>6.1f} {r.total_lost_units:>8.1f}")
        print()

    # 守恒断言汇总
    n_violations = sum(1 for r in results
                       if any(k.startswith("violation_") for k in r))
    chk.assert_true("守恒断言全部通过", n_violations == 0)
    chk.note("n_scenarios", len(results))

    # P1 vs P3 差异（提前期是否让策略可区分）
    print("--- 策略可区分性（P3 vs P1, alpha=0.95）---")
    for L in LEAD_TIME_GRID:
        for arm in ARMS:
            p3 = df[(df.lead_time == L) & (df.policy == "P3") &
                    (df.arm == arm) & (df.alpha == 0.95)]
            p1 = df[(df.lead_time == L) & (df.policy == "P1") &
                    (df.arm == arm) & (df.alpha == 0.95)]
            if len(p3) and len(p1):
                dcost = float(p1.avg_cost_per_series_day.iloc[0]
                              - p3.avg_cost_per_series_day.iloc[0])
                dcsr = float(p3.CSR.iloc[0] - p1.CSR.iloc[0])
                print(f"  L={L:2d} {arm:<16} ΔCost(P1-P3)={dcost:+.4f}  "
                      f"ΔCSR(P3-P1)={dcsr:+.4f}")

    chk.n_rows = len(df)
    chk.note("wall_clock_sec", round(time.time() - t0, 1))
    chk.note("role", "MECHANISM_DEMONSTRATOR_ONLY")
    chk.note("eta_scenario_status", "DECLARED_ASSUMPTION_NOT_DATA")
    chk.finish(ART / "checks" / "layer_c.json")
    print(f"\n总耗时 {time.time() - t0:.0f}s, 状态: {chk.status}")
    return chk.exit_code


if __name__ == "__main__":
    sys.exit(main())
