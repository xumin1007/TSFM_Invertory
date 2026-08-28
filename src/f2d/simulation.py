"""层 C 连续机制回放引擎。对应 docs/07_decision_layer.md §7。

日循环顺序固定（§7.1）：
    I_begin[t]   = I_end[t-1]
    arrivals[t]  = 在 t 日到达的已下单量
    I_avail[t]   = I_begin[t] + arrivals[t]
    fulfilled[t] = min(demand_path[t], I_avail[t])
    lost[t]      = demand_path[t] - fulfilled[t]
    I_end[t]     = I_avail[t] - fulfilled[t]
    pipeline[t]  = 已下单未到货量
    if t 是复核时点:
        IP[t]    = I_end[t] + pipeline[t]
        order[t] = max(0, S_t - IP[t])

五条守恒断言（§7.2）逐日逐序列检查。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .conventions import FLOAT_TOL


@dataclass
class ReplayConfig:
    """回放参数。"""
    n_days: int
    lead_time_days: int
    review_cadence_days: int = 7
    shortage_mechanism: str = "lost_sales"
    # None 时沿用固定 cadence；连续跨月回放可显式传自然月月初等不等距复核日。
    review_days: tuple[int, ...] | None = None


@dataclass
class ReplayResult:
    """逐序列×逐日的回放状态表。"""
    # 形状全部 (n_series, n_days)
    i_begin: np.ndarray
    arrivals: np.ndarray
    i_avail: np.ndarray
    demand: np.ndarray
    fulfilled: np.ndarray
    lost: np.ndarray
    i_end: np.ndarray
    pipeline: np.ndarray
    order: np.ndarray
    # 汇总
    n_series: int = 0
    n_days: int = 0
    conservation_violations: list = field(default_factory=list)


def replay(demand_path: np.ndarray, order_up_to: np.ndarray,
           initial_inventory: np.ndarray, cfg: ReplayConfig,
           initial_pipeline_arrivals: np.ndarray | None = None) -> ReplayResult:
    """运行连续回放。

    Parameters
    ----------
    demand_path : (n_series, n_days) 需求路径（观测销量或真实需求）
    order_up_to : (n_series, n_reviews) 每个复核时点的订至水平 S_t
    initial_inventory : (n_series,) 第一天的期初库存
    cfg : ReplayConfig
    initial_pipeline_arrivals : (n_series, n_schedule_days), optional
        回放开始前已经承诺、并将在各相对日到货的数量。它们是所有候选政策
        共享的 sunk commitments；默认无在途订单。日程可以长于评价窗口，
        因为窗口外到货在窗口内仍属于库存位置的一部分。

    Returns
    -------
    ReplayResult 包含逐日逐序列的完整状态表
    """
    n_ser, n_days = demand_path.shape
    assert n_days == cfg.n_days
    L = cfg.lead_time_days
    R = cfg.review_cadence_days

    if cfg.review_days is None:
        review_days = list(range(0, n_days, R))
    else:
        review_days = list(cfg.review_days)
        if (review_days != sorted(set(review_days))
                or any(t < 0 or t >= n_days for t in review_days)):
            raise ValueError("review_days 必须是在评价窗口内严格递增且不重复的日序")
    n_reviews = len(review_days)
    if order_up_to.shape != (n_ser, n_reviews):
        raise ValueError(
            f"order_up_to shape {order_up_to.shape} != expected ({n_ser}, {n_reviews})")

    # 状态矩阵
    i_begin = np.zeros((n_ser, n_days))
    arrivals_mat = np.zeros((n_ser, n_days))
    i_avail = np.zeros((n_ser, n_days))
    fulfilled = np.zeros((n_ser, n_days))
    lost = np.zeros((n_ser, n_days))
    i_end = np.zeros((n_ser, n_days))
    pipeline = np.zeros((n_ser, n_days))
    order = np.zeros((n_ser, n_days))

    # 到货调度表：orders_arriving[t] = 在 day t 到货的量 (n_ser,)
    schedule_days = n_days + L + 1
    if initial_pipeline_arrivals is not None:
        initial_pipeline_arrivals = np.asarray(initial_pipeline_arrivals, float)
        if (initial_pipeline_arrivals.ndim != 2
                or initial_pipeline_arrivals.shape[0] != n_ser
                or initial_pipeline_arrivals.shape[1] < n_days):
            raise ValueError(
                "initial_pipeline_arrivals 必须为 "
                f"({n_ser}, n_schedule_days>=n_days)；收到 "
                f"{initial_pipeline_arrivals.shape}")
        if (np.any(~np.isfinite(initial_pipeline_arrivals))
                or np.any(initial_pipeline_arrivals < 0)):
            raise ValueError("initial_pipeline_arrivals 必须为有限非负数")
        schedule_days = max(schedule_days, initial_pipeline_arrivals.shape[1])

    orders_arriving = np.zeros((n_ser, schedule_days))
    if initial_pipeline_arrivals is not None:
        orders_arriving[:, :initial_pipeline_arrivals.shape[1]] = \
            initial_pipeline_arrivals

    # 回放前的在途库存等于全部尚未到货的 sunk commitments。
    initial_pipeline = orders_arriving.sum(axis=1)
    cur_pipeline = initial_pipeline.copy()
    review_idx = 0

    for t in range(n_days):
        # --- Step 1: I_begin ---
        if t == 0:
            i_begin[:, t] = initial_inventory
        else:
            i_begin[:, t] = i_end[:, t - 1]

        # --- Step 2: Receive arrivals ---
        arr = orders_arriving[:, t]
        arrivals_mat[:, t] = arr
        cur_pipeline = cur_pipeline - arr

        # --- Step 3: I_avail ---
        i_avail[:, t] = i_begin[:, t] + arr

        # --- Step 4: Fulfil demand ---
        d = demand_path[:, t]
        fulfilled[:, t] = np.minimum(d, i_avail[:, t])
        lost[:, t] = d - fulfilled[:, t]
        i_end[:, t] = i_avail[:, t] - fulfilled[:, t]

        # --- Step 5: Place order (if review day) ---
        if review_idx < n_reviews and review_days[review_idx] == t:
            ip = i_end[:, t] + cur_pipeline
            S = order_up_to[:, review_idx]
            ord_qty = np.clip(S - ip, 0.0, None)
            order[:, t] = ord_qty
            # 到货在 t + L
            arrival_day = t + L
            if arrival_day < orders_arriving.shape[1]:
                orders_arriving[:, arrival_day] += ord_qty
            cur_pipeline = cur_pipeline + ord_qty
            review_idx += 1

        pipeline[:, t] = cur_pipeline

    # --- §7.2 守恒断言 ---
    violations = []
    tol = FLOAT_TOL

    # 1. I_end == I_avail - fulfilled
    v1 = np.abs(i_end - (i_avail - fulfilled))
    if np.any(v1 > tol):
        violations.append(f"§7.2.1 I_end != I_avail - fulfilled: max err {v1.max():.2e}")

    # 2. fulfilled <= demand AND fulfilled <= I_avail
    if np.any(fulfilled > demand_path + tol):
        violations.append("§7.2.2 fulfilled > demand")
    if np.any(fulfilled > i_avail + tol):
        violations.append("§7.2.2 fulfilled > I_avail")

    # 3. I_end >= 0 (lost-sales)
    if cfg.shortage_mechanism == "lost_sales" and np.any(i_end < -tol):
        violations.append(f"§7.2.3 I_end < 0: min {i_end.min():.2e}")

    # 4. pipeline[t] == pipeline[t-1] + order[t] - arrivals[t]
    pipe0_expected = initial_pipeline + order[:, 0] - arrivals_mat[:, 0]
    v40 = np.abs(pipeline[:, 0] - pipe0_expected)
    if np.any(v40 > tol):
        violations.append(
            f"§7.2.4 pipeline conservation at t=0: max err {v40.max():.2e}")
    for t in range(1, n_days):
        pipe_expected = pipeline[:, t - 1] + order[:, t] - arrivals_mat[:, t]
        v4 = np.abs(pipeline[:, t] - pipe_expected)
        if np.any(v4 > tol):
            violations.append(
                f"§7.2.4 pipeline conservation at t={t}: max err {v4.max():.2e}")
            break

    return ReplayResult(
        i_begin=i_begin, arrivals=arrivals_mat, i_avail=i_avail,
        demand=demand_path, fulfilled=fulfilled, lost=lost,
        i_end=i_end, pipeline=pipeline, order=order,
        n_series=n_ser, n_days=n_days,
        conservation_violations=violations)


def replay_metrics(result: ReplayResult, h_cost: np.ndarray, p_cost: np.ndarray,
                   review_cadence: int = 7) -> dict:
    """从回放结果计算服务-成本指标（§8）。

    h_cost, p_cost: (n_series,) 每序列的持有/短缺单位成本。
    """
    n_ser, n_days = result.demand.shape

    # 持有成本：按期末库存
    hold_total = (h_cost[:, None] * result.i_end).sum()
    short_total = (p_cost[:, None] * result.lost).sum()
    total_cost = hold_total + short_total

    # CSR：每个复核周期是否有缺货
    review_days = list(range(0, n_days, review_cadence))
    n_cycles = len(review_days)
    n_stockout_cycles = 0
    total_cycles = 0
    for i, start in enumerate(review_days):
        end = review_days[i + 1] if i + 1 < n_cycles else n_days
        cycle_lost = result.lost[:, start:end].sum(axis=1)
        n_stockout_cycles += int((cycle_lost > 0).sum())
        total_cycles += n_ser

    csr = 1.0 - n_stockout_cycles / max(total_cycles, 1)

    # 填补率
    total_demand = result.demand.sum()
    total_lost = result.lost.sum()
    fr = 1.0 - total_lost / max(total_demand, 1e-12)

    return {
        "CSR": round(csr, 4),
        "FR": round(fr, 4),
        "total_cost": round(float(total_cost), 2),
        "hold_cost": round(float(hold_total), 2),
        "short_cost": round(float(short_total), 2),
        "avg_cost_per_series_day": round(float(total_cost / max(n_ser * n_days, 1)), 4),
        "avg_inventory": round(float(result.i_end.mean()), 2),
        "total_orders": int((result.order > 0).sum()),
        "total_lost_units": round(float(total_lost), 1),
        "n_cycles": total_cycles,
        "n_stockout_cycles": n_stockout_cycles,
    }
