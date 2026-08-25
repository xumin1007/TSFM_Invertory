"""预测层指标。对应 docs/00_global_conventions.md §4-§5。

关键约定：
- NPL 的地板 max(y_i, y_min) 在求和内部、逐样本生效（§4.1）；
- NPL 不可分解，每个切片用自己的分母（§4.4）；
- 零分母返回 NaN + reason_code，不返回 0、不丢行（§5.5）。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .conventions import EPS_DENOM, TAU_MAIN, TAU_UPPER, Y_MIN_BIND_CAP, Y_MIN_LADDER


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    """l_tau(y, q) = max(tau*(y-q), (tau-1)*(y-q))"""
    e = np.asarray(y, float) - np.asarray(q, float)
    return np.maximum(tau * e, (tau - 1.0) * e)


@dataclass
class MetricRow:
    """一个切片的指标。字段名即 metrics_*.csv 的列名。"""

    n: int
    n_positive: int
    sum_w: float
    npl: float
    cov_50: float
    cov_85: float
    cov_50_w: float
    cov_85_w: float
    cov_50_pos: float  # 仅 y>0 子集；零膨胀下唯一有信息的覆盖率
    cov_85_pos: float
    zero_share: float
    gap_50: float
    gap_85: float
    crossing_rate: float
    wape: float
    wpe: float
    y_min: float
    y_min_bind_rate: float
    reason_code: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _safe_div(num: float, den: float) -> tuple[float, str]:
    if not np.isfinite(den) or den < EPS_DENOM:
        return float("nan"), "ZERO_DENOM"
    return num / den, ""


def npl(
    y: np.ndarray,
    q50: np.ndarray,
    q_upper: np.ndarray,
    w: np.ndarray,
    y_min: float,
    gamma: float = TAU_UPPER,
) -> tuple[float, str]:
    """加权归一化双分位 pinball loss（§4.1）。

    分母为 max(2 * sum_i w_i * max(y_i, y_min), EPS_DENOM) —— 地板逐样本生效。
    """
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    num = float(np.sum(w * (pinball(y, q50, TAU_MAIN) + pinball(y, q_upper, gamma))))
    den = 2.0 * float(np.sum(w * np.maximum(y, y_min)))
    return _safe_div(num, den)


def coverage(y: np.ndarray, q: np.ndarray, w: np.ndarray | None = None) -> float:
    hit = (np.asarray(y, float) <= np.asarray(q, float)).astype(float)
    if w is None:
        return float(np.mean(hit)) if hit.size else float("nan")
    w = np.asarray(w, float)
    v, _ = _safe_div(float(np.sum(w * hit)), float(np.sum(w)))
    return v


def wape(y: np.ndarray, yhat: np.ndarray) -> tuple[float, str]:
    """分母取 sum|y|（§5.3）。y 可能为负（Mendeley 净销量）时与来源式不可比。"""
    y = np.asarray(y, float)
    return _safe_div(float(np.sum(np.abs(y - np.asarray(yhat, float)))), float(np.sum(np.abs(y))))


def wpe(y: np.ndarray, yhat: np.ndarray) -> tuple[float, str]:
    """正值 = 高估（与 FreshRetailNet 来源 eq.5 一致）。"""
    y = np.asarray(y, float)
    return _safe_div(float(np.sum(np.asarray(yhat, float) - y)), float(np.sum(y)))


def evaluate_slice(
    y: np.ndarray,
    q50: np.ndarray,
    q85: np.ndarray,
    w: np.ndarray,
    y_min: float,
    gamma: float = TAU_UPPER,
) -> MetricRow:
    """计算一个切片的全部指标。空切片返回 EMPTY_SLICE 而非省略。"""
    y = np.asarray(y, float)
    q50 = np.asarray(q50, float)
    q85 = np.asarray(q85, float)
    w = np.asarray(w, float)

    if y.size == 0:
        nan = float("nan")
        return MetricRow(0, 0, 0.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
                         nan, nan, nan, y_min, nan, "EMPTY_SLICE")

    v_npl, rc = npl(y, q50, q85, w, y_min, gamma)
    v_wape, rc_a = wape(y, q50)
    v_wpe, rc_b = wpe(y, q50)
    c50, c85 = coverage(y, q50), coverage(y, q85)
    pos = y > 0
    c50p = coverage(y[pos], q50[pos]) if pos.any() else float("nan")
    c85p = coverage(y[pos], q85[pos]) if pos.any() else float("nan")
    reason = rc or rc_a or rc_b

    return MetricRow(
        n=int(y.size),
        n_positive=int(np.sum(y > 0)),
        sum_w=float(np.sum(w)),
        npl=v_npl,
        cov_50=c50,
        cov_85=c85,
        cov_50_w=coverage(y, q50, w),
        cov_85_w=coverage(y, q85, w),
        cov_50_pos=c50p,
        cov_85_pos=c85p,
        zero_share=float(np.mean(y == 0)),
        gap_50=c50 - TAU_MAIN,
        gap_85=c85 - gamma,
        crossing_rate=float(np.mean(q50 > q85)),
        wape=v_wape,
        wpe=v_wpe,
        y_min=y_min,
        y_min_bind_rate=float(np.mean(y < y_min)),
        reason_code=reason,
    )


def choose_y_min(y_train: np.ndarray, integer_units: bool,
                 cap: float = Y_MIN_BIND_CAP) -> tuple[float, float]:
    """§4.3：按目标量纲二选一，仅用训练窗，确定性，不可调优。

    integer_units=True  -> 路径 A：y_min = 1.0（最小业务单位），不适用 cap。
                           零膨胀目标上 bind_rate 高是数据性质，正是地板存在的目的。
    integer_units=False -> 路径 B：阶梯规则，取 bind_rate <= cap 的最大值。

    返回 (y_min, bind_rate)。
    """
    y = np.asarray(y_train, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        raise ValueError("choose_y_min: 训练窗为空")

    if integer_units:
        return 1.0, float(np.mean(y < 1.0))

    for v in Y_MIN_LADDER:
        rate = float(np.mean(y < v))
        if rate <= cap:
            return v, rate
    raise ValueError(
        f"choose_y_min 路径 B 无解：零占比 {float(np.mean(y <= 0)):.4f} > cap {cap}。"
        f"该目标疑为整数计数，应改用 integer_units=True（§4.3）")
