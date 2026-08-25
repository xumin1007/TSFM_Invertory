"""DDM (Data-Driven Method) baseline — Shi, Chen & Duenyas (OR 2016).

Nonparametric gradient descent on censored demand data.
Directly optimizes order-up-to level S without estimating demand distribution.

Update rule (lost-sales, single-period):
  S_{t+1} = [S_t - η_t · (h·1[S_t > d_t] - p·1[S_t ≤ d_t])]_+

where d_t = min(D_t, S_t) is the censored observation.

Key insight: even under censoring, the gradient direction is correct because
1[S_t > d_t] is always true when demand is censored (D_t ≥ S_t → d_t = S_t).

Achieves O(1/√T) regret under iid demand. Our Zhao data violates iid
(drift ratio 0.878), which is precisely where TSFM's advantage lies.
"""

from __future__ import annotations

import numpy as np


def ddm_online(
    sales: np.ndarray,
    h: float,
    p: float,
    S_init: float | None = None,
    eta_schedule: str = "sqrt",
) -> np.ndarray:
    """Run DDM on a single series.

    Parameters
    ----------
    sales : (T,) censored sales d_t = min(D_t, inventory_t)
    h : holding cost per unit per period
    p : shortage cost per unit per period
    S_init : initial order-up-to level (default: mean of first 7 obs)
    eta_schedule : "sqrt" for η_t = 1/√t (theory-optimal), "const" for η=0.1

    Returns
    -------
    S_history : (T,) the order-up-to level used at each period
    """
    T = len(sales)
    S = np.zeros(T)

    if S_init is None:
        warmup = min(7, T)
        S_init = float(np.mean(sales[:warmup])) * 1.5 if warmup > 0 else 10.0
    S[0] = max(S_init, 0.0)

    for t in range(T - 1):
        d_t = sales[t]
        if eta_schedule == "sqrt":
            eta = 1.0 / np.sqrt(t + 1)
        else:
            eta = 0.1

        if S[t] > d_t:
            grad = h
        else:
            grad = -p

        S[t + 1] = max(S[t] - eta * grad, 0.0)

    return S


def ddm_batch(
    sales_matrix: np.ndarray,
    h: np.ndarray,
    p: np.ndarray,
    S_init: np.ndarray | None = None,
    eta_schedule: str = "sqrt",
) -> np.ndarray:
    """Run DDM on multiple series in parallel.

    Parameters
    ----------
    sales_matrix : (n_series, T)
    h : (n_series,) per-series holding cost
    p : (n_series,) per-series shortage cost
    S_init : (n_series,) initial S (default: 1.5 × mean of first 7 obs)

    Returns
    -------
    S_matrix : (n_series, T) order-up-to levels
    """
    n_ser, T = sales_matrix.shape
    S = np.zeros((n_ser, T))

    if S_init is None:
        warmup = min(7, T)
        S_init = np.mean(sales_matrix[:, :warmup], axis=1) * 1.5

    S[:, 0] = np.maximum(S_init, 0.0)

    for t in range(T - 1):
        d_t = sales_matrix[:, t]
        if eta_schedule == "sqrt":
            eta = 1.0 / np.sqrt(t + 1)
        else:
            eta = 0.1

        grad = np.where(S[:, t] > d_t, h, -p)
        S[:, t + 1] = np.maximum(S[:, t] - eta * grad, 0.0)

    return S


def ddm_cost(
    sales: np.ndarray,
    S: np.ndarray,
    h: float | np.ndarray,
    p: float | np.ndarray,
) -> dict:
    """Compute newsvendor cost given S and censored sales.

    For censored data, this is a lower bound on true cost because
    actual demand may exceed observed sales when stockout occurs.
    """
    excess = np.clip(S - sales, 0, None)
    shortage = np.clip(sales - S, 0, None)

    hold = h * excess
    short = p * shortage

    return {
        "hold": hold,
        "short": short,
        "cost": hold + short,
        "mean_cost": float((hold + short).mean()),
        "mean_S": float(S.mean()),
    }
