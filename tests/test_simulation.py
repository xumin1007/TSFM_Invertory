"""闭环回放的状态与在途订单语义。"""

import numpy as np
import pytest

from f2d.simulation import ReplayConfig, replay


def test_orders_are_policy_specific_and_not_logged_orders():
    demand = np.array([[2.0, 2.0, 2.0]])
    cfg = ReplayConfig(n_days=3, lead_time_days=1, review_cadence_days=3)

    low = replay(demand, np.array([[3.0]]), np.array([0.0]), cfg)
    high = replay(demand, np.array([[8.0]]), np.array([0.0]), cfg)

    assert low.order[0, 0] == 3.0
    assert high.order[0, 0] == 8.0
    assert low.arrivals[0, 1] == 3.0
    assert high.arrivals[0, 1] == 8.0
    assert not np.array_equal(low.i_end, high.i_end)


def test_order_level_lead_times_override_the_fixed_lead_time():
    demand = np.zeros((1, 4))
    cfg = ReplayConfig(
        n_days=4, lead_time_days=1, review_days=(0, 3),
    )
    result = replay(
        demand,
        np.array([[5.0, 5.0]]),
        np.array([0.0]),
        cfg,
        order_lead_times=np.array([[2, 1]]),
    )

    assert result.arrivals[0, 1] == 0.0
    assert result.arrivals[0, 2] == 5.0
    assert result.conservation_violations == []


def test_preexisting_pipeline_is_shared_sunk_commitment():
    demand = np.zeros((1, 4))
    committed = np.array([[0.0, 4.0, 0.0, 0.0]])
    cfg = ReplayConfig(n_days=4, lead_time_days=1, review_cadence_days=4)

    result = replay(demand, np.array([[10.0]]), np.array([2.0]), cfg,
                    initial_pipeline_arrivals=committed)

    # t=0 的 IP 是 on-hand 2 + pre-existing pipeline 4，因此只新订 4。
    assert result.order[0, 0] == 4.0
    assert result.arrivals[0, 1] == 8.0  # sunk 4 + policy-specific 4
    assert result.pipeline[0, 0] == 8.0
    assert result.pipeline[0, 1] == 0.0
    assert result.conservation_violations == []


def test_explicit_review_days_support_calendar_month_boundaries():
    demand = np.zeros((1, 6))
    cfg = ReplayConfig(n_days=6, lead_time_days=1,
                       review_days=(0, 2, 5))
    result = replay(demand, np.array([[1.0, 2.0, 3.0]]),
                    np.array([0.0]), cfg)

    assert np.flatnonzero(result.order[0]).tolist() == [0, 2, 5]
    assert result.order[0, [0, 2, 5]].tolist() == [1.0, 1.0, 1.0]
    assert result.conservation_violations == []


def test_explicit_review_days_are_validated():
    cfg = ReplayConfig(n_days=3, lead_time_days=1, review_days=(1, 1))
    with pytest.raises(ValueError, match="review_days"):
        replay(np.zeros((1, 3)), np.zeros((1, 2)), np.zeros(1), cfg)
