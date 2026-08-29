"""Synthetic regression tests for reviewer-facing robustness utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f2d.run_review_robustness import (
    _construct_parametric_demand,
    _copula_uniforms,
)
from f2d.run_ws4_kappa_margin import summarize_margin_cost
from f2d.run_mechanism_analysis import summarize_support_binding


def _toy_inputs():
    start = pd.Timestamp("2019-03-01")
    total_days = sum(pd.Timestamp(f"2019-{m:02d}-01").days_in_month
                     for m in range(3, 11))
    y = np.zeros((2, total_days))
    # March is classified as censored and contains the only target day.
    y[:, 0] = [2, 3]
    # April is an uncensored donor month with a non-degenerate distribution.
    y[0, 31:61] = np.tile([0, 1, 2], 10)
    y[1, 31:61] = np.tile([0, 2, 5], 10)
    censor_months = {(i, oi): oi == 0 for i in range(2) for oi in range(8)}
    return y, np.array(["s1", "s2"]), start, total_days, censor_months


def test_parametric_dgp_preserves_observed_lower_bound_and_untargeted_days():
    y, sids, start, n_days, flags = _toy_inputs()
    d, _ = _construct_parametric_demand(
        y, sids, start, n_days, flags, 4, "negative_binomial", 42)
    assert np.all(d >= y[None, :, :])
    assert np.array_equal(d[:, :, 1:], np.tile(y[:, 1:], (4, 1, 1)))


def test_parametric_dgp_is_invariant_to_other_series():
    y, sids, start, n_days, flags = _toy_inputs()
    full, _ = _construct_parametric_demand(
        y, sids, start, n_days, flags, 4, "poisson", 42)
    sub_flags = {(0, oi): flags[(1, oi)] for oi in range(8)}
    sub, _ = _construct_parametric_demand(
        y[1:], sids[1:], start, n_days, sub_flags, 4, "poisson", 42)
    assert np.array_equal(full[:, 1], sub[:, 0])


def test_copula_draws_are_reproducible_and_dependence_changes():
    u1 = _copula_uniforms(31, .25, 256)
    u2 = _copula_uniforms(31, .25, 256)
    u3 = _copula_uniforms(31, .50, 256)
    assert np.array_equal(u1, u2)
    assert not np.array_equal(u1, u3)
    assert np.all((u1 > 0) & (u1 < 1))


def test_margin_cost_summary_uses_sku_specific_economic_outputs():
    aggregate = pd.DataFrame({
        "kappa_h": [.1, .1, .2, .2],
        "arm": ["chronos2-zs", "emp-daily"] * 2,
        "cost_margin": [9.0, 10.0, 19.0, 20.0],
    })
    alpha = pd.DataFrame({
        "kappa_h": [.1, .2],
        "alpha_implied_p10": [.90, .80],
        "alpha_implied_p50": [.95, .90],
        "alpha_implied_p90": [.99, .97],
    })
    got = summarize_margin_cost(aggregate, alpha)
    np.testing.assert_allclose(
        got["chronos2_zs_cost_reduction_pct"], [10.0, 5.0])
    assert list(got["alpha_implied_p50"]) == [.95, .90]


def test_support_binding_summary_uses_month_specific_convolution_edge():
    per_sku = pd.DataFrame({
        "month": ["2020-01-01", "2020-02-01"],
        "S_tsfm": [320.0, 299.0],
        "S_emp": [319.0, 300.0],
    })
    got = summarize_support_binding(per_sku, vmax=10, lead_days=1)
    by_policy = got.set_index("policy")
    assert by_policy.loc["chronos2-zs", "n_binding"] == 1
    assert by_policy.loc["emp-daily", "n_binding"] == 1
    assert by_policy.loc["chronos2-zs", "support_min"] == 300
    assert by_policy.loc["chronos2-zs", "support_max"] == 320
    assert np.isclose(
        by_policy.loc["chronos2-zs", "max_support_utilization_pct"],
        100.0,
    )
