"""Synthetic regression tests for reviewer-facing robustness utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f2d.run_review_robustness import (
    _construct_parametric_demand,
    _copula_uniforms,
)


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
