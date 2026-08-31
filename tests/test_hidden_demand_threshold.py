import numpy as np

from f2d.run_hidden_demand_threshold import (_crossing,
                                             _minimum_overlap_units,
                                             _positive_overlap_segments,
                                             _two_way_bootstrap_ratios)


def test_minimum_overlap_and_crossing():
    units, used = _minimum_overlap_units(
        np.array([2.0, 1.0]), np.array([3.0, 10.0]), 8.0)
    assert units == 5.0 and used == 2
    assert _crossing(np.array([0.0, 1.0]), np.array([1.0, -1.0])) == 0.5

    slopes, caps = _positive_overlap_segments(
        np.array([[0.0]]),
        np.array([[[0.0, 1.0]]]), np.array([[[2.0, 3.0]]]),
        np.ones((1, 1, 2)), np.array([[[2.0, 1.0]]]))
    assert np.allclose(sorted(zip(slopes, caps)), [(1, 1), (2, 1), (3, 1)])


def test_two_way_bootstrap_is_reproducible_and_resamples_draws():
    ds = np.array([[[-2.0, -1.0], [-6.0, -3.0]]])
    bs = np.full_like(ds, 10.0)
    dd = ds / 2
    bd = np.full_like(ds, 8.0)
    first = _two_way_bootstrap_ratios(ds, bs, dd, bd, 100, 42)
    second = _two_way_bootstrap_ratios(ds, bs, dd, bd, 100, 42)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert first[0].shape == (100, 1)
    assert first[0].std() > 0
