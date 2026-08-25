"""f2d.models.chronos 的测试。

含一条回归：`to_grid(step=...)` 曾在 prediction_length>1 时**静默返回错误形状**
（4 条序列 × 7 步被压成 28 行而非取第 step 步的 4 行），不抛错。
形状类 bug 若不抛错会一路污染下游，故此处对每个分支断言精确形状。
"""
from __future__ import annotations

import numpy as np
import pytest

from f2d.models.chronos import (NATIVE_LEVELS, QuantileRepair, implied_p0,
                                to_grid)

K = len(NATIVE_LEVELS)


def _fake_output(n_series: int, horizon: int):
    """模拟 predict_quantiles 的返回：list[(1, H, K)]，每条序列一个元素。"""
    rng = np.random.default_rng(0)
    return [np.sort(rng.random((1, horizon, K)), axis=-1) for _ in range(n_series)]


# --------------------------------------------------------------------------
# to_grid 形状
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n,h", [(4, 1), (4, 7), (1, 30), (13, 3)])
def test_to_grid_flattens_all_steps(n, h):
    assert to_grid(_fake_output(n, h)).shape == (n * h, K)


@pytest.mark.parametrize("h", [1, 7, 30])
def test_to_grid_step_selection(h):
    """回归：H>1 时选取单步必须得到 (n_series, K)，而非拍平的 (n*H, K)。"""
    n = 4
    for step in range(h):
        got = to_grid(_fake_output(n, h), step=step, prediction_length=h)
        assert got.shape == (n, K), f"h={h} step={step}: {got.shape}"


def test_to_grid_step_requires_prediction_length():
    with pytest.raises(ValueError, match="prediction_length"):
        to_grid(_fake_output(4, 7), step=0)


def test_to_grid_step_out_of_range():
    with pytest.raises(ValueError, match="超出"):
        to_grid(_fake_output(4, 7), step=7, prediction_length=7)


def test_to_grid_rejects_wrong_level_count():
    bad = [np.zeros((1, 1, K - 3))]
    with pytest.raises(ValueError, match="分位点数"):
        to_grid(bad)


def test_to_grid_selects_correct_step_values():
    """不仅形状对，取的必须是该步的值。"""
    n, h = 3, 5
    q = [np.tile(np.arange(h, dtype=float)[None, :, None], (1, 1, K)) + s
         for s in range(n)]
    for step in range(h):
        got = to_grid(q, step=step, prediction_length=h)
        assert np.allclose(got[:, 0], np.arange(n) + step)


# --------------------------------------------------------------------------
# QuantileRepair
# --------------------------------------------------------------------------

def test_repair_rejects_wrong_width():
    with pytest.raises(ValueError, match="to_grid"):
        QuantileRepair()(np.zeros((2, 5)))


def test_clip_negative_alone_does_not_recover_atom():
    """已实测结论：非负截断对零原子无效 —— 负值只在最低几档。"""
    q = np.array([[-0.13, -0.04, -0.02, -0.003, .005, .012, .022, .027, .033,
                   .043, .047, .06, .094, .16, .338, .938, 2.169, 4.232,
                   7.008, 10.639, 18.5]])
    raw_p0 = implied_p0(q)[0]
    clipped, _ = QuantileRepair(round_integer=False)(q)
    assert np.isclose(implied_p0(clipped)[0], raw_p0), "截断不应改变隐含 P(0)"


def test_rounding_recovers_atom():
    """取整才是主要修复：把零附近小正值归零。"""
    q = np.array([[-0.13, -0.04, -0.02, -0.003, .005, .012, .022, .027, .033,
                   .043, .047, .06, .094, .16, .338, .938, 2.169, 4.232,
                   7.008, 10.639, 18.5]])
    only_clip, _ = QuantileRepair(round_integer=False)(q)
    with_round, _ = QuantileRepair()(q)
    assert implied_p0(with_round)[0] > implied_p0(only_clip)[0] + 0.3


def test_splice_requires_context():
    with pytest.raises(ValueError, match="context"):
        QuantileRepair(splice_zero_atom=True)(np.zeros((1, K)))


def test_splice_forces_zero_below_empirical_rate():
    ctx = np.zeros((1, 100))
    ctx[0, :30] = 5.0                       # 经验零率 0.70
    q = np.ones((1, K))
    out, flags = QuantileRepair(splice_zero_atom=True)(q, context=ctx)
    assert np.isclose(flags["p0_hat_median"], 0.70)
    assert (out[0, NATIVE_LEVELS <= 0.70] == 0).all()
    assert (out[0, NATIVE_LEVELS > 0.70] > 0).all()


def test_output_is_monotone():
    rng = np.random.default_rng(1)
    q = rng.normal(0, 3, (50, K))           # 刻意非单调
    for rep in (QuantileRepair(), QuantileRepair(round_integer=False)):
        out, _ = rep(q.copy())
        assert (np.diff(out, axis=1) >= 0).all()


def test_repair_does_not_mutate_input():
    q = np.full((2, K), -1.0)
    before = q.copy()
    QuantileRepair()(q)
    assert np.array_equal(q, before)


# --------------------------------------------------------------------------
# implied_p0
# --------------------------------------------------------------------------

def test_implied_p0_reads_largest_zero_level():
    q = np.zeros((1, K))
    q[0, NATIVE_LEVELS > 0.65] = 1.0
    assert np.isclose(implied_p0(q)[0], 0.65)


def test_implied_p0_all_positive_is_zero():
    assert implied_p0(np.ones((1, K)))[0] == 0.0
