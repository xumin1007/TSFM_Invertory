"""f2d.aggregation 的回归测试。

重点是 test_cdf_estimator_* 一组：早先的测试套件曾**全部通过**，却漏掉了
`cdf_estimator="lower"` 的系统性偏差 —— 因为那些测试用
`np.searchsorted(cdf, levels)` 构造输入，恰好使下端点等于真值。
真实数据上该 bug 在 m=30 时使结果偏差约 20 倍。

教训：验证 CDF 重建时，真值必须**落在网格点之间**，否则测不出端点选择的影响。
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from f2d.aggregation import (aggregate, aggregate_numeric, fit_gamma,
                             quantile_grid_to_pmf)

# Chronos-2 原生的 21 个分位点
L21 = np.array([.01, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5,
                .55, .6, .65, .7, .75, .8, .85, .9, .95, .99])


def _atom_grid(p0: float, n_pos: int = 8, levels: np.ndarray = L21):
    """构造 P(y=0)=p0、正部均匀的离散分布，返回其在 levels 上的分位数。"""
    pmf = np.concatenate([[p0], np.full(n_pos, (1 - p0) / n_pos)])
    cdf = np.cumsum(pmf)
    return np.searchsorted(cdf, levels).astype(float)[None, :]


# --------------------------------------------------------------------------
# 回归：CDF 估计量的选择
# --------------------------------------------------------------------------

@pytest.mark.parametrize("p0", [0.68, 0.72, 0.83, 0.47])
def test_cdf_estimator_midpoint_beats_lower_on_atom(p0):
    """真值落在网格点**之间**时，midpoint 必须显著优于 lower。

    p0 取值刻意避开 L21 的格点，这正是旧测试缺失的条件。
    """
    vals = _atom_grid(p0)
    lo = quantile_grid_to_pmf(L21, vals, vmax=20, cdf_estimator="lower")[0, 0]
    mid = quantile_grid_to_pmf(L21, vals, vmax=20, cdf_estimator="midpoint")[0, 0]
    assert abs(mid - p0) < abs(lo - p0), (
        f"p0={p0}: midpoint 误差 {abs(mid-p0):.4f} 未优于 lower {abs(lo-p0):.4f}")
    # 网格间距 0.05 => 中点误差不应超过半格
    assert abs(mid - p0) <= 0.026


def test_cdf_estimator_bias_compounds_under_convolution():
    """lower 的偏差会被卷积放大；m 越大差距越明显。"""
    p0 = 0.68
    vals = _atom_grid(p0)
    gaps = []
    for m in (1, 7, 30):
        a = aggregate_numeric(L21, vals, m=m, taus=(.85,), vmax=20,
                              cdf_estimator="lower")[.85][0]
        b = aggregate_numeric(L21, vals, m=m, taus=(.85,), vmax=20,
                              cdf_estimator="midpoint")[.85][0]
        gaps.append(abs(a - b))
    assert gaps[2] > gaps[0], f"m=30 的差距 {gaps[2]} 未大于 m=1 的 {gaps[0]}"


def test_default_estimator_is_midpoint():
    """默认必须是 midpoint —— lower 只保留用于复现对照。"""
    vals = _atom_grid(0.68)
    d = quantile_grid_to_pmf(L21, vals, vmax=20)
    m = quantile_grid_to_pmf(L21, vals, vmax=20, cdf_estimator="midpoint")
    assert np.allclose(d, m)


def test_unknown_estimator_rejected():
    with pytest.raises(ValueError, match="cdf_estimator"):
        quantile_grid_to_pmf(L21, _atom_grid(0.68), vmax=20, cdf_estimator="upper")


def test_quantile_linear_is_normalised_and_monotone():
    """分位函数线性插值路径必须给出合法 PMF。"""
    vals = np.array([[0., 0., 1., 2., 5.],
                     [0., 1., 1., 4., 8.]])
    levels = np.array([.10, .30, .50, .80, .99])
    pmf = quantile_grid_to_pmf(
        levels, vals, vmax=12, cdf_estimator="quantile_linear")
    assert np.all(pmf >= 0)
    assert np.allclose(pmf.sum(axis=1), 1.0)
    assert np.all(np.diff(np.cumsum(pmf, axis=1), axis=1) >= -1e-12)


def test_quantile_linear_inverts_linear_quantile_knots():
    """在无重复结点的简单例子中，CDF 应等于分位函数的线性反演。"""
    levels = np.array([.20, .60, .90])
    vals = np.array([[0., 4., 7.]])
    pmf = quantile_grid_to_pmf(
        levels, vals, vmax=10, cdf_estimator="quantile_linear")
    cdf = np.cumsum(pmf[0])
    # Between q(.20)=0 and q(.60)=4, F(2)=.40.
    assert np.isclose(cdf[2], .40)
    # All mass above q(.90)=7 is conservatively capped at q(.90).
    assert np.isclose(cdf[7], 1.0)
    assert np.isclose(pmf[8:].sum(), 0.0)


# --------------------------------------------------------------------------
# PMF 重建的基本性质
# --------------------------------------------------------------------------

def test_pmf_normalised_and_nonnegative():
    for p0 in (0.0, 0.3, 0.68, 0.95):
        pmf = quantile_grid_to_pmf(L21, _atom_grid(p0), vmax=20)
        assert np.isclose(pmf.sum(), 1.0)
        assert (pmf >= 0).all()


def test_levels_must_be_increasing():
    bad = np.array([.5, .3, .9])
    with pytest.raises(ValueError, match="升序"):
        quantile_grid_to_pmf(bad, np.array([[1., 2., 3.]]), vmax=10)


def test_values_length_must_match_levels():
    with pytest.raises(ValueError, match="列数"):
        quantile_grid_to_pmf(L21, np.array([[1., 2.]]), vmax=10)


def test_reconstruction_independent_of_vmax():
    """回归：残余质量不得倾倒到 vmax。

    曾经的实现写 `cdf[:, -1] = 1.0`，把 tau>0.99 的残余质量堆到 vmax，
    使重建均值随 vmax **线性**增长（vmax=300 时对真值 0.96 偏 +1.49）。
    该 bug 未被任何既有测试捕获，因为它们都不检查 vmax 敏感性。
    """
    vals = _atom_grid(0.68, n_pos=5)
    means = []
    for vmax in (10, 30, 100, 300, 1000):
        pmf = quantile_grid_to_pmf(L21, vals, vmax=vmax)[0]
        means.append(float(np.arange(vmax + 1) @ pmf))
    assert max(means) - min(means) < 1e-9, f"重建均值随 vmax 变化: {means}"


def test_no_mass_above_highest_quantile():
    """支撑不得超出该行的 q_{tau_K}。"""
    vals = _atom_grid(0.68, n_pos=5)
    qmax = int(vals[0, -1])
    pmf = quantile_grid_to_pmf(L21, vals, vmax=300)[0]
    assert pmf[qmax + 1:].sum() == 0.0, "存在超出最高分位的质量"


# --------------------------------------------------------------------------
# 数值卷积
# --------------------------------------------------------------------------

def test_numeric_convolution_matches_monte_carlo():
    """闭式卷积对照蒙特卡洛。用连续 Gamma 避开原子问题，只验卷积本身。"""
    a, b, m = 2.5, 4.0, 7
    lv = np.linspace(.005, .995, 199)
    vals = stats.gamma.ppf(lv, a, scale=b)[None, :]
    got = aggregate_numeric(lv, vals, m=m, taus=(.5, .85), vmax=300,
                            support="continuous")   # 连续 Gamma，非计数
    samp = stats.gamma.rvs(a, scale=b, size=(200_000, m), random_state=1).sum(1)
    for t in (.5, .85):
        mc = np.quantile(samp, t)
        assert abs(got[t][0] - mc) / mc < 0.05, f"tau={t}: {got[t][0]} vs MC {mc}"


def test_numeric_requires_integer_m():
    with pytest.raises(ValueError, match="正整数"):
        aggregate_numeric(L21, _atom_grid(0.5), m=2.5, vmax=20)


def test_zero_series_stays_zero():
    """全零分布聚合后仍为零。"""
    vals = np.zeros((1, L21.size))
    out = aggregate_numeric(L21, vals, m=30, taus=(.5, .85), vmax=20)
    assert out[.5][0] == 0 and out[.85][0] == 0


# --------------------------------------------------------------------------
# 两分位数 Gamma 路径
# --------------------------------------------------------------------------

def test_gamma_roundtrip():
    rng = np.random.default_rng(0)
    q50 = rng.gamma(2, 5, 2000) + 0.01
    q85 = q50 * rng.uniform(1.05, 8.0, 2000)
    sh, sc, md = fit_gamma(q50, q85)
    back = aggregate(sh, sc, m=1)
    assert np.percentile(np.abs(back[0.5] - q50) / q50, 99) < 1e-3
    assert np.percentile(np.abs(back[0.85] - q85) / q85, 99) < 1e-3
    assert (md == "exact").all()


def test_gamma_edge_modes():
    sh, sc, md = fit_gamma(np.array([0., 0., 5.]), np.array([0., 12., 5.]))
    assert list(md) == ["zero", "exp_anchored_q85", "clamped_hi"]
    assert aggregate(sh, sc, m=7)[0.85][0] == 0.0


def test_gamma_rejects_crossed_quantiles():
    with pytest.raises(ValueError, match="单调重排"):
        fit_gamma(np.array([5.0]), np.array([3.0]))


def test_phi_preserves_mean_scales_variance():
    a, b, m = 2.0, 3.0, 7
    for phi in (1.0, 1.5, 2.0):
        k, th = m * a / phi, b * phi
        assert np.isclose(k * th, m * a * b)
        assert np.isclose(k * th ** 2, m * a * b * b * phi)


def test_pmf_floor_semantics_integer_support():
    """整数支撑下 q(τ)=v 必须蕴含 F(floor(v)) >= τ。

    回归：早先实现写 `v <= g`，等价于上取整，使 q(0.5)=0.8 的隐含
    P(0) 只有 0.125（应 >= 0.5）。该缺陷让显式 ceil 成为恒等操作。
    """
    lv = np.array([.25, .5, .85])
    pmf = quantile_grid_to_pmf(lv, np.array([[0.3, 0.8, 2.4]]), vmax=6)[0]
    cdf = np.cumsum(pmf)
    assert cdf[0] >= 0.5 - 1e-9, f"F(0)={cdf[0]}，q(0.5)=0.8 时应 >= 0.5"
    assert cdf[2] >= 0.85 - 1e-9, f"F(2)={cdf[2]}，q(0.85)=2.4 时应 >= 0.85"


def test_pmf_integer_input_is_identity_across_support():
    """输入本身为整数时，integer 与 continuous 两种支撑必须给出同一 PMF。"""
    lv = np.array([.25, .5, .85])
    v = np.array([[0.0, 1.0, 3.0]])
    a = quantile_grid_to_pmf(lv, v, vmax=6, support="integer")
    b = quantile_grid_to_pmf(lv, v, vmax=6, support="continuous")
    assert np.allclose(a, b)


def test_support_mode_bias_direction():
    """离散化偏置方向：整数支撑（floor）的均值必须不高于连续支撑（round）。"""
    lv = np.linspace(.005, .995, 199)
    vals = stats.gamma.ppf(lv, 2.5, scale=4.0)[None, :]
    g = np.arange(301)
    mi = float(quantile_grid_to_pmf(lv, vals, 300, support="integer")[0] @ g)
    mc = float(quantile_grid_to_pmf(lv, vals, 300, support="continuous")[0] @ g)
    assert mi < mc
    assert abs(mc - mi - 0.5) < 0.1, f"两者应相差约 0.5，实测 {mc - mi:.3f}"
