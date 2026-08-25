"""层 B 与策略层。对应 docs/07_decision_layer.md §4、§6、§8。"""
import numpy as np
import pytest
from scipy import stats

from f2d.aggregation import convolve_varying_pmf, pmf_moments, pmf_quantile
from f2d.decision import costs_from_alpha, layer_b, order_up_to
from f2d.models.chronos import NATIVE_LEVELS as LV


def _grid(vals, n=1):
    return np.tile(np.asarray(vals, float), (n, 1))


def test_layer_b_conservation():
    """守恒：position - y = i_end - short，逐行成立。"""
    rng = np.random.default_rng(0)
    S, ip = rng.integers(0, 50, 200).astype(float), rng.integers(0, 40, 200).astype(float)
    y = rng.integers(0, 60, 200).astype(float)
    h, p = np.full(200, 1.0), np.full(200, 19.0)
    r = layer_b(S, ip, y, h, p)
    assert np.allclose(r.position - y,
                       r.i_end - r.observed_shortage_lower_bound)
    assert np.all(r.i_end * r.observed_shortage_lower_bound == 0)          # 不可同时超配与缺配
    assert np.all(r.order >= 0)


def test_layer_b_accepts_negative_ip():
    """ip 是库存位置，负值表示欠单，必须被接受且推高订货量。"""
    r = layer_b(np.array([10.0]), np.array([-3.0]), np.array([5.0]),
                np.array([1.0]), np.array([19.0]))
    assert r.order[0] == 13.0
    assert r.position[0] == 10.0
    r2 = layer_b(np.array([-1.0]), np.array([2.0]), np.array([0.0]),
                 np.array([1.0]), np.array([19.0]))
    assert r2.order[0] == 0.0


def test_layer_b_rejects_negative_demand():
    with pytest.raises(ValueError, match="y 必须非负"):
        layer_b(np.array([1.0]), np.array([1.0]), np.array([-1.0]),
                np.array([1.0]), np.array([1.0]))


def test_service_rates_differ_and_bracket():
    """CSR 与 FR 是不同的量（§8.1），且都落在 [0,1]。"""
    S = np.array([10.0, 0.0, 5.0])
    r = layer_b(S, np.zeros(3), np.array([5.0, 20.0, 5.0]),
                np.ones(3), np.full(3, 19.0))
    assert r.csr_upper_bound == pytest.approx(2 / 3)           # 3 期中 1 期缺货
    assert r.fill_rate_upper_bound == pytest.approx(1 - 20 / 30)
    assert r.csr_upper_bound != r.fill_rate_upper_bound


def test_costs_derived_from_alpha():
    """p/h 必须等于 alpha/(1-alpha)，不是独立参数（§5.3）。"""
    h, p = costs_from_alpha(np.array([12.0]), 0.95, 0.20, 12)
    assert h[0] == pytest.approx(0.20 * 12.0 / 12)
    assert (p / h)[0] == pytest.approx(19.0)
    with pytest.raises(ValueError):
        costs_from_alpha(np.array([1.0]), 1.0)


def test_p3_exceeds_p1_when_pi_longer():
    """保护期更长 => P3 的订至水平必须不低于忽略 PI 的 P1（§3.2）。"""
    g = _grid(np.arange(21) * 0.5)
    pmf_r = convolve_varying_pmf(LV, [g] * 7, vmax=60)
    pmf_pi = convolve_varying_pmf(LV, [g] * 21, vmax=60)
    S = order_up_to(pmf_r, pmf_pi, 0.95, m=3.0)
    assert S["P3"][0] > S["P1"][0]


def test_p2_matches_p3_under_normal_like_demand():
    """需求近正态时 P2 的正态近似应贴近 P3；这是 P2 的适用边界（§4.1）。"""
    lv = LV
    vals = stats.norm.ppf(lv, loc=30, scale=5)[None, :]
    pmf_r = convolve_varying_pmf(lv, [vals], vmax=120, support="continuous")
    pmf_pi = convolve_varying_pmf(lv, [vals] * 2, vmax=120, support="continuous")
    S = order_up_to(pmf_r, pmf_pi, 0.95, m=2.0)
    assert abs(S["P2"][0] - S["P3"][0]) / S["P3"][0] < 0.05


def test_moments_and_quantiles_come_from_same_pmf():
    """P2 用的矩与 P1/P3 用的分位数必须同源，否则策略间不可比。"""
    g = _grid(np.arange(21) * 0.3)
    pmf = convolve_varying_pmf(LV, [g] * 5, vmax=60)
    mu, sd = pmf_moments(pmf)
    med = pmf_quantile(pmf, [0.5])[0.5]
    assert abs(mu[0] - med[0]) < 4 * sd[0]         # 同一分布的中位与均值不应远离
    assert sd[0] > 0


def test_costs_from_margin_is_data_bound_not_alpha_bound():
    """毛利口径的 p 只依赖数据，与 alpha 无关；h 仍依赖声明的 kappa_h。"""
    from f2d.decision import costs_from_margin, implied_alpha
    c = np.array([12.0, 12.0])
    m = np.array([3.0, 30.0])
    h, p = costs_from_margin(c, m, 0.20, 12)
    assert np.allclose(p, m)                       # p 就是单位毛利
    assert np.allclose(h, 0.20 * c / 12)
    a = implied_alpha(h, p)
    assert a[1] > a[0]                             # 高毛利品服务目标更高
    assert np.all((a > 0) & (a < 1))


def test_margin_clipped_at_zero():
    """售价低于进价时 p 必须截断为 0，否则'故意缺货'成为最优。"""
    from f2d.decision import costs_from_margin, implied_alpha
    h, p = costs_from_margin(np.array([10.0]), np.array([-5.0]))
    assert p[0] == 0.0
    assert implied_alpha(h, p)[0] == 0.0           # 退化为永不订货


def test_pmf_quantile_rowwise_matches_scalar():
    """逐行分位入口与标量入口在同 tau 上必须一致。"""
    from f2d.aggregation import pmf_quantile, pmf_quantile_rowwise
    rng = np.random.default_rng(3)
    pmf = rng.random((5, 40))
    pmf /= pmf.sum(axis=1, keepdims=True)
    got = pmf_quantile_rowwise(pmf, np.full(5, 0.85))
    assert np.allclose(got, pmf_quantile(pmf, [0.85])[0.85])
    with pytest.raises(ValueError, match="长度"):
        pmf_quantile_rowwise(pmf, np.full(4, 0.5))
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        pmf_quantile_rowwise(pmf, np.full(5, 1.5))


def test_margin_block_is_origin_observable():
    """毛利块只能用 month 之前的记录 —— 越界即泄漏。"""
    import pandas as pd
    from f2d.datasets import zhao
    m = pd.Timestamp("2019-07-01")
    sales = pd.DataFrame({
        "date": pd.to_datetime(["2019-06-01", "2019-08-01"]),
        "sku_ID": ["a", "a"], "subcategory": ["s", "s"],
        "original_unit_price": [10.0, 999.0]})
    orders = pd.DataFrame({
        "order_date": pd.to_datetime(["2019-06-01", "2019-08-01"]),
        "sku_ID": ["a", "a"], "subcategory": ["s", "s"],
        "unit_cost": [6.0, 900.0]})
    b = zhao.build_margin_block({"sales": sales, "orders": orders}, [m])
    assert b.loc[0, "price_hist"] == 10.0          # 8 月的 999 不得进入
    assert b.loc[0, "cost_hist"] == 6.0
    assert b.loc[0, "margin_unit"] == 4.0
    assert b.loc[0, "price_level"] == "L0"


def test_layer_b_field_names_comply_with_truncation_rule():
    """§10：截断数据集的短缺列名必须自带下界限定，服务率必须自带上界限定。

    §11 验收第 7 条查这一项。命名不是风格问题 —— Zhao 的观测销量是需求
    下界，短缺量因此是下界、服务率是上界，简写为 `shortage`/`fr` 会让
    读者把上界当作实测值。
    """
    from f2d.decision import LayerBResult
    f = set(LayerBResult.__dataclass_fields__)
    assert "observed_shortage_lower_bound" in f
    assert {"csr_upper_bound", "fill_rate_upper_bound",
            "lost_units_lower_bound"} <= f
    for banned in ("short", "shortage", "fr", "csr", "lost_units"):
        assert banned not in f, f"{banned} 是被 §10 禁止的简写"


def test_service_rate_is_an_upper_bound_under_truncation():
    """构造性验证方向：真实需求 >= 观测销量 => 真实服务率 <= 观测服务率。"""
    S, ip = np.array([10.0, 10.0]), np.zeros(2)
    y_obs = np.array([10.0, 8.0])          # 被库存截断的观测销量
    y_true = np.array([14.0, 8.0])         # 真实需求（第一条被截断了）
    h, p = np.ones(2), np.full(2, 19.0)
    obs = layer_b(S, ip, y_obs, h, p)
    true = layer_b(S, ip, y_true, h, p)
    assert obs.csr_upper_bound > true.csr_upper_bound
    assert obs.fill_rate_upper_bound > true.fill_rate_upper_bound
    assert (obs.observed_shortage_lower_bound
            <= true.observed_shortage_lower_bound).all()
