"""可行性缺口审计的回归守卫。对应 07 §6.2.0 step 1。

若日后有人改动分位重建或卷积而破坏了标定，本文件先失败 —— 那比在
决策层结果里发现覆盖异常要早得多，也更容易定位。
"""
import numpy as np
import pytest

from f2d.audit_coverage import ALPHAS_CHECK, pipeline_self_consistency


@pytest.fixture(scope="module")
def audit():
    return pipeline_self_consistency(n=2000)


def test_true_grid_delivers_nominal_alpha(audit):
    """喂入真分布时，实测覆盖必须落在 alpha ±0.02 内（泊松情形）。

    这是「缺口不是实现错误」这一结论的全部依据；一旦失效，07 §6.2.0
    与 §9.1.2 的论证同时作废。
    """
    row = audit[audit.case.str.startswith("iid 泊松")].iloc[0]
    for a in ALPHAS_CHECK:
        assert abs(row[a] - a) < 0.02, f"alpha={a} 实测 {row[a]:.4f}，管线标定已破坏"


def test_integer_support_never_undercovers_with_true_grid(audit):
    """整数支撑上 F^{-1}(a)=min{v:F(v)>=a} 只会超覆盖，不会欠覆盖。"""
    for _, row in audit[audit.exact_grid].iterrows():
        for a in ALPHAS_CHECK:
            assert row[a] >= a - 0.02, f"{row.case} 在 alpha={a} 欠覆盖"


def test_finite_context_causes_undercoverage_and_converges(audit):
    """有限 context 估计误差：必为欠覆盖，且随样本量收敛。

    这是 §6.2.0 分解表里「估计误差 ~0.025」一项的来源。
    """
    r200 = audit[audit.case.str.contains("200 样本")].iloc[0]
    r400 = audit[audit.case.str.contains("400 样本")].iloc[0]
    exact = audit[audit.exact_grid & audit.case.str.contains("负二项")].iloc[0]
    for a in ALPHAS_CHECK:
        assert r200[a] < exact[a], f"alpha={a}: 有限 context 应欠覆盖"
        assert r400[a] > r200[a], f"alpha={a}: 应随样本量收敛"
