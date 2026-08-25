"""策略与层 B 单期递推。对应 docs/07_decision_layer.md §4、§6、§8。

输入是**保护期需求的 PMF**（由 f2d.aggregation.convolve_varying_pmf 给出），
不是两个分位数 —— Zhao 日级零占比 0.68，两分位数 Gamma 拟合在此已实测失效
（07 §2.3.1）。P2 需要的 (mu, sigma) 也从同一个 PMF 出，保证与分位数同源。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .aggregation import pmf_moments, pmf_quantile

POLICIES = ("P1", "P2", "P3")


def order_up_to(pmf_R: np.ndarray, pmf_PI: np.ndarray, alpha: float,
                m: float) -> dict[str, np.ndarray]:
    """三个订至水平 S。P4/P5 需要 ETA 分位与固定成本，不在层 B 范围。

    P1  忽略保护期，只用复核期 R 的分位数 —— 故意的失配对照臂（07 §3.2）
    P2  正态近似 m*mu + z_alpha*sqrt(m)*sigma，其中 (mu,sigma) 是 R 期的矩
    P3  保护期需求的真实分位数（数值卷积，非闭式）
    """
    mu, sd = pmf_moments(pmf_R)
    z = float(stats.norm.ppf(alpha))
    return {
        "P1": pmf_quantile(pmf_R, [alpha])[alpha],
        "P2": np.clip(m * mu + z * np.sqrt(m) * sd, 0.0, None),
        "P3": pmf_quantile(pmf_PI, [alpha])[alpha],
    }


@dataclass
class LayerBResult:
    """§8 的必报量。每行是一个 (sku, month) 补货周期。

    **字段命名受 §10 约束。** 截断数据集（Zhao）上观测销量是需求的下界，
    故短缺量是**下界**、填补率是**上界**，命名必须自带这一限定，不得
    简写为 `shortage` / `fr`：

      observed_shortage_lower_bound   逐行短缺量（下界）
      csr_upper_bound                 周期服务率（上界）
      fill_rate_upper_bound           填补率（上界）
      lost_units_lower_bound          损失件数（下界）

    方向：真实需求 >= 观测销量 => 真实短缺 >= 观测短缺 => 真实服务率
    <= 观测服务率。故本类给出的服务率是**乐观**的，不可行判定因此更强
    ——测得 CSR < alpha 时，真实 CSR 只会更低。
    """
    order: np.ndarray
    position: np.ndarray
    i_end: np.ndarray
    observed_shortage_lower_bound: np.ndarray
    cost: np.ndarray
    csr_upper_bound: float
    fill_rate_upper_bound: float
    n_orders: int
    avg_inventory: float
    lost_units_lower_bound: float


def layer_b(S: np.ndarray, ip: np.ndarray, y: np.ndarray,
            h_cost: np.ndarray, p_cost: np.ndarray) -> LayerBResult:
    """§6.1 的每期递推。每月从观测快照重置，**不跨月连接**。

    position = max(IP, S) 依据 §3.1 实测（Zhao 提前期中位 1 天，月初下单当月
    可用）。该假设的越界行由调用方按 lead_time_exceeds_period 单独切片。

    短缺量是**观测下界** —— 销量已被库存截断，真实需求不可观测（§10）。
    返回字段名自带该限定，见 LayerBResult 的说明。
    """
    for name, a in (("S", S), ("ip", ip), ("y", y)):
        if np.any(~np.isfinite(a)):
            raise ValueError(f"{name} 含非有限值")
    if np.any(y < 0):
        raise ValueError("y 必须非负")
    # ip 按定义是**库存位置**（现货 + 在途 - 欠单），负值合法且表示欠单。
    # Zhao 验证窗实测 5/3902 行为负。不截断为 0 —— 那会掩盖欠单状态并
    # 低估应订量；order = max(0, S - ip) 本就正确处理负 ip。

    order = np.clip(S - ip, 0.0, None)
    position = np.maximum(ip, S)
    short = np.clip(y - position, 0.0, None)
    i_end = np.clip(position - y, 0.0, None)
    cost = h_cost * i_end + p_cost * short

    return LayerBResult(
        order=order, position=position, i_end=i_end,
        observed_shortage_lower_bound=short, cost=cost,
        csr_upper_bound=float(np.mean(short <= 0)),
        fill_rate_upper_bound=float(1.0 - short.sum() / max(y.sum(), 1e-12)),
        n_orders=int((order > 0).sum()),
        avg_inventory=float(np.mean(0.5 * (ip + i_end))),
        lost_units_lower_bound=float(short.sum()),
    )


def costs_from_alpha(unit_cost: np.ndarray, alpha: float, kappa_h: float = 0.20,
                     periods_per_year: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """§5.2 / §5.3：h 由 kappa_h 与单位成本定，p 由 newsvendor 临界比导出。

    p 是**导出量**（derived_from_alpha），不得表述为业务事实短缺成本。
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha 必须在 (0,1)")
    h = kappa_h * np.asarray(unit_cost, float) / periods_per_year
    return h, h * alpha / (1.0 - alpha)


def costs_from_margin(unit_cost: np.ndarray, margin_unit: np.ndarray,
                      kappa_h: float = 0.20,
                      periods_per_year: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """短缺成本 = **损失的单位毛利**（售价 - 进价），由数据算出。

    与 costs_from_alpha 的分工：后者把 p 绑在声明的服务目标上，是**导出量**；
    本函数把 p 绑在数据上，是**部分由数据支持的量**（h 仍依赖声明的 kappa_h）。
    两者必须并列报告，不得互相替代。

    实测（Zhao 验证窗）二者高度一致：毛利口径隐含的临界比
    alpha = p/(p+h) = 0.934，而事前独立声明的 alpha = 0.95。这是对该声明值
    目前最强的一条数据支持，但两条路径互相独立，故仍分开报告。

    毛利口径的另一处优势：p_i 随 SKU 异质（高毛利品缺货更贵），而
    p_i = 19 h_i 使 p/h 对所有 SKU 恒等，掩盖了这一事实。
    """
    h = kappa_h * np.asarray(unit_cost, float) / periods_per_year
    return h, np.clip(np.asarray(margin_unit, float), 0.0, None)


def implied_alpha(h_cost: np.ndarray, p_cost: np.ndarray) -> np.ndarray:
    """由成本对反解 newsvendor 临界比。逐 SKU，故在毛利口径下是异质的。"""
    tot = np.asarray(h_cost, float) + np.asarray(p_cost, float)
    return np.divide(p_cost, tot, out=np.zeros_like(tot), where=tot > 0)
