"""从两个分位数还原分布，并聚合到更长的覆盖期。

对应 docs/07_decision_layer.md §2–§3。核心用途有二：
  1. 决策层：把预测视界 h 的分位数聚合到保护期 PI
  2. 方案 A：把日级分位数聚合到周/月目标（TSFM 与 GBDT 同担同一近似）

选 Gamma 的理由：非负、右偏、且同尺度 iid 之和封闭
（Gamma(a,b) 的 m 项和 = Gamma(ma, b)），使聚合有闭式解。

**iid 是近似。** 真实需求有自相关与周内季节性，正相关会使 iid 卷积**低估**
聚合方差。因此提供方差膨胀因子 phi，可由训练窗实测（estimate_phi），
默认 1.0 即纯 iid。phi 的取值必须记录并对所有模型一致。
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

from .conventions import TAU_MAIN, TAU_UPPER

# 形状参数的夹逼区间。两端截断而非外推：
#   a -> 大  => 分布趋于确定性（q85/q50 -> 1）
#   a -> 小  => 极端重尾
SHAPE_LO, SHAPE_HI = 1e-2, 1e5

# 指数分布锚定：q50=0 时退化用，b = q85 / (-ln(1-0.85))
_EXP_ANCHOR = -np.log(1.0 - TAU_UPPER)      # ≈ 1.8971


def _ratio(a: np.ndarray | float) -> np.ndarray | float:
    """q85/q50 作为形状的函数（尺度相消），严格单调递减。"""
    return stats.gamma.ppf(TAU_UPPER, a) / stats.gamma.ppf(TAU_MAIN, a)


def _build_lut(n: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """预计算 shape->ratio 查找表，供向量化反解。

    逐行 brentq 在 60 万行上不可接受；LUT + 插值精度已验证（见 tests）。
    """
    a = np.geomspace(SHAPE_LO, SHAPE_HI, n)
    r = _ratio(a)
    order = np.argsort(r)                    # ratio 递减，转为递增以便 interp
    return r[order], a[order]


_LUT_R, _LUT_A = _build_lut()


def fit_gamma(q50: np.ndarray, q85: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """由 (q50, q85) 反解 Gamma(shape, scale)。向量化。

    返回 (shape, scale, mode)，mode 取值见 07_decision_layer.md §2.3：
      "exact" | "clamped_hi" | "clamped_lo" | "zero" | "exp_anchored_q85"
    """
    q50 = np.asarray(q50, float)
    q85 = np.asarray(q85, float)
    if q50.shape != q85.shape:
        raise ValueError(f"形状不一致: {q50.shape} vs {q85.shape}")
    if np.any(q85 < q50 - 1e-12):
        raise ValueError("存在 q85 < q50 的行；应先经 postprocess 单调重排")

    shape = np.full(q50.shape, np.nan)
    scale = np.full(q50.shape, np.nan)
    mode = np.full(q50.shape, "exact", dtype=object)

    is_zero = q85 <= 0
    is_exp = (~is_zero) & (q50 <= 0)
    ok = ~is_zero & ~is_exp

    # 1) q85 == 0 -> 分布退化为 0
    shape[is_zero], scale[is_zero], mode[is_zero] = np.nan, 0.0, "zero"

    # 2) q50 == 0 < q85 -> 比值无定义，用指数分布锚定在有信息的 q85
    shape[is_exp] = 1.0
    scale[is_exp] = q85[is_exp] / _EXP_ANCHOR
    mode[is_exp] = "exp_anchored_q85"

    # 3) 正常情形：由比值查表反解形状，再由 q50 定标度
    if np.any(ok):
        r = q85[ok] / q50[ok]
        r_hi, r_lo = _LUT_R[0], _LUT_R[-1]          # 已按 ratio 升序
        a = np.interp(r, _LUT_R, _LUT_A)
        m = np.full(r.shape, "exact", dtype=object)
        m[r <= r_hi] = "clamped_hi"                 # 近乎确定性
        m[r >= r_lo] = "clamped_lo"                 # 极端重尾
        shape[ok] = a
        scale[ok] = q50[ok] / stats.gamma.ppf(TAU_MAIN, a)
        mode[ok] = m

    return shape, scale, mode


def estimate_phi(y_single: np.ndarray, y_sum: np.ndarray, m: int) -> float:
    """由训练窗实测方差膨胀因子 phi。

    纯 iid 时 Var(sum) = m * Var(single)，故
        phi = Var(y_sum) / (m * Var(y_single))
    phi > 1 表示正自相关（iid 卷积会低估聚合方差）。
    **只能用训练窗数据**，且对所有模型使用同一 phi。
    """
    v1 = float(np.nanvar(np.asarray(y_single, float)))
    vm = float(np.nanvar(np.asarray(y_sum, float)))
    if v1 <= 0:
        return 1.0
    return max(vm / (m * v1), 1e-6)


def aggregate(shape: np.ndarray, scale: np.ndarray, m: float,
              taus=(TAU_MAIN, TAU_UPPER), phi: float = 1.0) -> dict[float, np.ndarray]:
    """把单期 Gamma(shape, scale) 聚合到 m 期之和的分位数。

    iid: Gamma(m*shape, scale)。
    带方差膨胀 phi（保持均值不变）: Gamma(m*shape/phi, scale*phi)。
        均值 = m*shape*scale（与 phi 无关）
        方差 = m*shape*scale^2*phi
    """
    if m <= 0:
        raise ValueError("m 必须为正")
    if phi <= 0:
        raise ValueError("phi 必须为正")
    shape = np.asarray(shape, float)
    scale = np.asarray(scale, float)

    a_agg = m * shape / phi
    b_agg = scale * phi
    out = {}
    for t in taus:
        q = stats.gamma.ppf(t, a_agg, scale=b_agg)
        # shape 为 NaN 的行（mode=="zero"）聚合后仍为 0
        out[t] = np.where(np.isnan(shape), 0.0, q)
    return out


def aggregate_quantiles(q50: np.ndarray, q85: np.ndarray, m: float,
                        taus=(TAU_MAIN, TAU_UPPER), phi: float = 1.0):
    """便捷入口：(q50,q85) -> 聚合到 m 期的分位数 + 拟合模式。"""
    shape, scale, mode = fit_gamma(q50, q85)
    return aggregate(shape, scale, m, taus, phi), mode


# ---------------------------------------------------------------------------
# 数值卷积路径（零膨胀目标的**唯一**可用路径，见 07_decision_layer.md §2.3.1）
#
# 两分位数 Gamma 拟合在 Zhao 日级实测 q85 MAE=0.290；改用完整分位点网格
# 重建离散分布再 FFT 卷积，同一数据上 MAE 降至 0.081 —— 剩余误差才是 iid
# 假设的真实代价。故零占比高时必须走本路径。
# ---------------------------------------------------------------------------

def quantile_grid_to_pmf(levels: np.ndarray, values: np.ndarray, vmax: int,
                         cdf_estimator: str = "midpoint",
                         support: str = "integer") -> np.ndarray:
    """由分位点网格重建整数支撑上的 PMF。支持批量：values 形状 (n_rows, K)。

    分位点网格只把每个支撑点 v 的 CDF **括在一个区间内**：

        F(v) ∈ [ max{τ_j : q_j ≤ v},  min{τ_j : q_j > v} )

    `cdf_estimator="midpoint"`（默认）取区间中点，`"lower"` 取下端点。

    **必须用 midpoint。** 下端点恒定低估 F(v)，在零原子处最严重：真值
    P(0)=0.68 落在 [0.65,0.70) 时下端点少算 0.03，而卷积 30 次后
    0.65^30 与 0.68^30 相差 2.5 倍。Zhao 日级->周级实测（21 点网格，
    n=3000, vmax=300）：

        下端点   q50 MAE 0.333   q85 MAE 0.400
        中点     q50 MAE 0.000   q85 MAE 0.123
        经验上界 q50 MAE 0.000   q85 MAE 0.081

    `"lower"` 仅保留用于复现该对照，不得用于生产路径。
    """
    levels = np.asarray(levels, float)
    values = np.atleast_2d(np.asarray(values, float))
    if values.shape[1] != levels.size:
        raise ValueError(f"values 列数 {values.shape[1]} != levels 长度 {levels.size}")
    if np.any(np.diff(levels) <= 0):
        raise ValueError("levels 必须严格升序")
    if cdf_estimator not in ("midpoint", "lower"):
        raise ValueError(f"未知 cdf_estimator={cdf_estimator!r}")
    if support not in ("integer", "continuous"):
        raise ValueError(f"未知 support={support!r}")

    v = np.maximum.accumulate(values, axis=1)          # 保证每行非降

    # 分数分位值如何落到整数网格上，取决于**目标的支撑**：
    #
    #  support="integer"（默认，本项目全部目标）——真实取值是计数，分布在整数上
    #    是阶梯函数。q(τ)=v 意味着 F(v) >= τ，而 F(v) = F(floor(v))，故该 τ 应
    #    赋给 floor(v)。对整数输入是恒等操作。
    #
    #  support="continuous" —— 真实取值连续，整数格点只是离散化载体。此时 floor
    #    使每期均值下移 0.5（m 期累计 -m/2），应改用**就近取整**使离散化无偏。
    #
    # 早先两者都不是：写的是 `v <= g`，对 v=0.3 有 `0.3 <= 0` 为 False，等价于
    # **上取整**，每期上移 0.5。后果：Chronos-2 输出 q(0.5)=0.8 时隐含 P(0) 由
    # 应有的 >=0.5 变成 0.125。实测该错误使显式 ceil 的 ΔNPL 恰为 +0.00000
    # （因重建已隐式上取整），而显式 floor 反而改善 -0.040 —— 那不是「取整有益」，
    # 是在抵消本函数自身的偏差。
    snap = np.floor if support == "integer" else np.round
    g = np.arange(vmax + 1)[None, :]                   # (1, vmax+1)
    le = snap(v)[:, :, None] <= g[:, None, :]           # (n, K, vmax+1)
    lower = np.where(le.any(axis=1), (levels[None, :, None] * le).max(axis=1), 0.0)

    if cdf_estimator == "lower":
        cdf = lower
    else:
        gt = ~le
        upper = np.where(gt.any(axis=1),
                         np.where(gt, levels[None, :, None], np.inf).min(axis=1), 1.0)
        cdf = np.clip(0.5 * (lower + upper), 0.0, 1.0)

    cdf = np.maximum.accumulate(cdf, axis=1)

    # 尾部处理：网格最高只到 τ_K（Chronos-2 为 0.99），其上的残余质量
    # **必须放在该行自己的 q_{τ_K} 处**，不得倾倒到 vmax。
    #
    # 曾经的实现写 `cdf[:, -1] = 1.0`，把残余质量堆到 vmax，实测偏差随 vmax
    # 精确线性增长（vmax=10/30/100/300/1000 -> 均值偏 +0.04/+0.14/+0.49/
    # +1.49/+4.99，真值均值 0.96）。卷积后更严重：每期 0.005 的质量在 vmax，
    # m=30 时至少命中一次的概率达 1-(1-0.005)^30 = 14%，直接抬高上分位。
    #
    # 放在 q_{τ_K} 处是**最小假设**：网格对该点以上一无所知，截断于此不引入
    # 任何外生尺度。代价是轻微低估尾部，且该低估有界。
    qmax = np.clip(snap(v[:, -1]), 0, vmax).astype(np.intp)
    cdf = np.where(np.arange(vmax + 1)[None, :] >= qmax[:, None], 1.0, cdf)

    pmf = np.clip(np.diff(cdf, axis=1, prepend=0.0), 0.0, None)
    ssum = pmf.sum(axis=1, keepdims=True)
    return np.divide(pmf, ssum, out=np.zeros_like(pmf), where=ssum > 0)


def aggregate_numeric(levels: np.ndarray, values: np.ndarray, m: int,
                      taus=(TAU_MAIN, TAU_UPPER), vmax: int = 300,
                      cdf_estimator: str = "midpoint",
                      support: str = "integer") -> dict[float, np.ndarray]:
    """由分位点网格 -> 离散分布 -> m 次自卷积 -> 聚合分位数。批量 FFT。

    仍假设 m 期 iid（实测在 Zhao 日级->周级上 q85 MAE 仅 0.081）。
    m 必须为正整数：卷积次数无法取分数。
    """
    if int(m) != m or m < 1:
        raise ValueError(f"数值卷积要求 m 为正整数，收到 {m}")
    m = int(m)
    pmf = quantile_grid_to_pmf(levels, values, vmax, cdf_estimator, support)
    out_len = m * vmax + 1
    n_fft = 1 << int(np.ceil(np.log2(out_len)))
    conv = np.fft.irfft(np.fft.rfft(pmf, n_fft, axis=1) ** m, n_fft, axis=1)[:, :out_len]
    conv = np.clip(conv, 0.0, None)
    ssum = conv.sum(axis=1, keepdims=True)
    conv = np.divide(conv, ssum, out=np.zeros_like(conv), where=ssum > 0)
    cdf = np.cumsum(conv, axis=1)
    return {t: np.array([np.searchsorted(cdf[i], t) for i in range(cdf.shape[0])], float)
            for t in taus}


def linear_scale(q: np.ndarray, m: float) -> np.ndarray:
    """**错误做法**的显式实现：把分位数当可加量线性外推。

    仅用于在报告中量化该错误的代价（07_decision_layer.md §3.2 实测
    在 m=3、alpha=.95 时偏高 24%）。**禁止用于生产路径。**
    """
    return np.asarray(q, float) * m


def convolve_varying_pmf(levels: np.ndarray, values_per_step: list[np.ndarray],
                         vmax: int = 300, cdf_estimator: str = "midpoint",
                         support: str = "integer") -> np.ndarray:
    """同 convolve_varying，但返回**完整 PMF**，形状 (n_rows, m*vmax+1)。

    决策层需要的不止两个分位数：P2 要均值与标准差，P1/P3 要任意 alpha 分位，
    服务率诊断要 P(缺货)。逐项各卷积一次是浪费且可能不自洽（同一分布的
    q50 与 mean 必须来自同一个 PMF），故统一由本函数出 PMF。
    """
    if not values_per_step:
        raise ValueError("values_per_step 为空")
    m = len(values_per_step)
    n_rows = np.atleast_2d(values_per_step[0]).shape[0]
    for i, v in enumerate(values_per_step):
        if np.atleast_2d(v).shape[0] != n_rows:
            raise ValueError(f"第 {i} 步的行数与第 0 步不一致")

    out_len = m * vmax + 1
    n_fft = 1 << int(np.ceil(np.log2(out_len)))
    acc = None
    for v in values_per_step:
        f = np.fft.rfft(quantile_grid_to_pmf(levels, v, vmax, cdf_estimator, support),
                        n_fft, axis=1)
        acc = f if acc is None else acc * f

    conv = np.clip(np.fft.irfft(acc, n_fft, axis=1)[:, :out_len], 0.0, None)
    ssum = conv.sum(axis=1, keepdims=True)
    return np.divide(conv, ssum, out=np.zeros_like(conv), where=ssum > 0)


def pmf_quantile(pmf: np.ndarray, taus) -> dict[float, np.ndarray]:
    """PMF -> 分位数。约定 F^{-1}(t) = min{v : F(v) >= t}，与 §2.3 一致。"""
    cdf = np.cumsum(pmf, axis=1)
    return {t: np.array([np.searchsorted(cdf[i], t) for i in range(cdf.shape[0])], float)
            for t in taus}


def pmf_quantile_rowwise(pmf: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """逐行取不同的 tau。毛利口径下临界比随 SKU 异质，需要这一入口。

    与 pmf_quantile 的约定一致：F^{-1}(t) = min{v : F(v) >= t}。
    """
    taus = np.asarray(taus, float)
    if taus.shape[0] != pmf.shape[0]:
        raise ValueError(f"taus 长度 {taus.shape[0]} != 行数 {pmf.shape[0]}")
    if np.any((taus < 0) | (taus > 1)):
        raise ValueError("taus 必须在 [0,1]")
    cdf = np.cumsum(pmf, axis=1)
    return np.array([np.searchsorted(cdf[i], taus[i]) for i in range(cdf.shape[0])],
                    float)


def pmf_moments(pmf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PMF -> (均值, 标准差)。P2 的正态近似需要，且必须与分位数同源。"""
    g = np.arange(pmf.shape[1], dtype=float)
    mu = pmf @ g
    var = np.clip(pmf @ (g ** 2) - mu ** 2, 0.0, None)
    return mu, np.sqrt(var)


def convolve_varying(levels: np.ndarray, values_per_step: list[np.ndarray],
                     taus=(TAU_MAIN, TAU_UPPER), vmax: int = 300,
                     cdf_estimator: str = "midpoint",
                     support: str = "integer") -> dict[float, np.ndarray]:
    """卷积 m 个**互不相同**的单期分布。

    `aggregate_numeric` 做的是同分布自卷积（`pmf ** m`），只在各期预测相同时
    正确。TSFM 对 h 步视界给出 h 个**不同**的分布，此时必须逐步卷积：
        F_sum = irfft( prod_i rfft(pmf_i) )

    `values_per_step`: 长度 m 的列表，每项形状 (n_rows, K)，对应第 i 步的分位网格。
    仍假设各期**独立**（非同分布，但独立）；自相关的影响见 07 §2.3.2 的 iid 讨论。
    """
    if not values_per_step:
        raise ValueError("values_per_step 为空")
    m = len(values_per_step)
    n_rows = np.atleast_2d(values_per_step[0]).shape[0]
    for i, v in enumerate(values_per_step):
        if np.atleast_2d(v).shape[0] != n_rows:
            raise ValueError(f"第 {i} 步的行数与第 0 步不一致")

    out_len = m * vmax + 1
    n_fft = 1 << int(np.ceil(np.log2(out_len)))
    acc = None
    for v in values_per_step:
        f = np.fft.rfft(quantile_grid_to_pmf(levels, v, vmax, cdf_estimator, support),
                        n_fft, axis=1)
        acc = f if acc is None else acc * f

    conv = np.clip(np.fft.irfft(acc, n_fft, axis=1)[:, :out_len], 0.0, None)
    ssum = conv.sum(axis=1, keepdims=True)
    conv = np.divide(conv, ssum, out=np.zeros_like(conv), where=ssum > 0)
    cdf = np.cumsum(conv, axis=1)
    return {t: np.array([np.searchsorted(cdf[i], t) for i in range(cdf.shape[0])], float)
            for t in taus}
