"""Chronos-2 适配器。对应 docs/06_model_hyperparams.md §4。

本模块把三条已实测的限制处理为显式、可审计的配置项，而不是留给调用方踩：

  B4  `use_target_encoding` 默认 True，对**静态**类别会塌成序列均值
      （实测：类别 A/B 的两条序列分别编码为其自身目标均值 10.0 / 50.0，
      类别身份完全丢失）。本模块**硬编码为 False**，不提供开关。

  B2  对非负序列输出负分位数（实测最低 5 档 −0.194 至 −0.007）。

  B3  低分位段是零附近噪声，**不表示零原子**。对 P(0)=0.68 的序列，
      模型在 τ=0.5 处给 0.275，而真值 q50 恰为 0。仅做非负截断只能
      推出 P(0)≈0.20。故提供 zero_atom 处理策略，见 QuantileRepair。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Chronos-2 训练过的 21 个分位点。请求其他 τ 只会得到对这 21 个值的插值
# （实测最大差 6e-6），故不提供自定义网格。
NATIVE_LEVELS = np.array([
    .01, .05, .10, .15, .20, .25, .30, .35, .40, .45, .50,
    .55, .60, .65, .70, .75, .80, .85, .90, .95, .99])

BASE_CHECKPOINT = "amazon/chronos-2"          # 119.5M，含原生 0.85
SMALL_CHECKPOINT = "autogluon/chronos-2-small"  # 27.9M，**不含** 0.85，须插值


def prepare_inputs(df: pd.DataFrame, target_columns: list[str], prediction_length: int,
                   future_df: pd.DataFrame | None = None, **kw):
    """构造 Chronos-2 输入。**强制 use_target_encoding=False**（B4）。

    该参数不作为入参暴露：开启后静态类别会退化为序列均值，而本项目的
    类别特征几乎全为静态（实测：五个数据集中仅 Mendeley 的 Sales Type
    与 Stock Status 时变，占比 16.7% / 12.2%）。详见 06 §4.3。
    """
    from chronos.chronos2.preprocess import from_data_frame
    return from_data_frame(df, target_columns=target_columns,
                           prediction_length=prediction_length,
                           future_df=future_df, use_target_encoding=False, **kw)


def to_grid(q, n_levels: int = len(NATIVE_LEVELS),
            step: int | None = None, prediction_length: int | None = None) -> np.ndarray:
    """把 `predict_quantiles` 的返回整形为 (n_rows, n_levels)。

    实测其返回 `list[Tensor]`，每个元素形状
    `(n_series_in_item, prediction_length, n_quantiles)`；直接 `np.asarray`
    得到 4 维数组。此处统一拍平前两维。

    `step=None`（默认）保留全部预测步，行数 = 序列数 × prediction_length，
    行序为「序列优先、步长次之」。给定 `step` 时只取该步，此时**必须**同时
    给出 `prediction_length` —— 否则无法从拍平后的行数还原序列数。
    """
    if isinstance(q, (list, tuple)):
        parts = [np.asarray(x, float) for x in q]
        a = np.concatenate([p.reshape(-1, p.shape[-1]) for p in parts], axis=0)
    else:
        a = np.asarray(q, float)
        a = a.reshape(-1, a.shape[-1])
    if a.shape[-1] != n_levels:
        raise ValueError(f"分位点数 {a.shape[-1]} != 预期 {n_levels}")

    if step is None:
        return a
    if prediction_length is None:
        raise ValueError("指定 step 时必须同时给出 prediction_length")
    if a.shape[0] % prediction_length:
        raise ValueError(
            f"行数 {a.shape[0]} 不是 prediction_length={prediction_length} 的整数倍")
    if not 0 <= step < prediction_length:
        raise ValueError(f"step={step} 超出 [0, {prediction_length})")
    return a.reshape(-1, prediction_length, n_levels)[:, step, :]


@dataclass
class QuantileRepair:
    """对 Chronos-2 原始分位数输出的修复策略（B2 / B3）。

    三步，可独立开关，均记录到 `repair_flags` 供审计：

    1. `clip_negative`  —— 非负截断。处理 B2，但**不足以**处理 B3。
    2. `round_integer`  —— 目标为整数计数时四舍五入。零附近的小正值
       （0.007…0.275）由此归零，零原子被部分恢复。
    3. `splice_zero_atom` —— 用 **context 的经验零率** 覆写零原子：
       强制 q_τ = 0 对所有 τ ≤ p0_hat。零率可由 context 直接观测
       （200 个观测下抽样标准差约 0.033，与 21 点网格的分辨极限 ±0.025
       同量级），且不受模型分布外风险影响。

    `splice_zero_atom` 是**建模决定而非纯后处理**：它把零发生率交给经验
    估计、把正部形状交给模型。启用时必须对所有被比较的模型一致启用。
    """

    clip_negative: bool = True
    round_integer: bool = True
    splice_zero_atom: bool = False

    def __call__(self, q: np.ndarray, levels: np.ndarray = NATIVE_LEVELS,
                 context: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        """q 形状 (n_rows, n_levels)；context 形状 (n_rows, ctx_len)，仅 splice 时需要。

        原始 pipeline 输出请先经 `to_grid()` 整形。
        """
        q = np.atleast_2d(np.asarray(q, float)).copy()
        if q.shape[-1] != levels.size:
            raise ValueError(
                f"q 的列数 {q.shape[-1]} != levels 长度 {levels.size}；"
                "是否忘了先调用 to_grid()？")
        flags: dict[str, float] = {}

        if self.clip_negative:
            flags["neg_share"] = float((q < 0).mean())
            np.clip(q, 0.0, None, out=q)

        if self.round_integer:
            q = np.rint(q)

        if self.splice_zero_atom:
            if context is None:
                raise ValueError("splice_zero_atom=True 需要提供 context 以估计零率")
            p0 = (np.asarray(context, float) == 0).mean(axis=1, keepdims=True)
            flags["p0_hat_median"] = float(np.median(p0))
            # τ <= p0_hat 的分位数强制为 0
            q = np.where(levels[None, :] <= p0, 0.0, q)

        # 单调重排（截断与取整都可能破坏单调性）
        q = np.maximum.accumulate(q, axis=1)
        i50 = int(np.searchsorted(levels, 0.5))
        flags["q50_zero_share"] = float((q[:, i50] == 0).mean())
        return q, flags


def implied_p0(q: np.ndarray, levels: np.ndarray = NATIVE_LEVELS) -> np.ndarray:
    """由分位数网格反推模型隐含的 P(y=0)：取值为 0 的最大 τ。

    用于诊断 B3：与 context 的经验零率对照，可量化模型对零原子的低估。
    """
    q = np.atleast_2d(np.asarray(q, float))
    is0 = q <= 0
    return np.where(is0.any(axis=1), (levels[None, :] * is0).max(axis=1), 0.0)
