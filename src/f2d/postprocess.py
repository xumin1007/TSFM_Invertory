"""预测输出后处理。对应 docs/00_global_conventions.md §2.3。

顺序固定，不得跳过或改序：
  1. 非有限值检出（返回掩码，交由调用方触发 §7.3 回退）
  2. 非负截断
  3. 单调重排（排序，非单侧压平）
"""

from __future__ import annotations

import numpy as np


def postprocess(q50: np.ndarray, q85: np.ndarray,
                round_integer: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """返回 (q50, q85, bad_mask, crossing_rate)。

    bad_mask 标记原始输出含非有限值的行；这些行必须由调用方走回退，
    不得在此静默填 0。crossing_rate 在重排**之前**统计，它是模型质量诊断。

    `round_integer` 对应 §2.3 第 3 步，**默认关闭**：
      - 送入 f2d.aggregation 做分布重建/卷积时 -> 必须 True（否则零原子丢失）
      - 直接进入 NPL 评价时 -> 保持 False（实测改善 -0.00005，远小于 MDE；
        且各取整方向的收益依赖模型自身偏倚，会引入不可比因素）
    取整状态须记入 run_manifest.json 的 `rounding_applied`，并对所有模型一致。
    """
    q50 = np.asarray(q50, float).copy()
    q85 = np.asarray(q85, float).copy()
    if q50.shape != q85.shape:
        raise ValueError(f"形状不一致: {q50.shape} vs {q85.shape}")

    # 1. 非有限值
    bad = ~(np.isfinite(q50) & np.isfinite(q85))

    # 交叉率在重排前统计（仅对有限值行）
    ok = ~bad
    crossing = float(np.mean(q50[ok] > q85[ok])) if ok.any() else float("nan")

    # 2. 非负截断（对非有限值行先置 0 以免污染排序；这些行由 bad_mask 标记）
    q50[bad] = 0.0
    q85[bad] = 0.0
    np.clip(q50, 0.0, None, out=q50)
    np.clip(q85, 0.0, None, out=q85)

    # 3. 整数取整（仅重建路径；见 docstring 与 00_global_conventions §2.3）
    if round_integer:
        np.rint(q50, out=q50)
        np.rint(q85, out=q85)

    # 4. 单调重排（排序，对称，不偏袒任一 tau 项）
    lo = np.minimum(q50, q85)
    hi = np.maximum(q50, q85)
    return lo, hi, bad, crossing
