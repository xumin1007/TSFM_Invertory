"""DeepAR 和 TFT 概率预测模型适配器。

使用 NeuralForecast 库，输出与 Chronos-2 相同的 21 点分位网格，
使下游的 convolve_varying 管线完全复用。

信息集与 chronos2-zs / emp-daily / gbdt-lean 对齐：仅用历史销量，
不加结构化特征。这确保对比的是模型能力而非信息集差异。
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from .chronos import NATIVE_LEVELS

ModelName = Literal["deepar", "tft"]


def _build_nf_df(contexts: dict[str, np.ndarray]) -> pd.DataFrame:
    """把 {series_id: 1d-array} 转为 NeuralForecast 格式。"""
    rows = []
    for sid, arr in contexts.items():
        n = len(arr)
        rows.append(pd.DataFrame({
            "unique_id": str(sid),
            "ds": pd.date_range(end="2099-01-01", periods=n, freq="D"),
            "y": arr.astype(float),
        }))
    return pd.concat(rows, ignore_index=True)


def train_and_predict(
    model_name: ModelName,
    train_contexts: dict[str, np.ndarray],
    pred_contexts: dict[str, np.ndarray],
    horizon: int = 7,
    max_steps: int = 500,
    input_size: int = 60,
    seed: int = 42,
    accelerator: str = "auto",
) -> dict[str, np.ndarray]:
    """训练 DeepAR/TFT 并生成 21 点分位网格预测。

    Returns
    -------
    {series_id: (horizon, 21) quantile grid at NATIVE_LEVELS}
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MQLoss
    from neuralforecast.models import DeepAR, TFT

    levels = list(NATIVE_LEVELS)
    loss = MQLoss(quantiles=levels)
    valid_loss = MQLoss(quantiles=levels)

    common_kw = dict(
        h=horizon,
        loss=loss,
        valid_loss=valid_loss,
        max_steps=max_steps,
        input_size=input_size,
        random_seed=seed,
        accelerator=accelerator,
        enable_progress_bar=False,
        enable_model_summary=False,
    )

    if model_name == "deepar":
        model = DeepAR(
            lstm_hidden_size=64,
            lstm_n_layers=2,
            learning_rate=1e-3,
            batch_size=64,
            **common_kw,
        )
    elif model_name == "tft":
        model = TFT(
            hidden_size=64,
            n_head=4,
            learning_rate=1e-3,
            batch_size=64,
            **common_kw,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    train_df = _build_nf_df(train_contexts)
    pred_df = _build_nf_df(pred_contexts)

    nf = NeuralForecast(models=[model], freq="D")
    nf.fit(df=train_df)
    forecasts = nf.predict(df=pred_df).reset_index()

    # 提取分位数列（排除 unique_id, ds, index）
    meta_cols = {"unique_id", "ds", "index"}
    q_cols = [c for c in forecasts.columns if c not in meta_cols]

    if len(q_cols) != len(levels):
        raise ValueError(
            f"Expected {len(levels)} quantile columns, got {len(q_cols)}: {q_cols}")

    result = {}
    for sid, grp in forecasts.groupby("unique_id"):
        grp = grp.sort_values("ds")
        grid = grp[q_cols].values  # (horizon, 21)
        grid = np.clip(grid, 0.0, None)
        for h in range(grid.shape[0]):
            grid[h] = np.maximum.accumulate(grid[h])
        result[str(sid)] = grid

    return result
