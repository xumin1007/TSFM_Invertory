"""Decision-aware fine-tuning losses for Chronos-2. WS-3。

两种损失函数：

1. **newsvendor_cost_loss**: 不等权 pinball loss，只在 α 附近的分位数上
   施加损失。等价于 newsvendor 成本 h·(S-y)⁺ + p·(y-S)⁺ 在 S=q_α 时的
   梯度——pinball(α) 的梯度与 newsvendor 成本的次梯度方向一致。

2. **alpha_focused_loss**: 加权版标准 pinball。对 α 附近的分位数给高权重，
   远离 α 的分位数给低权重。保留了对全分布的校准，但重心移向决策相关区域。

用法：在 fit() 前 monkey-patch Chronos2Model._compute_loss。
"""

from __future__ import annotations

import types
from contextlib import contextmanager

import torch
from einops import rearrange


def _make_newsvendor_loss(alpha: float = 0.95):
    """只在最接近 α 的分位数上计算 pinball loss。

    Chronos-2 原生 9 个分位数 [0.1, ..., 0.9]。α=0.95 超出范围，
    取最近的 0.9。这是一个近似——真正的 α-分位数需要外推。
    """
    def newsvendor_compute_loss(
        self, quantile_preds, future_target, future_target_mask,
        patched_future_covariates_mask, loc_scale, num_output_patches,
    ):
        batch_size = future_target.shape[0]
        output_patch_size = self.chronos_config.output_patch_size

        future_target, _ = self.instance_norm(future_target, loc_scale)
        future_target = future_target.unsqueeze(1).to(self.device)
        future_target_mask = (
            future_target_mask.unsqueeze(1).to(self.device)
            if future_target_mask is not None
            else ~torch.isnan(future_target)
        )
        future_target = torch.where(future_target_mask > 0.0, future_target, 0.0)

        if quantile_preds.shape[-1] > future_target.shape[-1]:
            pad = (*future_target.shape[:-1],
                   quantile_preds.shape[-1] - future_target.shape[-1])
            future_target = torch.cat(
                [future_target, torch.zeros(pad).to(future_target)], dim=-1)
            future_target_mask = torch.cat(
                [future_target_mask, torch.zeros(pad).to(future_target_mask)], dim=-1)

        quantiles = self.quantiles  # shape (num_quantiles,)
        # Find closest quantile to alpha
        dists = torch.abs(quantiles - alpha)
        closest_idx = torch.argmin(dists)

        # Weight: 1.0 for closest to alpha, 0.0 for others
        weights = torch.zeros_like(quantiles)
        weights[closest_idx] = 1.0
        weights = rearrange(weights, "q -> 1 q 1")

        quantiles_r = rearrange(quantiles, "q -> 1 q 1")
        pinball = 2 * torch.abs(
            (future_target - quantile_preds)
            * ((future_target <= quantile_preds).float() - quantiles_r))

        inv_mask = 1 - rearrange(
            patched_future_covariates_mask,
            "b n p -> b 1 (n p)",
            b=batch_size, n=num_output_patches, p=output_patch_size)
        loss_mask = future_target_mask.float() * inv_mask

        loss = (pinball * weights * loss_mask)
        loss = loss.mean(dim=-1).sum(dim=-1).mean()
        return loss

    return newsvendor_compute_loss


def _make_alpha_focused_loss(alpha: float = 0.95, focus_width: float = 0.15):
    """加权 pinball loss：α ± focus_width 内高权重，外部低权重。

    权重用高斯核 exp(-0.5 * ((q-α)/σ)²)，σ = focus_width。
    归一化后使权重和 = num_quantiles（保持梯度量级与标准损失相当）。
    """
    def alpha_focused_compute_loss(
        self, quantile_preds, future_target, future_target_mask,
        patched_future_covariates_mask, loc_scale, num_output_patches,
    ):
        batch_size = future_target.shape[0]
        output_patch_size = self.chronos_config.output_patch_size

        future_target, _ = self.instance_norm(future_target, loc_scale)
        future_target = future_target.unsqueeze(1).to(self.device)
        future_target_mask = (
            future_target_mask.unsqueeze(1).to(self.device)
            if future_target_mask is not None
            else ~torch.isnan(future_target)
        )
        future_target = torch.where(future_target_mask > 0.0, future_target, 0.0)

        if quantile_preds.shape[-1] > future_target.shape[-1]:
            pad = (*future_target.shape[:-1],
                   quantile_preds.shape[-1] - future_target.shape[-1])
            future_target = torch.cat(
                [future_target, torch.zeros(pad).to(future_target)], dim=-1)
            future_target_mask = torch.cat(
                [future_target_mask, torch.zeros(pad).to(future_target_mask)], dim=-1)

        quantiles = self.quantiles
        # Gaussian weights centered at alpha
        w = torch.exp(-0.5 * ((quantiles - alpha) / focus_width) ** 2)
        w = w * len(quantiles) / w.sum()  # normalize to preserve gradient magnitude
        weights = rearrange(w, "q -> 1 q 1")

        quantiles_r = rearrange(quantiles, "q -> 1 q 1")
        pinball = 2 * torch.abs(
            (future_target - quantile_preds)
            * ((future_target <= quantile_preds).float() - quantiles_r))

        inv_mask = 1 - rearrange(
            patched_future_covariates_mask,
            "b n p -> b 1 (n p)",
            b=batch_size, n=num_output_patches, p=output_patch_size)
        loss_mask = future_target_mask.float() * inv_mask

        loss = (pinball * weights * loss_mask)
        loss = loss.mean(dim=-1).sum(dim=-1).mean()
        return loss

    return alpha_focused_compute_loss


@contextmanager
def patched_loss(loss_name: str, alpha: float = 0.95, **kwargs):
    """Context manager that monkey-patches Chronos2Model._compute_loss.

    Usage::

        with patched_loss("newsvendor", alpha=0.95):
            pipe.fit(...)
    """
    from chronos.chronos2.model import Chronos2Model

    original = Chronos2Model._compute_loss

    if loss_name == "newsvendor":
        replacement = _make_newsvendor_loss(alpha)
    elif loss_name == "alpha_focused":
        replacement = _make_alpha_focused_loss(alpha, **kwargs)
    elif loss_name == "pinball":
        replacement = original  # no-op, standard loss
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    Chronos2Model._compute_loss = replacement
    try:
        yield
    finally:
        Chronos2Model._compute_loss = original
