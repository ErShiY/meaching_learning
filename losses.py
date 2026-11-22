"""Loss functions for image-level multi-label classification."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidFocalLoss(nn.Module):
    """Binary focal loss averaged over all images and independent labels."""

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        if not math.isfinite(gamma) or gamma < 0:
            raise ValueError("gamma must be finite and non-negative.")
        if alpha is not None and not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1].")
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        if logits.dtype not in (torch.float32, torch.float64):
            logits = logits.float()
        targets = targets.to(dtype=logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if self.gamma:
            # Compute 1 - p_t without subtracting a saturated sigmoid from 1.
            error_probability = (
                targets * torch.sigmoid(-logits)
                + (1.0 - targets) * torch.sigmoid(logits)
            )
            focal_weight = error_probability.clamp_min(torch.finfo(logits.dtype).tiny)
            loss = loss * focal_weight.pow(self.gamma)
        if self.alpha is not None:
            loss = loss * (self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets))
        return loss.mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric focal loss on logits, with clipped easy negatives.

    This follows the positive/negative weighting in Alibaba-MIIL/ASL. Focal
    weights are detached by default, as in its original AsymmetricLoss class;
    the option is explicit here because their optimized implementation uses a
    different default. Mean reduction averages over both samples and labels.

    ASL addresses positive/negative label imbalance. It is not class-frequency
    reweighting and cannot compensate for a lack of rare positive examples.
    """

    def __init__(
        self,
        gamma_pos=0.0,
        gamma_neg=4.0,
        clip=0.05,
        detach_focal_weight=True,
        reduction="mean",
    ):
        super().__init__()
        if not math.isfinite(gamma_pos) or gamma_pos < 0:
            raise ValueError("gamma_pos must be finite and non-negative.")
        if not math.isfinite(gamma_neg) or gamma_neg < 0:
            raise ValueError("gamma_neg must be finite and non-negative.")
        if not math.isfinite(clip) or not 0 <= clip < 1:
            raise ValueError("clip must be finite and in [0, 1).")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'.")
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.clip = float(clip)
        self.detach_focal_weight = bool(detach_focal_weight)
        self.reduction = reduction

    def forward(self, logits, targets):
        if logits.shape != targets.shape:
            raise ValueError(
                f"Logits and targets must have identical shapes, received "
                f"{tuple(logits.shape)} and {tuple(targets.shape)}."
            )
        # Avoid fp16 sigmoid/log underflow while retaining fp64 for numerical QA.
        if logits.dtype not in (torch.float32, torch.float64):
            logits = logits.float()
        targets = targets.to(dtype=logits.dtype)

        positive_log_probability = F.logsigmoid(logits)
        negative_log_probability = F.logsigmoid(-logits)
        if self.clip:
            # log(min(sigmoid(-logits) + clip, 1)), computed without first
            # rounding sigmoid to zero. The clipped plateau has zero gradient.
            negative_log_probability = torch.logaddexp(
                negative_log_probability,
                logits.new_tensor(math.log(self.clip)),
            ).clamp(max=0.0)

        log_likelihood = (
            targets * positive_log_probability
            + (1.0 - targets) * negative_log_probability
        )
        if self.gamma_pos or self.gamma_neg:
            positive_probability = torch.sigmoid(logits)
            negative_probability = negative_log_probability.exp()
            true_probability = (
                targets * positive_probability
                + (1.0 - targets) * negative_probability
            )
            gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            # The small floor also keeps fractional-gamma gradients finite.
            focal_weight = (1.0 - true_probability).clamp(min=1e-8).pow(gamma)
            if self.detach_focal_weight:
                focal_weight = focal_weight.detach()
            log_likelihood = log_likelihood * focal_weight

        loss = -log_likelihood
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
