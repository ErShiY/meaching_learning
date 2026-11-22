"""Optional optimizers and explicit per-update learning-rate scheduling."""

import math

import torch


def build_optimizer(model, name="adam", learning_rate=1e-4, weight_decay=0.0):
    """Keep legacy Adam behavior, or use AdamW with un-decayed norm/bias terms."""
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative.")
    if name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    if name != "adamw":
        raise ValueError(f"Unknown optimizer '{name}'. Choose 'adam' or 'adamw'.")

    decayed_parameters = []
    un_decayed_parameters = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1:
            un_decayed_parameters.append(parameter)
        else:
            decayed_parameters.append(parameter)
    parameter_groups = []
    if decayed_parameters:
        parameter_groups.append(
            {"params": decayed_parameters, "weight_decay": weight_decay}
        )
    if un_decayed_parameters:
        parameter_groups.append(
            {"params": un_decayed_parameters, "weight_decay": 0.0}
        )
    return torch.optim.AdamW(
        parameter_groups, lr=learning_rate, weight_decay=weight_decay
    )


class WarmupCosineScheduler:
    """Set LR before each optimizer update, using a zero-based global step.

    For warmup_steps >= 2, warmup includes both its initial fraction and base
    LR. The cosine section starts at base LR and reaches min_lr on the last
    update. A one-update warmup uses only the initial fraction; a one-update
    cosine section uses min_lr. Constructing the scheduler does not mutate LR.
    """

    def __init__(
        self,
        optimizer,
        total_steps,
        warmup_steps=0,
        min_lr=1e-6,
        warmup_start_factor=0.1,
    ):
        self.optimizer = optimizer
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr = float(min_lr)
        self.warmup_start_factor = float(warmup_start_factor)
        self.last_step = -1
        self._validate_configuration()

    def _validate_configuration(self):
        if not isinstance(self.total_steps, int) or self.total_steps < 1:
            raise ValueError("total_steps must be a positive integer.")
        if not isinstance(self.warmup_steps, int) or not (
            0 <= self.warmup_steps < self.total_steps
        ):
            raise ValueError("warmup_steps must satisfy 0 <= warmup < total_steps.")
        if not self.base_lrs or any(
            not math.isfinite(lr) or lr <= 0 for lr in self.base_lrs
        ):
            raise ValueError("Base learning rates must be finite and positive.")
        if not math.isfinite(self.min_lr) or not 0 <= self.min_lr <= min(self.base_lrs):
            raise ValueError("min_lr must be finite and between zero and each base LR.")
        if not math.isfinite(self.warmup_start_factor) or not (
            0 < self.warmup_start_factor <= 1
        ):
            raise ValueError("warmup_start_factor must be in (0, 1].")

    def set_learning_rate(self, step):
        if not isinstance(step, int) or not 0 <= step < self.total_steps:
            raise ValueError(
                f"step must be an integer in [0, {self.total_steps - 1}]."
            )
        if step < self.warmup_steps:
            fraction = step / max(self.warmup_steps - 1, 1)
            factor = self.warmup_start_factor + (
                1.0 - self.warmup_start_factor
            ) * fraction
            learning_rates = [base_lr * factor for base_lr in self.base_lrs]
        else:
            decay_steps = self.total_steps - self.warmup_steps
            fraction = (
                (step - self.warmup_steps) / (decay_steps - 1)
                if decay_steps > 1 else 1.0
            )
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * fraction))
            learning_rates = [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]
        for group, learning_rate in zip(self.optimizer.param_groups, learning_rates):
            group["lr"] = learning_rate
        self.last_step = step
        return learning_rates

    def state_dict(self):
        return {
            "base_lrs": list(self.base_lrs),
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
            "warmup_start_factor": self.warmup_start_factor,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state):
        if len(state["base_lrs"]) != len(self.optimizer.param_groups):
            raise ValueError("Saved scheduler and optimizer have different group counts.")
        self.base_lrs = [float(value) for value in state["base_lrs"]]
        self.total_steps = state["total_steps"]
        self.warmup_steps = state["warmup_steps"]
        self.min_lr = float(state["min_lr"])
        self.warmup_start_factor = float(state["warmup_start_factor"])
        self._validate_configuration()
        last_step = state["last_step"]
        if not isinstance(last_step, int) or not -1 <= last_step < self.total_steps:
            raise ValueError("Saved scheduler has an invalid last_step.")
        self.last_step = last_step
        if last_step >= 0:
            self.set_learning_rate(last_step)
