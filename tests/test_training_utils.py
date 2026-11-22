import copy
import math
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import AsymmetricLoss
from training_utils import WarmupCosineScheduler, build_optimizer


class AsymmetricLossTests(unittest.TestCase):
    def test_disabled_asymmetry_matches_bce_value_and_gradients(self):
        logits = torch.tensor([[-1000.0, -1.7, 0.0, 2.1, 1000.0]], requires_grad=True)
        targets = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0]])
        actual = AsymmetricLoss(gamma_pos=0, gamma_neg=0, clip=0)(logits, targets)
        expected = F.binary_cross_entropy_with_logits(logits, targets)
        actual_gradient = torch.autograd.grad(actual, logits)[0]
        expected_gradient = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_gradient, expected_gradient)

    def test_matches_explicit_clipping_and_detached_focal_formula(self):
        logits = torch.tensor([[-2.0, -0.5, 0.3, 2.4]], requires_grad=True)
        targets = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
        actual = AsymmetricLoss(gamma_pos=1, gamma_neg=4, clip=0.05)(logits, targets)
        p_positive = logits.sigmoid()
        p_negative = ((1 - p_positive) + 0.05).clamp(max=1)
        log_likelihood = targets * p_positive.log() + (1 - targets) * p_negative.log()
        true_probability = targets * p_positive + (1 - targets) * p_negative
        gamma = targets + (1 - targets) * 4
        expected = -(log_likelihood * ((1 - true_probability) ** gamma).detach()).mean()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, logits)[0],
            torch.autograd.grad(expected, logits)[0],
        )

    def test_extreme_logits_and_half_precision_remain_finite(self):
        for detach in (True, False):
            logits = torch.tensor(
                [[-1000, 1000, -1000, 1000]], dtype=torch.float16, requires_grad=True
            )
            targets = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
            actual = AsymmetricLoss(detach_focal_weight=detach)(logits, targets)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertTrue(torch.isfinite(actual))
            actual.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_grad_mode_is_not_changed_and_reductions_are_consistent(self):
        logits = torch.tensor([[-2.0, 0.5], [1.0, 2.0]], requires_grad=True)
        targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        unreduced = AsymmetricLoss(reduction="none")(logits, targets)
        self.assertTrue(torch.is_grad_enabled())
        torch.testing.assert_close(AsymmetricLoss()(logits, targets), unreduced.mean())
        torch.testing.assert_close(
            AsymmetricLoss(reduction="sum")(logits, targets), unreduced.sum()
        )
        with torch.no_grad():
            AsymmetricLoss()(logits, targets)
            self.assertFalse(torch.is_grad_enabled())


class OptimizerTests(unittest.TestCase):
    def test_default_adam_preserves_legacy_update(self):
        torch.manual_seed(7)
        first = nn.Linear(3, 2)
        second = copy.deepcopy(first)
        actual = build_optimizer(first)
        expected = torch.optim.Adam(second.parameters(), lr=1e-4)
        inputs = torch.randn(4, 3)
        first(inputs).square().mean().backward()
        second(inputs).square().mean().backward()
        actual.step()
        expected.step()
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_adamw_does_not_decay_biases_or_normalization(self):
        model = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4), nn.Linear(4, 2))
        optimizer = build_optimizer(model, name="adamw", weight_decay=0.03)
        group_decay = {
            id(parameter): group["weight_decay"]
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(len(group_decay), sum(1 for _ in model.parameters()))
        for parameter in model.parameters():
            self.assertEqual(group_decay[id(parameter)], 0.0 if parameter.ndim <= 1 else 0.03)


class WarmupCosineTests(unittest.TestCase):
    def make_optimizer(self, lr=1e-3):
        return torch.optim.Adam([nn.Parameter(torch.ones(1))], lr=lr)

    def test_warmup_cosine_endpoints_and_monotonic_sections(self):
        optimizer = self.make_optimizer()
        scheduler = WarmupCosineScheduler(
            optimizer, total_steps=10, warmup_steps=4, min_lr=1e-5
        )
        self.assertEqual(optimizer.param_groups[0]["lr"], 1e-3)
        values = [scheduler.set_learning_rate(step)[0] for step in range(10)]
        self.assertAlmostEqual(values[0], 1e-4)
        self.assertAlmostEqual(values[3], 1e-3)
        self.assertAlmostEqual(values[4], 1e-3)
        self.assertAlmostEqual(values[-1], 1e-5)
        self.assertEqual(values[:4], sorted(values[:4]))
        self.assertEqual(values[4:], sorted(values[4:], reverse=True))
        self.assertAlmostEqual(values[6], 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * 2 / 5)))

    def test_no_warmup_and_short_schedules(self):
        scheduler = WarmupCosineScheduler(self.make_optimizer(), 3, min_lr=1e-5)
        self.assertAlmostEqual(scheduler.set_learning_rate(0)[0], 1e-3)
        self.assertAlmostEqual(scheduler.set_learning_rate(2)[0], 1e-5)
        one_step = WarmupCosineScheduler(self.make_optimizer(), 1, min_lr=1e-5)
        self.assertAlmostEqual(one_step.set_learning_rate(0)[0], 1e-5)
        one_warmup = WarmupCosineScheduler(self.make_optimizer(), 4, 1)
        self.assertAlmostEqual(one_warmup.set_learning_rate(0)[0], 1e-4)
        self.assertAlmostEqual(one_warmup.set_learning_rate(1)[0], 1e-3)

    def test_roundtrip_restores_schedule_and_current_learning_rate(self):
        original = WarmupCosineScheduler(self.make_optimizer(), 10, 3, min_lr=1e-5)
        original.set_learning_rate(5)
        optimizer = self.make_optimizer(lr=0.1)
        restored = WarmupCosineScheduler(optimizer, 2)
        restored.load_state_dict(original.state_dict())
        self.assertEqual(restored.state_dict(), original.state_dict())
        self.assertEqual(optimizer.param_groups[0]["lr"], original.optimizer.param_groups[0]["lr"])
        self.assertEqual(restored.set_learning_rate(6), original.set_learning_rate(6))

    def test_rejects_invalid_configuration_or_step(self):
        with self.assertRaises(ValueError):
            WarmupCosineScheduler(self.make_optimizer(), 5, 5)
        with self.assertRaises(ValueError):
            WarmupCosineScheduler(self.make_optimizer(), 5, min_lr=0.1)
        scheduler = WarmupCosineScheduler(self.make_optimizer(), 5)
        for step in (-1, 5, 0.5):
            with self.assertRaises(ValueError):
                scheduler.set_learning_rate(step)


if __name__ == "__main__":
    unittest.main()
