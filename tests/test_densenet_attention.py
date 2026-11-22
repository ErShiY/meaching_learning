import io
import unittest

import torch
import torch.nn.functional as F

from model import build_model
from models.heads import LearnableGeM


class DenseNetAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gem_mean_limit_and_exponent_gradient(self):
        features = torch.rand(2, 4, 6, 6) + 0.1
        gem = LearnableGeM(p=1)
        torch.testing.assert_close(gem(features), features.mean(dim=(-2, -1)))
        gem.p.data.fill_(3)
        gem(features).sum().backward()
        self.assertTrue(torch.isfinite(gem.p.grad))
        self.assertGreater(gem.p.grad.abs().item(), 0)
        self.assertTrue(torch.isfinite(gem(torch.zeros_like(features))).all())

    def test_multiscale_features_and_gradient_paths(self):
        torch.manual_seed(42)
        model = build_model("densenet_attention", 28)
        shapes = {}
        handles = [
            model.classifier.mid_projection.register_forward_pre_hook(
                lambda module, args: shapes.update(mid=tuple(args[0].shape))
            ),
            model.classifier.deep_projection.register_forward_pre_hook(
                lambda module, args: shapes.update(deep=tuple(args[0].shape))
            ),
        ]
        logits = model(torch.randn(2, 4, 64, 64))
        for handle in handles:
            handle.remove()
        self.assertEqual(shapes, {"mid": (2, 1024, 4, 4), "deep": (2, 1024, 2, 2)})
        self.assertEqual(tuple(logits.shape), (2, 28))
        targets = torch.zeros_like(logits)
        targets[:, [0, 21]] = 1
        F.binary_cross_entropy_with_logits(logits, targets).backward()
        for parameter in (
            model.encoder.conv0.weight,
            model.classifier.label_queries,
            model.classifier.mid_projection[0].weight,
            model.classifier.deep_projection[0].weight,
            model.classifier.label_weight,
            model.classifier.global_pool.p,
        ):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_pretrained_encoder_interface_and_checkpoint_roundtrip(self):
        torch.manual_seed(7)
        baseline = build_model("baseline", 28)
        torch.manual_seed(7)
        model = build_model("densenet_attention", 28).eval()
        # Creating the new head must not reinitialize the reused encoder.
        for key, value in baseline.encoder.state_dict().items():
            torch.testing.assert_close(value, model.encoder.state_dict()[key], atol=0, rtol=0)
        images = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            expected = model(images)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = build_model("densenet_attention", 28).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        with torch.no_grad():
            actual = restored(images)
            batched = restored(torch.cat([images, torch.randn_like(images)]))
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
        torch.testing.assert_close(actual, batched[:1], atol=1e-5, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
