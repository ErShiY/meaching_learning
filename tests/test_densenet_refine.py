import io
import unittest

import torch
import torch.nn.functional as F

from model import build_model
from models.densenet_refine import MultiScaleResidualNeck


class DenseNetRefineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_neck_preserves_non_square_feature_shapes(self):
        neck = MultiScaleResidualNeck(hidden_channels=32)
        c2 = torch.rand(2, 512, 9, 13)
        c3 = torch.rand(2, 1024, 4, 6)
        c4 = torch.rand(2, 1024, 2, 3)
        enhanced3, enhanced4 = neck(c2, c3, c4)
        self.assertEqual(enhanced3.shape, c3.shape)
        self.assertEqual(enhanced4.shape, c4.shape)
        self.assertTrue(torch.isfinite(enhanced3).all())
        self.assertTrue(torch.isfinite(enhanced4).all())
        with torch.no_grad():
            neck.mid_residual_scale.zero_()
            neck.deep_residual_scale.zero_()
        actual3, actual4 = neck(c2, c3, c4)
        torch.testing.assert_close(actual3, c3, atol=0, rtol=0)
        torch.testing.assert_close(actual4, c4, atol=0, rtol=0)

    def test_attention_checkpoint_transfer_and_identity_residual(self):
        torch.manual_seed(42)
        previous = build_model("densenet_attention", 28).eval()
        refined = build_model("densenet_refine", 28).eval()
        checkpoint = {
            "model_name": "densenet_attention",
            "model_state_dict": previous.state_dict(),
        }
        refined.initialize_from_checkpoint(checkpoint)
        state = refined.state_dict()
        for key, value in previous.state_dict().items():
            torch.testing.assert_close(value, state[key], atol=0, rtol=0)
        with torch.no_grad():
            refined.neck.mid_residual_scale.zero_()
            refined.neck.deep_residual_scale.zero_()
            images = torch.randn(1, 4, 64, 64)
            expected = previous(images)
            actual = refined(images)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_invalid_transfer_is_rejected_before_mutation(self):
        model = build_model("densenet_refine", 28)
        source = build_model("densenet_attention", 28).state_dict()
        original = model.encoder.conv0.weight.detach().clone()
        broken = dict(source)
        del broken["classifier.label_queries"]
        with self.assertRaisesRegex(ValueError, "keys mismatch"):
            model.initialize_from_checkpoint(
                {"model_name": "densenet_attention", "model_state_dict": broken}
            )
        broken = dict(source)
        broken["classifier.label_queries"] = torch.zeros(29, 256)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            model.initialize_from_checkpoint(
                {"model_name": "densenet_attention", "model_state_dict": broken}
            )
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            model.initialize_from_checkpoint(
                {"model_name": "baseline", "model_state_dict": source}
            )
        torch.testing.assert_close(original, model.encoder.conv0.weight, atol=0, rtol=0)

    def test_all_feature_branches_and_classifier_receive_gradients(self):
        torch.manual_seed(17)
        model = build_model("densenet_refine", 28)
        images = torch.randn(2, 4, 64, 64)
        logits = model(images)
        self.assertEqual(tuple(logits.shape), (2, 28))
        targets = torch.zeros_like(logits)
        targets[:, [0, 14]] = 1
        F.binary_cross_entropy_with_logits(logits, targets).backward()
        selected = {
            "encoder": model.encoder.conv0.weight,
            "c2": model.neck.lateral2[0].weight,
            "c3": model.neck.lateral3[0].weight,
            "c4": model.neck.lateral4[0].weight,
            "mid_scale_fusion": model.neck.mid_fusion_weights,
            "deep_scale_fusion": model.neck.deep_fusion_weights,
            "local_conv": model.neck.mid_refine.local_conv[0].weight,
            "channel_attention": model.neck.mid_refine.attention.channel_mlp[0].weight,
            "spatial_attention": model.neck.deep_refine.attention.spatial_conv.weight,
            "mid_residual": model.neck.mid_residual_scale,
            "deep_residual": model.neck.deep_residual_scale,
            "label_queries": model.classifier.label_queries,
            "gem": model.classifier.global_pool.p,
        }
        for name, parameter in selected.items():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_refine_checkpoint_roundtrip_and_custom_labels(self):
        model = build_model("densenet_refine", 5).eval()
        images = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            expected = model(images)
        self.assertEqual(tuple(expected.shape), (1, 5))
        buffer = io.BytesIO()
        torch.save(
            {"model_name": "densenet_refine", "model_state_dict": model.state_dict()},
            buffer,
        )
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        restored = build_model("densenet_refine", 5).eval()
        restored.initialize_from_checkpoint(checkpoint)
        with torch.no_grad():
            actual = restored(images)
            batched = restored(torch.cat([images, torch.randn_like(images)]))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual, batched[:1], atol=1e-5, rtol=1e-4)
        with self.assertRaisesRegex(ValueError, "Expected"):
            restored(torch.randn(1, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()
