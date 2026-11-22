import io
import unittest

import torch
import torch.nn.functional as F

from dataset import CHANNELS
from model import build_model


class ChannelFusionResNet18Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)
        self.model = build_model("resnet18_fusion", 28)

    def test_independent_encoders_and_identity_gates(self):
        convolution_pointers = {
            self.model.encoders[channel].stem[0].weight.data_ptr()
            for channel in CHANNELS
        }
        bn_pointers = {
            self.model.encoders[channel].stem[1].running_mean.data_ptr()
            for channel in CHANNELS
        }
        self.assertEqual(len(convolution_pointers), 4)
        self.assertEqual(len(bn_pointers), 4)
        with torch.no_grad():
            gates = 2 * self.model.fusion.gate(torch.randn(2, 1024)).sigmoid()
        torch.testing.assert_close(gates, torch.ones_like(gates))

    def test_two_training_steps_reach_all_model_parts(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        images = torch.randn(2, 4, 64, 64)
        targets = torch.zeros(2, 28)
        targets[0, [0, 2]] = 1
        targets[1, [1, 25]] = 1
        before = self.model.encoders["green"].stem[0].weight.detach().clone()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            self.assertEqual(tuple(logits.shape), (2, 28))
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            # Includes the gate's early layer after its zero-initialized final
            # layer has learned during the first optimizer step.
            selected_parameters = {
                f"encoder_{channel}": self.model.encoders[channel].stem[0].weight
                for channel in CHANNELS
            }
            selected_parameters.update(
                fusion=self.model.fusion.refine[0].weight,
                gate_output=self.model.fusion.gate[-1].weight,
                shared_layer3=self.model.layer3[0].conv1.weight,
                shared_layer4=self.model.layer4[0].conv1.weight,
                queries=self.model.classifier.label_queries,
                attention=self.model.classifier.cross_attention.in_proj_weight,
                mid_projection=self.model.classifier.mid_projection[0].weight,
                deep_projection=self.model.classifier.deep_projection[0].weight,
                per_label=self.model.classifier.label_weight,
                global_head=self.model.classifier.global_head[-1].weight,
            )
            for name, parameter in selected_parameters.items():
                with self.subTest(parameter=name):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
            optimizer.step()
        self.assertGreater(self.model.fusion.gate[0].weight.grad.abs().sum(), 0)
        self.assertFalse(torch.equal(before, self.model.encoders["green"].stem[0].weight))

    def test_eval_has_no_cross_sample_attention(self):
        self.model.eval()
        images = torch.randn(2, 4, 64, 64)
        with torch.no_grad():
            batched = self.model(images)
            single = self.model(images[:1])
        torch.testing.assert_close(batched[:1], single, atol=1e-5, rtol=1e-4)

    def test_checkpoint_roundtrip_and_custom_label_count(self):
        self.model.eval()
        images = torch.randn(1, 4, 64, 64)
        with torch.no_grad():
            expected = self.model(images)
        buffer = io.BytesIO()
        torch.save(
            {
                "model_name": "resnet18_fusion",
                "model_state_dict": self.model.state_dict(),
            },
            buffer,
        )
        buffer.seek(0)
        checkpoint = torch.load(buffer, weights_only=True)
        restored = build_model(checkpoint["model_name"], 28)
        restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
        restored.eval()
        with torch.no_grad():
            actual = restored(images)
            custom = build_model("resnet18_fusion", 5).eval()(images)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(tuple(custom.shape), (1, 5))

    def test_rejects_incorrect_channel_count(self):
        with self.assertRaisesRegex(ValueError, "Expected input shape"):
            self.model(torch.randn(2, 3, 64, 64))


if __name__ == "__main__":
    unittest.main()
