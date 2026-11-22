"""DenseNet feature refinement, independent of the existing attention model."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import CHANNELS, NUM_CLASSES
from .densenet_attention import DenseNet121Attention


class FeatureCBAM(nn.Module):
    """Channel and spatial reweighting of learned CNN features, not raw stains."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size=7, padding=3, bias=False
        )

    def forward(self, features):
        average = features.mean(dim=(-2, -1), keepdim=True)
        maximum = features.amax(dim=(-2, -1), keepdim=True)
        channel_weights = torch.sigmoid(
            self.channel_mlp(average) + self.channel_mlp(maximum)
        )
        features = features * channel_weights
        spatial_descriptor = torch.cat(
            [
                features.mean(dim=1, keepdim=True),
                features.amax(dim=1, keepdim=True),
            ],
            dim=1,
        )
        return features * torch.sigmoid(self.spatial_conv(spatial_descriptor))


class LocalFeatureRefinement(nn.Module):
    """Lightweight neighborhood interaction followed by feature attention."""

    def __init__(self, channels):
        super().__init__()
        self.local_conv = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=1,
                groups=channels, bias=False,
            ),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, channels),
            nn.GELU(),
        )
        self.attention = FeatureCBAM(channels)

    def forward(self, features):
        return self.attention(self.local_conv(features))


class MultiScaleResidualNeck(nn.Module):
    """Align C2/C3/C4 and return residual-enhanced C3/C4 of the same shape.

    At 512px: C2=512x64x64, C3=1024x32x32, C4=1024x16x16.
    C2 supplies earlier detail; upsampling C4 supplies semantic guidance, not
    reconstruction of details already lost by downsampling the input.
    """

    def __init__(self, hidden_channels=256):
        super().__init__()
        if hidden_channels < 32 or hidden_channels % 32:
            raise ValueError("hidden_channels must be a positive multiple of 32.")

        def lateral(in_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
                nn.GroupNorm(32, hidden_channels),
                nn.GELU(),
            )

        def output_projection():
            return nn.Sequential(
                nn.Conv2d(hidden_channels, 1024, kernel_size=1, bias=False),
                nn.GroupNorm(32, 1024),
            )

        self.lateral2 = lateral(512)
        self.lateral3 = lateral(1024)
        self.lateral4 = lateral(1024)
        self.mid_fusion_weights = nn.Parameter(torch.zeros(3))
        self.deep_fusion_weights = nn.Parameter(torch.zeros(2))
        self.mid_refine = LocalFeatureRefinement(hidden_channels)
        self.deep_refine = LocalFeatureRefinement(hidden_channels)
        self.mid_output = output_projection()
        self.deep_output = output_projection()
        self.mid_residual_scale = nn.Parameter(torch.tensor(0.1))
        self.deep_residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, shallow_features, mid_features, deep_features):
        lateral2 = self.lateral2(shallow_features)
        lateral3 = self.lateral3(mid_features)
        lateral4 = self.lateral4(deep_features)

        weights3 = self.mid_fusion_weights.softmax(dim=0)
        aligned2 = F.adaptive_avg_pool2d(lateral2, mid_features.shape[-2:])
        aligned4 = F.interpolate(
            lateral4, size=mid_features.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        refined3 = self.mid_refine(
            weights3[0] * aligned2
            + weights3[1] * lateral3
            + weights3[2] * aligned4
        )

        weights4 = self.deep_fusion_weights.softmax(dim=0)
        aligned3 = F.adaptive_avg_pool2d(refined3, deep_features.shape[-2:])
        refined4 = self.deep_refine(
            weights4[0] * lateral4 + weights4[1] * aligned3
        )

        enhanced3 = F.relu(
            mid_features + self.mid_residual_scale * self.mid_output(refined3),
            inplace=False,
        )
        enhanced4 = F.relu(
            deep_features + self.deep_residual_scale * self.deep_output(refined4),
            inplace=False,
        )
        return enhanced3, enhanced4


class DenseNet121Refine(nn.Module):
    """Pretrained DenseNet -> residual feature neck -> existing label/GeM head.

    ``pretrained_weights`` supports either the official local DenseNet ImageNet
    weights, a complete densenet_attention checkpoint (encoder/head warm start),
    or a complete densenet_refine checkpoint (weight initialization only).
    Optimizer, schedule, split and epoch are always controlled by train.py.
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained_weights=""):
        super().__init__()
        initialization = None
        imagenet_path = pretrained_weights
        if pretrained_weights:
            weights_path = Path(pretrained_weights)
            if not weights_path.is_file():
                raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")
            loaded = torch.load(weights_path, map_location="cpu", weights_only=True)
            if isinstance(loaded, dict) and "model_state_dict" in loaded:
                source_model = loaded.get("model_name")
                if source_model not in ("densenet_attention", "densenet_refine"):
                    raise ValueError(
                        "Expected ImageNet weights or a complete densenet_attention/"
                        f"densenet_refine checkpoint, received model {source_model!r}."
                    )
                initialization = loaded
                imagenet_path = ""

        base = DenseNet121Attention(
            num_classes=num_classes, pretrained_weights=imagenet_path
        )
        self.encoder = base.encoder
        self.classifier = base.classifier
        self.neck = MultiScaleResidualNeck()
        if initialization is not None:
            self.initialize_from_checkpoint(initialization)

    def initialize_from_checkpoint(self, checkpoint):
        """Validate all shared keys/shapes before copying a previous model."""
        source_model = checkpoint.get("model_name")
        if source_model not in ("densenet_attention", "densenet_refine"):
            raise ValueError(f"Unsupported initialization model: {source_model!r}")
        incoming = checkpoint.get("model_state_dict")
        if not isinstance(incoming, dict):
            raise ValueError("Checkpoint must contain a model_state_dict mapping.")
        current = self.state_dict()
        expected = {
            key for key in current
            if source_model == "densenet_refine" or not key.startswith("neck.")
        }
        missing = expected - set(incoming)
        unexpected = set(incoming) - expected
        if missing or unexpected:
            raise ValueError(
                f"Initialization keys mismatch: {len(missing)} missing, "
                f"{len(unexpected)} unexpected."
            )
        for key in expected:
            if not isinstance(incoming[key], torch.Tensor) or (
                incoming[key].shape != current[key].shape
            ):
                raise ValueError(f"Initialization tensor shape mismatch: {key}")
        result = self.load_state_dict(
            incoming, strict=source_model == "densenet_refine"
        )
        if source_model == "densenet_attention":
            expected_missing = {key for key in current if key.startswith("neck.")}
            if set(result.missing_keys) != expected_missing or result.unexpected_keys:
                raise RuntimeError("Only new neck parameters may be missing.")
        print(
            f"Initialized encoder/classifier from {source_model}; "
            "this initializes weights, not optimizer or training progress."
        )

    def forward(self, images):
        if images.ndim != 4 or images.size(1) != len(CHANNELS):
            raise ValueError(
                f"Expected [batch, {len(CHANNELS)}, height, width], "
                f"received {tuple(images.shape)}"
            )
        features = images
        for name, layer in self.encoder.named_children():
            features = layer(features)
            if name == "denseblock2":
                shallow_features = F.relu(features, inplace=False).clone()
            elif name == "denseblock3":
                mid_features = F.relu(features, inplace=False).clone()
        deep_features = F.relu(features, inplace=False)
        mid_features, deep_features = self.neck(
            shallow_features, mid_features, deep_features
        )
        return self.classifier(mid_features, deep_features)
