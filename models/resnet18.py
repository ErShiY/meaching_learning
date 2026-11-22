import copy
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from dataset import CHANNELS, NUM_CLASSES


class SingleChannelResNet18Encoder(nn.Module):
    """ResNet18 feature encoder for one grayscale fluorescence channel."""

    out_channels = 512

    def __init__(self, pretrained_weights=""):
        super().__init__()

        backbone = models.resnet18(weights=None)
        if pretrained_weights:
            self.load_pretrained_weights(backbone, pretrained_weights)

        old_conv = backbone.conv1
        single_channel_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            single_channel_conv.weight.copy_(
                old_conv.weight.mean(dim=1, keepdim=True)
            )
            if old_conv.bias is not None:
                single_channel_conv.bias.copy_(old_conv.bias)

        self.conv1 = single_channel_conv
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    @staticmethod
    def load_pretrained_weights(backbone, pretrained_weights):
        weights_path = Path(pretrained_weights)
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"ResNet18 pretrained weights not found: {weights_path}"
            )

        checkpoint = torch.load(
            weights_path, map_location="cpu", weights_only=True
        )
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break

        state_dict = {}
        for name, value in checkpoint.items():
            clean_name = name
            for prefix in ("module.", "backbone.", "model."):
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix) :]
            state_dict[clean_name] = value

        missing_keys, unexpected_keys = backbone.load_state_dict(
            state_dict, strict=False
        )
        if missing_keys or unexpected_keys:
            raise ValueError(
                "The pretrained file does not match torchvision ResNet18. "
                f"Missing keys: {missing_keys[:5]}, "
                f"unexpected keys: {unexpected_keys[:5]}"
            )
        print(f"Loaded ResNet18 ImageNet weights from: {weights_path}")

    def forward(self, images):
        features = self.conv1(images)
        features = self.bn1(features)
        features = self.relu(features)
        features = self.maxpool(features)

        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        features = self.layer4(features)
        return features


class FourChannelResNet18(nn.Module):
    """Four independent ResNet18 encoders followed by simple feature fusion."""

    def __init__(self, num_classes=NUM_CLASSES, pretrained_weights=""):
        super().__init__()

        prototype_encoder = SingleChannelResNet18Encoder(
            pretrained_weights=pretrained_weights
        )
        self.encoders = nn.ModuleDict(
            {
                channel: copy.deepcopy(prototype_encoder)
                for channel in CHANNELS
            }
        )

        encoder_channels = prototype_encoder.out_channels
        fused_input_channels = encoder_channels * len(CHANNELS)
        self.fusion = nn.Sequential(
            nn.Conv2d(
                fused_input_channels,
                encoder_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(encoder_channels),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(encoder_channels, num_classes)

    def forward(self, images):
        if images.ndim != 4 or images.size(1) != len(CHANNELS):
            raise ValueError(
                f"Expected input shape [batch, {len(CHANNELS)}, height, width], "
                f"but received {tuple(images.shape)}."
            )

        channel_features = []
        for channel_index, channel in enumerate(CHANNELS):
            channel_image = images[:, channel_index : channel_index + 1]
            features = self.encoders[channel](channel_image)
            channel_features.append(features)

        fused_features = torch.cat(channel_features, dim=1)
        fused_features = self.fusion(fused_features)
        pooled_features = self.global_pool(fused_features)
        pooled_features = torch.flatten(pooled_features, start_dim=1)
        logits = self.classifier(pooled_features)
        return logits
