"""Channel-aware ResNet18 with mid-level fusion and a multi-label head.

The original four-full-encoder model remains in resnet18.py so existing
``resnet18`` checkpoints can still be loaded without changing their structure.
"""

import copy

import torch
import torch.nn as nn

from dataset import CHANNELS, NUM_CLASSES
from .heads import LabelAttentionHead
from .resnet18 import SingleChannelResNet18Encoder


class ChannelResNet18EarlyEncoder(nn.Module):
    """An independent single-channel ResNet18 up to and including layer2."""

    out_channels = 128

    def __init__(self, prototype):
        super().__init__()
        self.stem = nn.Sequential(
            copy.deepcopy(prototype.conv1),
            copy.deepcopy(prototype.bn1),
            copy.deepcopy(prototype.relu),
            copy.deepcopy(prototype.maxpool),
        )
        self.layer1 = copy.deepcopy(prototype.layer1)
        self.layer2 = copy.deepcopy(prototype.layer2)

    def forward(self, images):
        features = self.stem(images)
        features = self.layer1(features)
        return self.layer2(features)


class AdaptiveChannelFusion(nn.Module):
    """Reweight aligned channel features, then learn a spatial residual.

    Gates start at one. Independent sigmoid gates allow multiple fluorescence
    channels to be useful at once, rather than forcing them to compete.
    """

    def __init__(self, feature_channels=128, num_channels=4):
        super().__init__()
        self.feature_channels = feature_channels
        self.num_channels = num_channels
        concatenated_channels = feature_channels * num_channels
        self.gate = nn.Sequential(
            nn.Linear(2 * concatenated_channels, feature_channels),
            nn.ReLU(inplace=True),
            nn.Linear(feature_channels, concatenated_channels),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

        self.refine = nn.Sequential(
            nn.Conv2d(
                concatenated_channels, feature_channels, kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                feature_channels, feature_channels, kernel_size=3,
                padding=1, bias=False,
            ),
            nn.BatchNorm2d(feature_channels),
        )
        # A small correction initially preserves the scale of pretrained features.
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, channel_features):
        descriptors = []
        for features in channel_features:
            descriptors.extend(
                [features.mean(dim=(-2, -1)), features.amax(dim=(-2, -1))]
            )
        descriptor = torch.cat(descriptors, dim=1)
        gates = 2.0 * torch.sigmoid(self.gate(descriptor))
        gates = gates.reshape(
            descriptor.size(0), self.num_channels, self.feature_channels, 1, 1
        )
        weighted_features = [
            features * gates[:, index]
            for index, features in enumerate(channel_features)
        ]
        mean_features = sum(weighted_features) / self.num_channels
        correction = self.refine(torch.cat(weighted_features, dim=1))
        return self.activation(
            mean_features + self.residual_scale * correction
        )


class ChannelFusionResNet18(nn.Module):
    """Four independent early encoders, shared deep stages and label attention.

    With 512x512 input:
      4 x [B,1,512,512] -> 4 x [B,128,64,64]
      fusion -> [B,128,64,64]
      layer3 -> [B,256,32,32], layer4 -> [B,512,16,16]
      multi-scale label pooling + global head -> [B,num_classes] logits
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained_weights=""):
        super().__init__()
        prototype = SingleChannelResNet18Encoder(
            pretrained_weights=pretrained_weights
        )
        self.encoders = nn.ModuleDict(
            {
                channel: ChannelResNet18EarlyEncoder(prototype)
                for channel in CHANNELS
            }
        )
        self.fusion = AdaptiveChannelFusion(
            ChannelResNet18EarlyEncoder.out_channels, len(CHANNELS)
        )
        self.layer3 = prototype.layer3
        self.layer4 = prototype.layer4
        self.classifier = LabelAttentionHead(num_classes)

    def forward(self, images):
        if images.ndim != 4 or images.size(1) != len(CHANNELS):
            raise ValueError(
                f"Expected input shape [batch, {len(CHANNELS)}, height, width], "
                f"but received {tuple(images.shape)}."
            )
        channel_features = [
            self.encoders[channel](images[:, index : index + 1])
            for index, channel in enumerate(CHANNELS)
        ]
        fused_features = self.fusion(channel_features)
        mid_features = self.layer3(fused_features)
        deep_features = self.layer4(mid_features)
        return self.classifier(mid_features, deep_features)
