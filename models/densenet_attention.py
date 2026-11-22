"""Four-channel DenseNet121 with multi-scale label pooling and GeM context."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import CHANNELS, NUM_CLASSES
from .densenet import DenseNet121Baseline
from .heads import LabelAttentionHead


class DenseNet121Attention(nn.Module):
    """Preserve the baseline pretrained encoder; replace its aggregation/head.

    For 512 inputs, denseblock3 supplies [B,1024,32,32] local detail while
    norm5/ReLU supplies [B,1024,16,16] semantics. Label queries gather different
    evidence per class from both scales; GeM supplies a global residual path.
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained_weights=""):
        super().__init__()
        baseline = DenseNet121Baseline(
            num_classes=num_classes, pretrained_weights=pretrained_weights
        )
        self.encoder = baseline.encoder
        self.classifier = LabelAttentionHead(
            num_classes=num_classes,
            mid_channels=1024,
            deep_channels=1024,
            global_pooling="gem",
        )

    def forward(self, images):
        if images.ndim != 4 or images.size(1) != len(CHANNELS):
            raise ValueError(
                f"Expected [batch, {len(CHANNELS)}, height, width], "
                f"received {tuple(images.shape)}"
            )
        features = images
        mid_features = None
        for name, layer in self.encoder.named_children():
            features = layer(features)
            if name == "denseblock3":
                # transition3 begins with an in-place ReLU. Clone so it cannot
                # change the captured branch or its autograd saved tensors.
                mid_features = F.relu(features, inplace=False).clone()
        deep_features = F.relu(features, inplace=False)
        return self.classifier(mid_features, deep_features)
