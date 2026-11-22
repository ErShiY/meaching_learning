import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from dataset import NUM_CLASSES
from .densenet import DenseNet121Baseline


class ExperimentalDenseNet121(nn.Module):
    """Editable model used for experiments without changing the baseline.

    The initial architecture intentionally matches DenseNet121Baseline. Future
    channel, pooling and classifier changes should be implemented in this file.
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained_weights=""):
        super().__init__()

        backbone = models.densenet121(weights=None)
        if pretrained_weights:
            DenseNet121Baseline.load_pretrained_weights(
                backbone, pretrained_weights
            )

        old_conv = backbone.features.conv0
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:4] = old_conv.weight.mean(
                dim=1, keepdim=True
            )
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        backbone.features.conv0 = new_conv

        self.encoder = backbone.features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(
            backbone.classifier.in_features, num_classes
        )

    def forward(self, images):
        features = self.encoder(images)
        features = F.relu(features, inplace=True)
        features = self.global_pool(features)
        features = torch.flatten(features, start_dim=1)
        logits = self.classifier(features)
        return logits
