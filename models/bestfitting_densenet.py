from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from dataset import CHANNELS, NUM_CLASSES
from .densenet import DenseNet121Baseline


class BestfittingDenseNet121(nn.Module):
    """Bestfitting's 512-pixel HPA classifier, adapted to the project loader."""

    def __init__(self, num_classes=NUM_CLASSES, pretrained_weights=""):
        super().__init__()
        backbone = models.densenet121(weights=None)
        state = None
        if pretrained_weights:
            state = torch.load(Path(pretrained_weights), map_location="cpu", weights_only=True)
            if "encoder.conv0.weight" not in state:
                DenseNet121Baseline.load_pretrained_weights(backbone, pretrained_weights)
                state = None

        old_conv = backbone.features.conv0
        backbone.features.conv0 = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            backbone.features.conv0.weight[:, :3].copy_(old_conv.weight)
            backbone.features.conv0.weight[:, 3:4].copy_(old_conv.weight[:, :1])

        self.encoder = backbone.features
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc1 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.logit = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.channel_order = tuple(CHANNELS.index(name) for name in ("red", "green", "blue", "yellow"))
        self.register_buffer(
            "mean", torch.tensor([0.074598, 0.050630, 0.050891, 0.076287]).view(1, 4, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std", torch.tensor([0.122813, 0.085745, 0.129882, 0.119411]).view(1, 4, 1, 1),
            persistent=False,
        )
        if state is not None:
            self.load_state_dict(state, strict=True)
            print(f"Loaded Bestfitting HPA weights from: {pretrained_weights}")

    @staticmethod
    def normalize_features(features, batch_norm):
        if batch_norm.training and features.size(0) == 1:
            # Some folds leave one image in the last training batch.
            return F.batch_norm(
                features, batch_norm.running_mean, batch_norm.running_var,
                batch_norm.weight, batch_norm.bias, training=False, eps=batch_norm.eps,
            )
        return batch_norm(features)

    def forward(self, images):
        # The shared dataset produces BGRY in [-1, 1]. Upstream uses RGBY in [0, 1].
        images = (images[:, self.channel_order] + 1.0) * 0.5
        images = (images - self.mean) / self.std
        features = F.relu(self.encoder(images), inplace=True)
        pooled = torch.cat(
            (F.adaptive_avg_pool2d(features, 1), F.adaptive_max_pool2d(features, 1)),
            dim=1,
        ).flatten(1)
        pooled = self.dropout(self.normalize_features(pooled, self.bn1))
        features = F.relu(self.fc1(pooled), inplace=True)
        features = self.dropout(self.normalize_features(features, self.bn2))
        return self.logit(features)
