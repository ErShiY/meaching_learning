import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from dataset import NUM_CLASSES


class DenseNet121Baseline(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained_weights=""):
        super().__init__()

        backbone = models.densenet121(weights=None)
        if pretrained_weights:
            self.load_pretrained_weights(backbone, pretrained_weights)

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
            new_conv.weight[:, 3:4] = old_conv.weight.mean(dim=1, keepdim=True)
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        backbone.features.conv0 = new_conv

        self.encoder = backbone.features
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(backbone.classifier.in_features, num_classes)

    @staticmethod
    def load_pretrained_weights(backbone, pretrained_weights):
        weights_path = Path(pretrained_weights)
        if not weights_path.is_file():
            raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")

        checkpoint = torch.load(
            weights_path, map_location="cpu", weights_only=True
        )
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break

        old_key_pattern = re.compile(
            r"^(.*denselayer\d+\.(?:norm|relu|conv))\."
            r"((?:[12])\.(?:weight|bias|running_mean|running_var))$"
        )
        state_dict = {}
        for name, value in checkpoint.items():
            clean_name = name
            if clean_name.startswith("module."):
                clean_name = clean_name[len("module.") :]
            if clean_name.startswith("backbone."):
                clean_name = clean_name[len("backbone.") :]
            if clean_name.startswith("model."):
                clean_name = clean_name[len("model.") :]

            # Older official torchvision DenseNet weights used names such as
            # "norm.1.weight", while current torchvision uses "norm1.weight".
            match = old_key_pattern.match(clean_name)
            if match:
                clean_name = match.group(1) + match.group(2)

            if clean_name.startswith("classifier."):
                continue
            state_dict[clean_name] = value

        missing_keys, unexpected_keys = backbone.load_state_dict(
            state_dict, strict=False
        )
        print(f"Loaded ImageNet weights from: {weights_path}")
        print(
            f"Missing keys: {len(missing_keys)}, "
            f"unexpected keys: {len(unexpected_keys)}"
        )
        important_missing_keys = [
            key for key in missing_keys if not key.startswith("classifier.")
        ]
        if important_missing_keys:
            raise ValueError(
                "The pretrained file does not match torchvision DenseNet121. "
                f"First missing keys: {important_missing_keys[:5]}"
            )

    def forward(self, images):
        features = self.encoder(images)
        features = F.relu(features, inplace=True)
        features = self.global_pool(features)
        features = torch.flatten(features, start_dim=1)
        logits = self.classifier(features)
        return logits
