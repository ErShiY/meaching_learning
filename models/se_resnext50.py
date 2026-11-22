from pathlib import Path

import torch
import torch.nn as nn

from dataset import CHANNELS, NUM_CLASSES


class SEResNeXt50(nn.Module):
    """Standard SE-ResNeXt50 32x4d adapted to four-channel HPA images."""

    def __init__(self, num_classes=NUM_CLASSES, pretrained_weights=""):
        super().__init__()
        from timm import create_model

        architecture = "seresnext50_32x4d"
        if not pretrained_weights:
            self.backbone = create_model(
                architecture, pretrained=False, in_chans=len(CHANNELS),
                num_classes=num_classes,
            )
            return

        self.backbone = create_model(architecture, pretrained=False)
        weights_path = Path(pretrained_weights)
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.backbone.load_state_dict(state, strict=True)

        rgb_conv = self.backbone.conv1
        hpa_conv = nn.Conv2d(
            len(CHANNELS), rgb_conv.out_channels, rgb_conv.kernel_size,
            stride=rgb_conv.stride, padding=rgb_conv.padding, bias=False,
        )
        rgb = ("red", "green", "blue")
        with torch.no_grad():
            for index, channel in enumerate(CHANNELS):
                weight = (
                    rgb_conv.weight[:, rgb.index(channel)] if channel in rgb
                    else rgb_conv.weight.mean(dim=1)
                )
                hpa_conv.weight[:, index].copy_(weight)
        self.backbone.conv1 = hpa_conv
        self.backbone.reset_classifier(num_classes)
        print(f"Loaded SE-ResNeXt50 ImageNet weights from: {weights_path}")

    def forward(self, images):
        return self.backbone(images)
