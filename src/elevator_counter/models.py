"""Baseline detection models for the elevator counter project."""

from __future__ import annotations

from torch import nn
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn


def build_grayscale_fasterrcnn_mobilenet(
    *,
    num_classes: int = 2,
    image_size: int = 192,
    pretrained_backbone: bool = False,
):
    """Build a small Faster R-CNN baseline adapted for 1-channel 192x192 input."""

    weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained_backbone else None
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None,
        weights_backbone=weights_backbone,
        num_classes=num_classes,
        min_size=image_size,
        max_size=image_size,
    )

    old_conv = model.backbone.body["0"][0]
    new_conv = nn.Conv2d(
        1,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with_weights = old_conv.weight.data
    new_conv.weight.data.copy_(with_weights.mean(dim=1, keepdim=True))
    if old_conv.bias is not None and new_conv.bias is not None:
        new_conv.bias.data.copy_(old_conv.bias.data)
    model.backbone.body["0"][0] = new_conv

    # Use simple grayscale normalization and keep images at their native 192x192 size.
    model.transform.image_mean = [0.5]
    model.transform.image_std = [0.5]
    model.transform.min_size = (image_size,)
    model.transform.max_size = image_size
    return model

