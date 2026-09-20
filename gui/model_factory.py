"""Build the trained architecture without downloading ImageNet weights.

The saved checkpoint replaces every parameter, so inference only needs the
architecture. This subclass preserves the exact module names/state-dict layout
from model_cnn_lstm.CNNLSTM while constructing torchvision backbones with
``weights=None``.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import efficientnet_b3, resnet50, resnet152, vgg16, vgg19

from model_cnn_lstm import CNNLSTM


class CheckpointCNNLSTM(CNNLSTM):
    def _create_backbone(self, backbone_name: str):
        if backbone_name == "vgg16":
            return vgg16(weights=None).features, 512
        if backbone_name == "vgg19":
            return vgg19(weights=None).features, 512
        if backbone_name == "resnet50":
            model = resnet50(weights=None)
            return nn.Sequential(*list(model.children())[:-2]), 2048
        if backbone_name == "resnet152":
            model = resnet152(weights=None)
            return nn.Sequential(*list(model.children())[:-2]), 2048
        if backbone_name == "efficientnet_b3":
            return efficientnet_b3(weights=None).features, 1536
        raise ValueError(f"Unsupported backbone: {backbone_name}")
