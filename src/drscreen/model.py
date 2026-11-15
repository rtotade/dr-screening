"""Model: a timm CNN backbone with a 5-class ICDR head.

Default backbone efficientnet_b3 is a good accuracy / CPU-latency trade-off for
primary-care deployment. Probability of referable DR = P(grade >= 2).
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from . import NUM_CLASSES, REFERABLE_MIN_GRADE


class DRNet(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b3", pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="")
        feat = self.backbone.num_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(feat, NUM_CLASSES))

    def features(self, x):
        return self.backbone(x)  # B x C x h x w feature map (used by Grad-CAM)

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


def referable_prob(logits: torch.Tensor) -> torch.Tensor:
    return logits.softmax(-1)[:, REFERABLE_MIN_GRADE:].sum(-1)


def save_checkpoint(path, model: DRNet, cfg: dict, extra: dict | None = None):
    torch.save({"state_dict": model.state_dict(), "config": cfg, **(extra or {})}, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = DRNet(cfg["model"]["backbone"], pretrained=False, dropout=cfg["model"].get("dropout", 0.3))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
