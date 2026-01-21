import torch
from torch import nn
from torchvision import models


class BeverageModel(nn.Module):
    """
    ResNet18 transfer-learning model for 3-class classification.

    - Input: (B, 3, H, W) with H=W=224
    - Output: (B, num_classes) logits
    """

    def __init__(
        self,
        num_classes: int = 3,
        dropout: float = 0.2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        # Ensure the classifier is always trainable even if backbone is frozen
        for p in self.backbone.fc.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input x must be a torch.Tensor")
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 channels (RGB), got C={x.shape[1]}")
        return self.backbone(x)
