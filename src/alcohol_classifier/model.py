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
        return self.backbone(x)
