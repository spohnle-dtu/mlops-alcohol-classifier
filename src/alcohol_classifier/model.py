from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision import models


class BeverageModelResnet(nn.Module):
    """ResNet18-based image classification model using transfer learning.

    This model wraps a torchvision ResNet18 backbone and replaces the final
    fully connected layer with a task-specific classification head.

    Design choices:
    - Uses ImageNet-pretrained weights by default to speed up convergence.
    - Optionally freezes the backbone to train only the classification head.
    - Applies dropout before the final linear layer for regularization.

    Input:
        Tensor of shape (B, 3, H, W), where H=W=224 by convention.
    Output:
        Tensor of shape (B, num_classes) containing raw logits.
    """

    def __init__(
        self,
        num_classes: int = 3,
        dropout: float = 0.2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize the ResNet-based classifier.

        Args:
            num_classes: Number of output classes.
            dropout: Dropout probability applied before the final classifier.
            pretrained: If True, initialize backbone with ImageNet weights.
            freeze_backbone: If True, freeze backbone parameters so only the
                classification head is trained.
        """
        super().__init__()

        # Select pretrained weights if requested. Using ImageNet weights is a
        # common and effective strategy for transfer learning on small datasets.
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone: nn.Module = models.resnet18(weights=weights)

        # Optionally freeze the backbone to reduce overfitting and training cost.
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace the original fully connected layer with a custom head.
        in_features: int = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

        # Ensure the classification head is always trainable, even if the
        # backbone has been frozen.
        for param in self.backbone.fc.parameters():
            param.requires_grad = True

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the network.

        Args:
            x: Input batch of images as a tensor of shape (B, 3, H, W).

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        return self.backbone(x)


if __name__ == "__main__":
    # Minimal sanity check: instantiate the model and run a dummy forward pass.
    model = BeverageModelResnet(num_classes=3)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    print(model)
    print(f"Output shape: {output.shape}")
