from __future__ import annotations

import pytest
import torch

from src.alcohol_classifier.model import BeverageModel


def test_forward_returns_correct_logits_shape():
    model = BeverageModel(num_classes=3, pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 3)


def test_forward_rejects_missing_batch_dim():
    model = BeverageModel(num_classes=3, pretrained=False).eval()

    with pytest.raises(ValueError, match=r"Expected input shape"):
        model(torch.randn(3, 224, 224))


def test_forward_rejects_non_rgb_inputs():
    model = BeverageModel(num_classes=3, pretrained=False).eval()

    with pytest.raises(ValueError, match=r"Expected 3 channels"):
        model(torch.randn(2, 1, 224, 224))


def test_freezing_backbone_leaves_head_trainable():
    model = BeverageModel(num_classes=3, pretrained=False, freeze_backbone=True)

    # Head must always be trainable
    assert all(p.requires_grad for p in model.backbone.fc.parameters())

    # Some backbone params should be frozen
    backbone_flags = [
        p.requires_grad
        for name, p in model.backbone.named_parameters()
        if not name.startswith("fc.")
    ]
    assert any(flag is False for flag in backbone_flags)
