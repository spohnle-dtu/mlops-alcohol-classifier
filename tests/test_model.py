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
    backbone_flags = [p.requires_grad for name, p in model.backbone.named_parameters() if not name.startswith("fc.")]
    assert any(flag is False for flag in backbone_flags)


def test_forward_is_deterministic_in_eval_mode():
    """When the model is in eval() mode, repeated forward passes on the same input should return identical logits.

    This catches issues like dropout still active or other non-deterministic layers left enabled in eval mode.
    """
    model = BeverageModel(num_classes=3, pretrained=False).eval()
    x = torch.randn(1, 3, 224, 224)

    y1 = model(x)
    y2 = model(x)

    # exact equality may be too strict for floating point, but in eval mode repeated calls should match
    assert torch.allclose(y1, y2, rtol=1e-6, atol=1e-6)


# New test: verify state_dict save/load reproduces outputs
def test_state_dict_roundtrip_reproduces_outputs():
    """Copy state_dict from one model instance to another and assert outputs match.

    This ensures that loading weights reproduces the same forward behavior.
    """
    model_a = BeverageModel(num_classes=3, pretrained=False)
    model_b = BeverageModel(num_classes=3, pretrained=False)

    # Create deterministic input
    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224)

    # Copy weights from a to b
    model_b.load_state_dict(model_a.state_dict())

    model_a.eval()
    model_b.eval()

    with torch.no_grad():
        y_a = model_a(x)
        y_b = model_b(x)

    assert torch.allclose(y_a, y_b, rtol=1e-6, atol=1e-6)
