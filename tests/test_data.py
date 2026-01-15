from __future__ import annotations

from pathlib import Path

import torch

from src.alcohol_classifier.data import DataConfig, make_dataloaders


def _create_mock_processed_dir(processed_dir: Path, n: int = 40, k: int = 3) -> None:
    """Write dummy all_images.pt / all_labels.pt in the expected format."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    images = torch.rand(n, 3, 224, 224)
    labels = torch.randint(0, k, (n,))
    torch.save(images, processed_dir / "all_images.pt")
    torch.save(labels, processed_dir / "all_labels.pt")


def test_dataloaders_yield_batches(tmp_path: Path):
    processed = tmp_path / "processed"
    _create_mock_processed_dir(processed)

    cfg = DataConfig(
        processed_path=processed,
        batch_size=8,
        val_fraction=0.25,
        seed=123,
        num_workers=0,
    )

    train_loader, val_loader, class_names = make_dataloaders(cfg)

    assert len(class_names) == 3
    assert class_names == ["beer", "whiskey", "wine"]

    xb, yb = next(iter(train_loader))
    assert xb.shape[1:] == (3, 224, 224)
    assert xb.shape[0] == yb.shape[0]
    assert yb.dtype in (torch.int64, torch.int32)


def test_dataset_split_counts(tmp_path: Path):
    processed = tmp_path / "processed"
    _create_mock_processed_dir(processed, n=20)

    cfg = DataConfig(
        processed_path=processed,
        batch_size=4,
        val_fraction=0.2,
        seed=42,
        num_workers=0,
    )

    train_loader, val_loader, _ = make_dataloaders(cfg)

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)

    assert n_train + n_val == 20
    assert n_val in (4, 5)  # because round() can tip either way
