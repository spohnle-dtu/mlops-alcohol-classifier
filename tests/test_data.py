from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.alcohol_classifier.data import make_dataloaders


def _write_dummy_processed(processed_dir: Path, n: int = 40, k: int = 3) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    images = torch.rand(n, 3, 224, 224)
    labels = torch.randint(0, k, (n,), dtype=torch.long)
    classes = ["beer", "whiskey", "wine"][:k]

    torch.save(images, processed_dir / "images.pt")
    torch.save(labels, processed_dir / "labels.pt")
    torch.save(classes, processed_dir / "classes.pt")


def _make_cfg(processed_dir: Path, *, batch_size=8, val_fraction=0.25, seed=123, num_workers=0):
    return OmegaConf.create(
        {
            "dataset": {
                "path_processed": str(processed_dir),
                "val_fraction": float(val_fraction),
                "seed": int(seed),
                "batch_size": int(batch_size),
                "num_workers": int(num_workers),
            }
        }
    )


def test_dataloaders_yield_batches(tmp_path: Path):
    processed = tmp_path / "processed"
    _write_dummy_processed(processed)

    cfg = _make_cfg(processed, batch_size=8, val_fraction=0.25, seed=123, num_workers=0)
    train_loader, val_loader, class_names = make_dataloaders(cfg)

    assert class_names == ["beer", "whiskey", "wine"]

    xb, yb = next(iter(train_loader))
    assert xb.shape[1:] == (3, 224, 224)
    assert xb.shape[0] == yb.shape[0]

    # val loader should also work
    xb2, yb2 = next(iter(val_loader))
    assert xb2.shape[1:] == (3, 224, 224)
    assert xb2.shape[0] == yb2.shape[0]


def test_split_sizes_add_up(tmp_path: Path):
    processed = tmp_path / "processed"
    _write_dummy_processed(processed, n=20)

    cfg = _make_cfg(processed, batch_size=4, val_fraction=0.2, seed=42, num_workers=0)
    train_loader, val_loader, _ = make_dataloaders(cfg)

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    assert n_train + n_val == 20
