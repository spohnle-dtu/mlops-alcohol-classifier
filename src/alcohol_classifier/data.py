from __future__ import annotations

from loguru import logger

import warnings
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# Some images may be palette-based PNGs. PIL can emit warnings for those; we silence them to keep logs readable.
warnings.filterwarnings("ignore", message="Palette images with Transparency expressed in bytes")


class AlcDataset(Dataset[tuple[Tensor, int]]):
    """Dataset wrapper for preprocessed tensors stored on disk.

    This dataset assumes that preprocessing has already produced two files:
    - images.pt: a tensor of shape (N, C, H, W)
    - labels.pt: a tensor of shape (N,)

    Args:
        images_path: Path to the saved image tensor.
        labels_path: Path to the saved label tensor.

    Returns:
        Each item is a tuple (image, label) where image is a torch.Tensor and label is an int.
    """

    def __init__(self, images_path: str | Path, labels_path: str | Path) -> None:
        super().__init__()
        self.images: Tensor = torch.load(Path(images_path))
        self.labels: Tensor = torch.load(Path(labels_path))

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return int(self.images.shape[0])

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        """Return a single sample.

        Args:
            idx: Index of the sample.

        Returns:
            (image, label)
        """
        image = self.images[idx]
        label = int(self.labels[idx].item())

        return image, label


def get_transforms() -> transforms.Compose:
    """Create image preprocessing transforms.

    Returns:
        A torchvision transforms.Compose object.
    """

    logger.info("Rezinging images to 224x224 and normalizing with ImageNet stats.")

    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Resize((224, 224)),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

def robust_loader(path: str | Path) -> Image.Image:
    """Load an image robustly and convert to RGB.

    ImageFolder may encounter images with different modes (RGBA, palette, grayscale). Converting
    to RGB ensures a consistent 3-channel input for pretrained CNN backbones.

    Args:
        path: File path to the image.

    Returns:
        A PIL RGB image.
    """

    with Image.open(path) as img:
        return img.convert("RGB")


@hydra.main(config_path="../../configs", config_name="run", version_base="1.3")
def preprocess(cfg: DictConfig) -> None:
    """Preprocess raw images into tensors stored in the processed folder.

    This step loads the dataset from an ImageFolder structure, applies the transforms, and writes:
    - images.pt
    - labels.pt
    - classes.pt (class name list)

    Args:
        cfg: Hydra/OMEGACONF configuration.

    Returns:
        None
    """

    logger.info(f"Preprocessing started")
    logger.info(f"Preprocessing images from {cfg.dataset.path_raw}...")
    raw_dataset = datasets.ImageFolder(
        root=cfg.dataset.path_raw,
        transform=get_transforms(),
        loader=robust_loader,
    )

    # We load everything into memory to save it. This is acceptable for small datasets (~3k images).
    all_images: list[Tensor] = []
    all_labels: list[int] = []

    for img, label in raw_dataset:
        all_images.append(img)
        all_labels.append(int(label))

    processed_path = Path(cfg.dataset.path_processed)
    processed_path.mkdir(parents=True, exist_ok=True)

    torch.save(torch.stack(all_images), processed_path / "images.pt")
    torch.save(torch.tensor(all_labels, dtype=torch.long), processed_path / "labels.pt")
    # Persist class names so downstream steps can map indices -> readable labels.
    torch.save(raw_dataset.classes, processed_path / "classes.pt")

    logger.info(f"✅ Preprocessing complete. Files saved in {processed_path}")


def make_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader, list[str]]:
    """Create train/validation dataloaders from preprocessed tensors.

    The split is stratified by class to preserve label distribution in train/val.

    Args:
        cfg: Hydra/OMEGACONF configuration. Expected fields:
            - dataset.path_processed
            - dataset.val_fraction
            - dataset.seed
            - dataset.batch_size
            - dataset.num_workers

    Returns:
        train_loader: Dataloader for the training subset.
        val_loader: Dataloader for the validation subset.
        class_names: List of class names in the same order as the label indices.

    Raises:
        FileNotFoundError: If preprocessed tensors are missing.
    """

    processed_path = Path(cfg.dataset.path_processed)

    images_file = processed_path / "images.pt"
    labels_file = processed_path / "labels.pt"
    classes_file = processed_path / "classes.pt"

    if not images_file.exists() or not labels_file.exists() or not classes_file.exists():
        logger.error("There's been a mistake with the process. Processed data not found.")
        raise FileNotFoundError(
            "Processed data not found. Run preprocessing first: `uv run python src/alcohol_classifier/data.py` "
            "(or the equivalent invoke task) to generate data/processed/*.pt files."
        )

    dataset = AlcDataset(images_file, labels_file)
    class_names: list[str] = torch.load(classes_file)

    logger.info("Splitting into train/val sets...")
    # Split into train/val indices.
    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        test_size=float(cfg.dataset.val_fraction),
        stratify=dataset.labels.tolist(),
        random_state=int(cfg.seed),
    )

    train_ds: Subset[tuple[Tensor, int]] = Subset(dataset, train_idx)
    val_ds: Subset[tuple[Tensor, int]] = Subset(dataset, val_idx)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.dataset.batch_size),
        shuffle=True,
        num_workers=int(cfg.dataset.num_workers),
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.dataset.batch_size),
        shuffle=False,
        num_workers=int(cfg.dataset.num_workers),
        pin_memory=pin_memory,
    )

    # print("✅ Dataloaders successfully created.")
    logger.info("✅ Dataloaders successfully created.")
    return train_loader, val_loader, class_names


if __name__ == "__main__":
    preprocess()
