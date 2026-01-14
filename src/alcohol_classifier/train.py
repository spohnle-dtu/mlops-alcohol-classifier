from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch.optim import Adam

from data import make_dataloaders
from model import BeverageModelResnet


@hydra.main(config_path="../../configs", config_name="run", version_base="1.3")
def train(cfg: DictConfig) -> None:
    """Train an image classification model using PyTorch.

    This function orchestrates the full training loop:
    - data loading
    - model initialization
    - training and validation loops
    - checkpointing the best model
    - persisting training metrics

    Args:
        cfg: Hydra/OMEGACONF configuration containing:
            - model hyperparameters (lr, epochs, dropout, pretrained, freeze_backbone)
            - dataset parameters (batch_size, seed)
            - output paths for model checkpoints and metrics

    Returns:
        None
    """

    # 1. Initialization & reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(cfg.dataset.seed))

    print("\n--- STARTING TRAINING ---")
    print(f"Device:   {device}")
    print(f"Model:    {cfg.model.name} (LR: {cfg.model.lr}, Epochs: {cfg.model.epochs})")
    print(f"Batch:    {cfg.dataset.batch_size}")
    print(f"Seed:     {cfg.dataset.seed}")
    print("-------------------------\n")

    # 2. Data loaders
    train_loader, val_loader, class_names = make_dataloaders(cfg)

    # 3. Model setup
    model = BeverageModelResnet(
        num_classes=len(class_names),
        dropout=float(cfg.model.dropout),
        pretrained=bool(cfg.model.pretrained),
        freeze_backbone=bool(cfg.model.freeze_backbone),
    ).to(device)

    # 4. Optimizer and loss function
    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer: Adam = Adam(model.parameters(), lr=float(cfg.model.lr))

    history: list[dict[str, float]] = []
    best_acc: float = 0.0

    start_total = time.time()

    # 5. Training loop
    for epoch in range(1, int(cfg.model.epochs) + 1):
        # --- TRAINING PHASE ---
        model.train()
        train_loss: float = 0.0
        train_correct: int = 0
        train_total: int = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Standard PyTorch optimization step
            optimizer.zero_grad()
            outputs: Tensor = model(images)
            loss: Tensor = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            train_loss += loss.item() * images.size(0)
            preds: Tensor = torch.argmax(outputs, dim=1)
            train_correct += int((preds == labels).sum().item())
            train_total += int(labels.size(0))

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss: float = 0.0
        val_correct: int = 0
        val_total: int = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs: Tensor = model(images)
                loss: Tensor = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += int((torch.argmax(outputs, dim=1) == labels).sum().item())
                val_total += int(labels.size(0))

        # Epoch-level metrics
        epoch_train_acc: float = train_correct / train_total if train_total > 0 else 0.0
        epoch_val_acc: float = val_correct / val_total if val_total > 0 else 0.0

        print(
            f"Epoch [{epoch}/{cfg.model.epochs}] - "
            f"Train Acc: {epoch_train_acc:.3f} | Val Acc: {epoch_val_acc:.3f}"
        )

        # Save checkpoint if validation accuracy improves
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            model_path = Path(cfg.path_model)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)

        history.append(
            {
                "epoch": float(epoch),
                "train_acc": epoch_train_acc,
                "val_acc": epoch_val_acc,
            }
        )

    total_duration: float = time.time() - start_total
    print(f"Total Training Time: {total_duration / 60:.2f} minutes\n")

    # 6. Persist training metrics
    metrics_path = Path(cfg.path_metrics_train)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w") as f:
        json.dump(history, f, indent=4)

    print("✅ Training complete. Model checkpoint and metrics saved.")


if __name__ == "__main__":
    # Allow running the training script directly
    train()
