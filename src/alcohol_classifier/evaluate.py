from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch import Tensor

from data import make_dataloaders
from model import BeverageModelResnet


@hydra.main(config_path="../../configs", config_name="run", version_base="1.3")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained model on the validation dataset.

    This script loads a trained model checkpoint, runs inference on the
    validation split, computes accuracy, and stores evaluation metrics
    to disk for later inspection or reporting.

    Args:
        cfg: Hydra/OMEGACONF configuration containing:
            - path_model: Path to the trained model checkpoint.
            - path_metrics_eval: Output path for evaluation metrics (JSON).
            - model.dropout: Dropout rate used when instantiating the model.
            - dataset configuration used to build dataloaders.

    Returns:
        None
    """

    # 1. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # 2. Load validation data
    # We only need the validation loader for evaluation.
    _, val_loader, class_names = make_dataloaders(cfg)

    # 3. Load model architecture and weights
    model = BeverageModelResnet(
        num_classes=len(class_names),
        dropout=float(cfg.model.dropout),
        pretrained=False,  # We load our own trained weights, not ImageNet weights
    ).to(device)

    print(f"Loading checkpoint: {cfg.path_model}")
    model.load_state_dict(torch.load(cfg.path_model, map_location=device))
    model.eval()

    # 4. Evaluation loop
    correct: int = 0
    total: int = 0
    start_eval = time.time()

    print("Running evaluation...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs: Tensor = model(images)
            preds: Tensor = torch.argmax(outputs, dim=1)

            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

    # 5. Compute metrics
    accuracy: float = correct / total if total > 0 else 0.0
    duration: float = time.time() - start_eval

    metrics: dict[str, object] = {
        "validation_accuracy": accuracy,
        "eval_duration_seconds": duration,
        "class_names": class_names,
    }

    # 6. Persist results to disk
    metrics_path = Path(cfg.path_metrics_eval)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=4)

    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Time Taken: {duration:.2f}s")
    print(f"Results saved to: {metrics_path}")


if __name__ == "__main__":
    # Allow running the evaluation script directly without Hydra launcher tools
    evaluate()
