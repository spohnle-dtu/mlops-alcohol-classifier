import torch
import hydra
from omegaconf import DictConfig
import json
import time
from pathlib import Path

from data import make_dataloaders
from model import BeverageModelResnet


@hydra.main(config_path="../../configs", config_name="run", version_base="1.3")
def evaluate(cfg: DictConfig):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # 2. Load Data (We only need the val_loader)
    _, val_loader, class_names = make_dataloaders(cfg)

    # 3. Load Model
    model = BeverageModelResnet(
        num_classes=len(class_names),
        dropout=cfg.model.dropout,
        pretrained=False,  # No need for ImageNet weights, we are loading our own
    ).to(device)

    # Load the saved state_dict from training
    print(f"Loading checkpoint: {cfg.path_model}")
    model.load_state_dict(torch.load(cfg.path_model, map_location=device))
    model.eval()

    # 4. Evaluation Loop
    correct = 0
    total = 0
    start_eval = time.time()

    print("Running evaluation...")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # 5. Calculate Metrics
    accuracy = correct / total
    duration = time.time() - start_eval

    metrics = {"test_accuracy": accuracy, "eval_duration_seconds": duration, "class_names": class_names}

    # 6. Save Results
    Path(cfg.path_metrics_eval).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.path_metrics_eval, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Time Taken: {duration:.2f}s")
    print(f"Results saved to: {cfg.path_metrics_eval}")


if __name__ == "__main__":
    evaluate()
