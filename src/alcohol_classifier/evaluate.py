import time

import hydra
import torch
from omegaconf import DictConfig

from src.alcohol_classifier.data import make_dataloaders
from src.alcohol_classifier.model import BeverageModel
from src.alcohol_classifier.utils import _get_device, _set_seed


@hydra.main(config_path="../../configs", config_name="run", version_base="1.3")
def evaluate(cfg: DictConfig) -> None:
    _set_seed(cfg.seed)
    device = _get_device(cfg.device)

    _, val_loader, class_names = make_dataloaders(cfg)

    checkpoint = torch.load(cfg.path_model, map_location=device)

    model = BeverageModel(num_classes=len(class_names), dropout=cfg.model.dropout, pretrained=False).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    correct, total = 0, 0
    start_eval = time.time()

    print(f"🚀 Evaluating model: {cfg.path_model}")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

    accuracy = correct / total if total > 0 else 0.0
    duration = time.time() - start_eval

    print(f"✅ Evaluation Complete | Accuracy: {accuracy:.4f} | Time: {duration:.2f}s")


if __name__ == "__main__":
    evaluate()
