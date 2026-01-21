import time
import torch
import hydra
from omegaconf import DictConfig

from loguru import logger

from src.alcohol_classifier.data import make_dataloaders
from src.alcohol_classifier.model import BeverageModel

def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _get_device(device: str) -> torch.device:
    if device != "auto":                    return torch.device(device)
    if torch.backends.mps.is_available():   return torch.device("mps")
    if torch.cuda.is_available():           return torch.device("cuda")
    
    return torch.device("cpu")

@hydra.main(config_path="../../configs", config_name="run", version_base="1.3")
def evaluate(cfg: DictConfig) -> None:
    logger.info("Evaluation started")

    _set_seed(cfg.dataset.seed)
    device = _get_device(cfg.device)
    
    _, val_loader, class_names = make_dataloaders(cfg)

    checkpoint = torch.load(cfg.path_model, map_location=device)
    
    model = BeverageModel(
        num_classes=len(class_names),
        dropout=cfg.model.dropout,
        pretrained=False
    ).to(device)
    
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    correct, total = 0, 0
    start_eval = time.time()

    logger.info(f"🚀 Evaluating model: {cfg.path_model}")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

    accuracy = correct / total if total > 0 else 0.0
    duration = time.time() - start_eval

    logger.info(f"✅ Evaluation Complete | Accuracy: {accuracy:.4f} | Time: {duration:.2f}s")

if __name__ == "__main__":
    evaluate()