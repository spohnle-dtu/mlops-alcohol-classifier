from typing import Dict, List

import hydra
import torch
import torchvision.transforms as T
from loguru import logger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.alcohol_classifier.data import make_dataloaders
from src.alcohol_classifier.model import BeverageModel
from src.alcohol_classifier.utils import _get_device, _set_seed


def evaluate_with_transform(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    transform: T.Compose | None = None,
) -> float:
    """Evaluate accuracy with an optional image transform applied."""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in loader:
            if transform is not None:
                images = torch.stack([transform(img) for img in images])

            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

    return correct / total if total > 0 else 0.0


def brightness_transform(severity: int):
    return T.Compose(
        [
            T.Lambda(lambda x: x + severity * 0.1),
            T.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        ]
    )


def noise_transform(severity: int):
    return T.Compose(
        [
            T.Lambda(lambda x: x + torch.randn_like(x) * severity * 0.05),
            T.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        ]
    )


def blur_transform(severity: int):
    return T.GaussianBlur(kernel_size=3, sigma=severity * 0.5)


@hydra.main(config_path="../../configs", config_name="run", version_base="1.3")
def run_drift_robustness(cfg: DictConfig):
    logger.info("🔍 Running drift robustness evaluation")

    _set_seed(cfg.seed)
    device = _get_device(cfg.device)

    # Load data
    _, val_loader, class_names = make_dataloaders(cfg)

    # Load model
    checkpoint = torch.load(cfg.path_model, map_location=device)
    model = BeverageModel(
        num_classes=len(class_names),
        pretrained=False,
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    results: Dict[str, List[float]] = {}

    # Baseline (no drift)
    baseline_acc = evaluate_with_transform(model, val_loader, device)
    results["clean"] = [baseline_acc]
    logger.info(f"Clean accuracy: {baseline_acc:.4f}")

    # Drift experiments
    severities = [1, 2, 3, 4, 5]
    drift_types = {
        "brightness": brightness_transform,
        "noise": noise_transform,
        "blur": blur_transform,
    }

    for drift_name, drift_fn in drift_types.items():
        accs = []
        for s in severities:
            acc = evaluate_with_transform(
                model,
                val_loader,
                device,
                transform=drift_fn(s),
            )
            accs.append(acc)
            logger.info(f"{drift_name} (severity={s}) → acc={acc:.4f}")
        results[drift_name] = accs

    logger.info("✅ Drift robustness evaluation complete")
    logger.info(results)


if __name__ == "__main__":
    run_drift_robustness()
