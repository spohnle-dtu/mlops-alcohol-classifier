from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from omegaconf import DictConfig
import torch
import hydra

from src.alcohol_classifier.data import make_dataloaders
from src.alcohol_classifier.model import BeverageModel
from src.alcohol_classifier.utils import _set_seed, _get_device

def denormalize(tensor: torch.Tensor):
    """Reverses ImageNet normalization for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

@hydra.main(config_path="../../configs", config_name="run", version_base="1.3")
def sample_prediction(cfg: DictConfig) -> None:
    _set_seed(cfg.seed)
    device = _get_device(cfg.device)
    
    # We only need the dataset object and class names here
    _, val_loader, class_names = make_dataloaders(cfg)
    dataset = val_loader.dataset

    # Load Model
    checkpoint = torch.load(cfg.path_model, map_location=device)
    model = BeverageModel(
        num_classes=len(class_names),
        dropout=cfg.model.dropout,
        pretrained=False
    ).to(device)
    
    # Handle both lightning and standard state_dicts
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # SOLUTION: Pick a random index from the WHOLE dataset
    random_idx = torch.randint(0, len(dataset), (1,)).item()
    img_tensor, true_label_idx = dataset[random_idx]
    
    # Prepare image for model (add batch dimension: [C, H, W] -> [1, C, H, W])
    input_tensor = img_tensor.unsqueeze(0).to(device)

    # Run Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        pred_label_idx = torch.argmax(outputs, dim=1).item()

    # Convert back for Matplotlib
    # 1. Denormalize 2. Move to CPU 3. Permute to [H, W, C]
    vis_img = denormalize(img_tensor).cpu().permute(1, 2, 0).numpy()
    vis_img = vis_img.clip(0, 1) # Ensure values stay in valid RGB range

    true_name = class_names[true_label_idx]
    pred_name = class_names[pred_label_idx]

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.imshow(vis_img)
    plt.title(f"True: {true_name} | Pred: {pred_name}", 
              color=("green" if true_name == pred_name else "red"))
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(cm: torch.Tensor, class_names: Sequence[str], out_path: str) -> None:
    """Plot and save a confusion matrix to disk.

    This utility function visualizes a confusion matrix produced during
    evaluation. It is intentionally kept framework-agnostic so it can be
    reused for offline analysis and reporting.

    Args:
        cm: Confusion matrix tensor of shape (C, C), where cm[true, pred]
            corresponds to the number of samples.
        class_names: Class labels in the same order as the confusion matrix axes.
        out_path: File path where the figure should be saved
            (e.g., "reports/figures/confusion_matrix.png").

    Returns:
        None
    """

    # Ensure input is a torch.Tensor for consistent downstream handling
    if not isinstance(cm, torch.Tensor):
        cm = torch.tensor(cm)

    # Detach from computation graph and move to CPU for plotting
    cm = cm.detach().cpu()

    # Convert output path to Path object for safer filesystem handling
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm.numpy(), interpolation="nearest")
    fig.colorbar(im, ax=ax)

    # Configure axis labels and ticks
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate each cell with the corresponding count
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j].item()), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    sample_prediction()