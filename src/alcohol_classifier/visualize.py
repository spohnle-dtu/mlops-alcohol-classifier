from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch


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
