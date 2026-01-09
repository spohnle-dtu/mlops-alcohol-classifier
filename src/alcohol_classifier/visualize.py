from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import matplotlib.pyplot as plt
import torch


def plot_confusion_matrix(cm: torch.Tensor, class_names: Sequence[str], out_path: str) -> None:
    """
    Save a confusion matrix plot to disk.

    Args:
        cm: (C, C) tensor where cm[true, pred] is the count
        class_names: list of class names in the same order as labels
        out_path: file path to save (e.g., reports/confusion_matrix.png)
    """
    if not isinstance(cm, torch.Tensor):
        cm = torch.tensor(cm)

    cm = cm.detach().cpu()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm.numpy(), interpolation="nearest")
    fig.colorbar(im, ax=ax)

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

    # annotate counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j].item()), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
