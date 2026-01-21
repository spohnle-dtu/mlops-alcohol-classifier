from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from torchvision import transforms

from .model import BeverageModel

app = FastAPI(title="BeverageModel API")

# ---- config ----
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.pt"))
DEVICE = "cpu"  # Cloud Run default; you can switch later if needed

# ---- preprocessing (matches ResNet18 expectations) ----
_preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

_model: BeverageModel | None = None
_class_names: list[str] | None = None


def _load_checkpoint(path: Path) -> tuple[BeverageModel, list[str]]:
    if not path.exists():
        # helpful fallback: pick first checkpoint if user didn’t set MODEL_PATH
        candidates = list(Path("models").glob("*.pt")) + list(Path("models").glob("*.pth"))
        if candidates:
            path = candidates[0]
        else:
            raise FileNotFoundError(
                f"Model checkpoint not found at {path}. "
                f"Set MODEL_PATH env var or put a .pt/.pth file in models/."
            )

    ckpt: dict[str, Any] = torch.load(path, map_location="cpu")
    class_names = ckpt.get("class_names", ["class0", "class1", "class2"])

    model = BeverageModel(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, list(class_names)


@app.on_event("startup")
def startup():
    global _model, _class_names
    _model, _class_names = _load_checkpoint(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if _model is None or _class_names is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if image.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="Upload a JPEG/PNG/WEBP image")

    img_bytes = await image.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    x = _preprocess(img).unsqueeze(0)  # (1, 3, 224, 224)

    with torch.no_grad():
        logits = _model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0).tolist()

    pred_idx = int(torch.tensor(probs).argmax().item())
    return {
        "predicted_class": _class_names[pred_idx],
        "probabilities": {name: float(p) for name, p in zip(_class_names, probs)},
    }

