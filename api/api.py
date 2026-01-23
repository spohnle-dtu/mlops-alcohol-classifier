# # src/alcohol_classifier/api.py
# import io

# import numpy as np
# import onnxruntime as ort
# from fastapi import FastAPI, File, UploadFile
# from PIL import Image

# app = FastAPI(title="Alcohol Classifier API (ONNX)")

# MODEL_PATH = "models/model.onnx"
# ort_session = ort.InferenceSession(MODEL_PATH)

# input_name = ort_session.get_inputs()[0].name
# output_name = ort_session.get_outputs()[0].name


# def preprocess_image(image_bytes: bytes):
#     """Prepares raw image bytes for the ResNet18 ONNX model."""
#     img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

#     img = img.resize((224, 224))

#     img_data = np.array(img).astype(np.float32) / 255.0

#     img_data = np.transpose(img_data, (2, 0, 1))
#     img_data = np.expand_dims(img_data, axis=0)
#     return img_data


# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     image_bytes = await file.read()
#     input_tensor = preprocess_image(image_bytes)

#     outputs = ort_session.run([output_name], {input_name: input_tensor})

#     logits = outputs[0]
#     predicted_class_idx = int(np.argmax(logits))
#     confidence = float(np.max(np.exp(logits) / np.sum(np.exp(logits))))  # Softmax

#     predicted_class = ""
#     if predicted_class_idx == 0:
#         predicted_class = "Beer"
#     elif predicted_class_idx == 1:
#         predicted_class = "Whiskey"
#     elif predicted_class_idx == 2:
#         predicted_class = "Wine"

#     return {
#         "predicted_class": predicted_class,  # Matches frontend result.get("predicted_class")
#         "confidence": round(confidence, 4),
#         "probabilities": {str(i): float(logits[0][i]) for i in range(len(logits[0]))},  # Optional: helps the bar chart
#     }


# @app.get("/")
# def root():
#     return {"message": "Alcohol Classifier API is running on ONNX!"}


# src/alcohol_classifier/api.py (or api/api.py depending on your repo)
import io
import json
import os
import uuid
from datetime import datetime, timezone

import numpy as np
import onnxruntime as ort
import torch
from fastapi import FastAPI, File, UploadFile

# NEW
from google.cloud import storage
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

app = FastAPI(title="Alcohol Classifier API (ONNX + Logging)")

MODEL_PATH = "models/model.onnx"
ort_session = ort.InferenceSession(MODEL_PATH)
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# ===== NEW: logging config =====
GCS_BUCKET = os.environ.get("GCS_BUCKET", "omega-healer-97518-mlops-monitoring")
LOG_PREFIX = os.environ.get("LOG_PREFIX", "inference_logs")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "onnx_v1")

gcs_client = storage.Client()

# ===== NEW: CLIP embedder =====
clip_device = "cpu"  # Cloud Run CPU is fine
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(clip_device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()


def preprocess_image(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_data = np.array(img).astype(np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    return img_data


def compute_clip_embedding(pil_img: Image.Image) -> list[float]:
    """Return CLIP image embedding as a Python list[float]."""
    inputs = clip_processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        feats = clip_model.get_image_features(**inputs)
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)  # normalize
    return feats[0].cpu().numpy().astype(float).tolist()


def upload_log_to_gcs(payload: dict):
    """Upload one JSON log record to GCS."""
    bucket = gcs_client.bucket(GCS_BUCKET)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    obj_name = f"{LOG_PREFIX}/{day}/{payload['request_id']}.json"
    blob = bucket.blob(obj_name)
    blob.upload_from_string(json.dumps(payload), content_type="application/json")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # For embedding we want a PIL image too
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ONNX prediction (your existing code)
    input_tensor = preprocess_image(image_bytes)
    outputs = ort_session.run([output_name], {input_name: input_tensor})
    logits = outputs[0]
    predicted_class_idx = int(np.argmax(logits))
    softmax = np.exp(logits) / np.sum(np.exp(logits))
    confidence = float(np.max(softmax))

    if predicted_class_idx == 0:
        predicted_class = "Beer"
    elif predicted_class_idx == 1:
        predicted_class = "Whiskey"
    else:
        predicted_class = "Wine"

    # NEW: embedding + logging
    emb = compute_clip_embedding(pil_img)
    request_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    log_payload = {
        "request_id": request_id,
        "timestamp": ts,
        "model_version": MODEL_VERSION,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 6),
        "logits": [float(x) for x in logits[0].tolist()],
        "embedding": emb,
    }

    # Upload log (best effort)
    try:
        upload_log_to_gcs(log_payload)
    except Exception as e:
        # Don't fail the prediction just because logging failed
        log_payload["logging_error"] = str(e)

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": {str(i): float(logits[0][i]) for i in range(len(logits[0]))},
        "request_id": request_id,
    }


@app.get("/")
def root():
    return {"message": "Alcohol Classifier API is running (ONNX + logging)!"}
