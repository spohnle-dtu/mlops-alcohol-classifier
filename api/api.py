# src/alcohol_classifier/api.py
import io

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile
from PIL import Image

app = FastAPI(title="Alcohol Classifier API (ONNX)")

MODEL_PATH = "models/model.onnx"
ort_session = ort.InferenceSession(MODEL_PATH)

input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

def preprocess_image(image_bytes: bytes):
    """Prepares raw image bytes for the ResNet18 ONNX model."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img = img.resize((224, 224))

    img_data = np.array(img).astype(np.float32) / 255.0

    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)
    return img_data

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_tensor = preprocess_image(image_bytes)

    outputs = ort_session.run([output_name], {input_name: input_tensor})

    logits = outputs[0]
    predicted_class_idx = int(np.argmax(logits))
    confidence = float(np.max(np.exp(logits) / np.sum(np.exp(logits)))) # Softmax

    predicted_class = ""
    if predicted_class_idx == 0:
        predicted_class = "Beer"
    elif predicted_class_idx == 1:
        predicted_class = "Whiskey"
    elif predicted_class_idx == 2:
        predicted_class = "Wine"

    return {
    "predicted_class": predicted_class, # Matches frontend result.get("predicted_class")
    "confidence": round(confidence, 4),
    "probabilities": {str(i): float(logits[0][i]) for i in range(len(logits[0]))} # Optional: helps the bar chart
    }

@app.get("/")
def root():
    return {"message": "Alcohol Classifier API is running on ONNX!"}
