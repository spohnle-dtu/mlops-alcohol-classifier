import importlib
import io
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


def _import_api_with_dummy(monkeypatch, logits: np.ndarray):
    """Monkeypatch onnxruntime.InferenceSession to return a dummy session that produces `logits`.

    Returns the imported api module.
    """
    # Patch the onnxruntime InferenceSession before importing the API module
    import onnxruntime as ort

    def make_dummy_class(logits_arr):
        class DummyIO:
            def __init__(self, name):
                self.name = name

        class DummySession:
            def __init__(self, path):
                # accept the model path parameter but ignore it
                self._logits = logits_arr

            def get_inputs(self):
                # Return a list with an object that has a .name attribute
                return [DummyIO("input")]

            def get_outputs(self):
                return [DummyIO("output")]

            def run(self, outputs, feed):
                return [self._logits]

        return DummySession

    dummy = make_dummy_class(logits)
    monkeypatch.setattr(ort, "InferenceSession", dummy)

    # Ensure a fresh import of the api module
    if "api.api" in sys.modules:
        del sys.modules["api.api"]
    api_module = importlib.import_module("api.api")
    return api_module


def test_read_root(monkeypatch):
    # Use simple logits so import doesn't fail
    logits = np.array([[0.1, 0.2, 0.3]])
    api_module = _import_api_with_dummy(monkeypatch, logits)

    with TestClient(api_module.app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Alcohol Classifier API is running on ONNX!"}


def test_predict_no_image(monkeypatch):
    logits = np.array([[0.1, 0.2, 0.3]])
    api_module = _import_api_with_dummy(monkeypatch, logits)

    with TestClient(api_module.app) as client:
        response = client.post("/predict")
        assert response.status_code == 422  # FastAPI's default for missing data


def test_predict_image_confidence_and_probabilities(monkeypatch):
    # Choose logits where class 1 is highest -> predicted_class should be 'Whiskey'
    logits = np.array([[1.0, 2.0, 0.1]], dtype=np.float32)
    api_module = _import_api_with_dummy(monkeypatch, logits)

    # Create an in-memory JPEG image
    img = Image.new("RGB", (224, 224), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    image_bytes = buf.read()

    with TestClient(api_module.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()

        # Predicted class
        assert data["predicted_class"] == "Whiskey"

        # Confidence should match softmax max (rounded to 4 decimals by API)
        softmax = np.exp(logits) / np.sum(np.exp(logits))
        expected_conf = float(np.max(softmax))
        # API rounds to 4 decimals
        assert round(data["confidence"], 4) == round(expected_conf, 4)

        # Probabilities field in response contains raw logits per the API
        assert "probabilities" in data
        probs = data["probabilities"]
        assert all(k in probs for k in ("0", "1", "2"))
        # Values should match the logits array
        for i in range(3):
            assert pytest.approx(float(logits[0][i]), rel=1e-6) == float(probs[str(i)])


def test_predict_invalid_file_returns_500(monkeypatch):
    # Dummy logits to allow import
    logits = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    api_module = _import_api_with_dummy(monkeypatch, logits)

    # Upload non-image bytes
    bad_bytes = b"this is not an image"

    # Prevent TestClient from re-raising server exceptions so we can assert a 5xx response
    with TestClient(api_module.app, raise_server_exceptions=False) as client:
        response = client.post(
            "/predict",
            files={"file": ("bad.txt", bad_bytes, "text/plain")},
        )
        # PIL.Image.open will raise and FastAPI should return a 500 error
        assert response.status_code >= 400
        assert response.status_code < 600


def test_predict_wrong_field_name_returns_422(monkeypatch):
    # Ensure API imported with dummy session
    logits = np.array([[0.1, 0.2, 0.3]])
    api_module = _import_api_with_dummy(monkeypatch, logits)

    # Create an in-memory JPEG image
    img = Image.new("RGB", (224, 224), color=(0, 255, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    image_bytes = buf.read()

    with TestClient(api_module.app) as client:
        # Use wrong form field name: 'image' instead of 'file'
        response = client.post(
            "/predict",
            files={"image": ("test.png", image_bytes, "image/png")},
        )
        assert response.status_code == 422
