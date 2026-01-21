# tests/test_api.py
from fastapi.testclient import TestClient
from src.alcohol_classifier.api import app

def test_read_health():
    """Test the health check endpoint while the model is loaded."""
    with TestClient(app) as client: # Ensures startup events (model loading) run
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "model_loaded": True}

def test_predict_no_image():
    """Test that the API correctly rejects a request without a file."""
    with TestClient(app) as client:
        response = client.post("/predict")
        assert response.status_code == 422 # FastAPI's default for missing data
