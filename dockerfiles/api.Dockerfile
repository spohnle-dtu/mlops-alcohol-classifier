FROM python:3.12-slim

WORKDIR /app

# Prevent python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --- FIX 1: Use the slim requirements ---
COPY requirements_docker.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# --- FIX 2: Copy the correct folders ---
COPY src /app/src
COPY api /app/api
COPY configs /app/configs
COPY models /app/models

# --- FIX 3: Set paths so Python can find 'api' module ---
ENV PYTHONPATH=/app/src:/app
ENV MODEL_PATH=/app/models/model.onnx

# --- FIX 4: Correct the CMD module path ---
# Since api.py is inside the api/ folder
CMD ["python", "-m", "uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8080"]
