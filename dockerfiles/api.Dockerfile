FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Code
COPY src /app/src
COPY configs /app/configs
ENV PYTHONPATH=/app/src

# Model artifact
COPY models /app/models
ENV MODEL_PATH=/app/models/model.pt

# Cloud Run
ENV PORT=8080

# IMPORTANT: change module path if your package isn't alcohol_classifier
CMD ["bash", "-lc", "python -m uvicorn alcohol_classifier.api:app --host 0.0.0.0 --port ${PORT}"]

