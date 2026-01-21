FROM python:3.12-slim

WORKDIR /app

# (Optional but nice) avoid pyc + ensure logs flush
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy code
COPY src /app/src
COPY configs /app/configs
ENV PYTHONPATH=/app/src

# Copy model artifact (tracked via DVC, but present in repo as checkpoints/best.pt)
COPY checkpoints /app/checkpoints
ENV MODEL_PATH=/app/checkpoints/best.pt

# Cloud Run listens on $PORT (provided at runtime). Fallback for local runs.
CMD ["bash", "-lc", "python -m uvicorn alcohol_classifier.api:app --host 0.0.0.0 --port ${PORT:-8080}"]