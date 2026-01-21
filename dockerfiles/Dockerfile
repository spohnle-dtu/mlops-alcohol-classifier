FROM python:3.12-slim

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Code + config
COPY src /app/src
COPY configs /app/configs
COPY pyproject.toml /app/pyproject.toml

# Model artifact
COPY checkpoints /app/checkpoints

# Fail fast at build-time if the checkpoint isn't in the build context
#RUN ls -lah /app/checkpoints && test -f /app/checkpoints/best.pt

ENV PYTHONPATH=/app/src
ENV HYDRA_FULL_ERROR=1
ENV MODEL_PATH=/app/checkpoints/best.pt

# Cloud Run expects the container to listen on $PORT
ENV PORT=8080
EXPOSE 8080

CMD ["bash", "-lc", "python -m uvicorn alcohol_classifier.api:app --host 0.0.0.0 --port ${PORT:-8080}"]