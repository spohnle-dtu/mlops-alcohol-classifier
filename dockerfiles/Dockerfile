FROM python:3.12-slim

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY configs /app/configs
COPY pyproject.toml /app/pyproject.toml
COPY checkpoints /app/checkpoints

ENV PYTHONPATH=/app/src
ENV HYDRA_FULL_ERROR=1
ENV MODEL_PATH=/app/checkpoints/best.pt


CMD ["python", "-c", "print('Container ready. Example: python -m alcohol_classifier.train')"]
