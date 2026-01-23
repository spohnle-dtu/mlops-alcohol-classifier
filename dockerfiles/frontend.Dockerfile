FROM python:3.11-slim

WORKDIR /app

# --- FIX 1: Point to the root for the requirements file ---
COPY requirements_docker.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# --- FIX 2: Copy from your 'api' folder where frontend.py lives ---
COPY api/frontend.py /app/frontend.py

# Cloud Run / Container settings
ENV PORT=8501
EXPOSE 8501

# --- FIX 3: Simplified CMD ---
CMD ["sh", "-c", "streamlit run /app/frontend.py --server.port=$PORT --server.address=0.0.0.0"]
