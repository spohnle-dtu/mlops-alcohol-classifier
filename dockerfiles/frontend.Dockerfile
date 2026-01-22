FROM python:3.11-slim

WORKDIR /app

COPY frontend/requirements_frontend.txt /app/requirements_frontend.txt
RUN pip install --no-cache-dir -r /app/requirements_frontend.txt

COPY frontend/frontend.py /app/frontend.py

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "streamlit run /app/frontend.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"]

