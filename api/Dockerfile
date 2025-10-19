# api/Dockerfile
FROM python:3.11-slim

# System deps (optional but useful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . /app

# Expose port for the service
EXPOSE 8000

# Environment (override at run/deploy time)
# ENV N8N_WEBHOOK_URL="https://<your-n8n-host>/webhook-test/<path>"
# ENV ALLOWED_ORIGINS="*"
# ENV REQUEST_TIMEOUT="60"

# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]