FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for compiling or PDF handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose port 8000 for FastAPI + Web UI (Render overrides with $PORT)
EXPOSE 8000

ENV PYTHONUNBUFFERED=1

# Start the full-stack web service (FastAPI + embedded SPA UI)
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
