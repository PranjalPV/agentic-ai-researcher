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

# Expose ports: 8501 for Streamlit UI, 8000 for FastAPI
EXPOSE 8501
EXPOSE 8000

ENV PYTHONUNBUFFERED=1

# Default command starts the Streamlit researcher dashboard
CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
