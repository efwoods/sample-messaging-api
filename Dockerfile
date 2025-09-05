# Dockerfile.cpu - CPU-only version for local production-like environment
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install CPU-only versions of packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .

# Instead, ensure they exist at build time
RUN mkdir -p /app/adapters /app/data /app/training_data /app/chroma_db /app/model_cache /app/sample_data


# Set up environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
# Force CPU-only mode
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CPU=1

EXPOSE 8080

# Use non-reload for production-like environment
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]