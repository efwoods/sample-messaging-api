# Use a standard Python base image instead of CUDA for Cloud Run
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

# Copy all files at once
COPY . .

# Create/ensure all required directories exist
RUN mkdir -p /app/adapters /app/data /app/training_data /app/chroma_db /app/model_cache /app/sample_data

# Remove unnecessary files that were copied
RUN rm -rf .git/ __pycache__/ .gitignore README.md LICENSE Dockerfile docker-compose.yml .dockerignore .github/ || true

# Set up environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
# Force CPU-only mode
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CPU=1

EXPOSE 8080

# Use non-reload for production
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]