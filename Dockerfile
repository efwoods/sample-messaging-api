FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# Copy requirements first for better layer caching
COPY requirements.txt .
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

EXPOSE 8080

# Use non-reload for production
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]