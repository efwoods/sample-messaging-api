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

# Create all required directories
RUN mkdir -p /app/adapters /app/data /app/training_data /app/chroma_db /app/model_cache /app/sample_data

# Copy application files
COPY app.py .

# Copy data files and folders
COPY elon_musk.md .
COPY elon_q_and_a.txt .
COPY elon_speech.txt .
COPY sample_data/ ./sample_data/

# Copy empty directories (they will be created as empty if they don't contain files)
COPY adapters/ ./adapters/
COPY data/ ./data/
COPY training_data/ ./training_data/

# Set up environment variables
ENV PYTHONUNBUFFERED=1
ENV HUGGINGFACE_TOKEN=""
ENV PORT=8080

EXPOSE 8080

# Use non-reload for production
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]