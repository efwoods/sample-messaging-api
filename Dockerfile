FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create all required directories
RUN mkdir -p /app/adapters /app/data /app/training_data /app/chroma_db

# Copy the entire context and then move/organize files
COPY . /tmp/build_context/

# Conditionally copy directories if they exist in the build context
RUN if [ -d "/tmp/build_context/adapters" ]; then cp -r /tmp/build_context/adapters/* /app/adapters/ 2>/dev/null || true; fi
RUN if [ -d "/tmp/build_context/data" ]; then cp -r /tmp/build_context/data/* /app/data/ 2>/dev/null || true; fi  
RUN if [ -d "/tmp/build_context/training_data" ]; then cp -r /tmp/build_context/training_data/* /app/training_data/ 2>/dev/null || true; fi

# Clean up temporary build context
RUN rm -rf /tmp/build_context

# Uncomment when you have app.py
# COPY app.py .

ENV PYTHONUNBUFFERED=1
ENV HUGGINGFACE_TOKEN=""
ENV PORT=8080

EXPOSE 8080

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]