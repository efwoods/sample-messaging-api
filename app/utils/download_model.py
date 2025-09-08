#!/usr/bin/env python3
"""
Script to download and cache the Llama model during Docker build.
This ensures the model is baked into the Docker image.
"""

import os
import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch

def download_model():
    """Download and cache the model to the specified directory."""
    
    # Model configuration
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model_dir = Path("/app/models/llama-3.2-1b-instruct")
    
    print(f"Starting download of {model_name}...")
    print(f"Cache directory: {model_dir}")
    
    # Create directory if it doesn't exist
    model_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Login to Hugging Face if token is provided
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            print("Logging into Hugging Face Hub...")
            login(token=hf_token)
        else:
            print("No Hugging Face token provided. Attempting to download public model...")
        
        # Download tokenizer
        print("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(model_dir),
            token=hf_token if hf_token else None
        )
        
        # Save tokenizer to model directory
        tokenizer.save_pretrained(str(model_dir))
        print(f"Tokenizer saved to {model_dir}")
        
        # Download model
        print("Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=str(model_dir),
            torch_dtype=torch.float16,
            token=hf_token if hf_token else None,
            # Don't load to device during download to save memory
            device_map=None
        )
        
        # Save model to model directory
        model.save_pretrained(str(model_dir))
        print(f"Model saved to {model_dir}")
        
        # Verify the download
        required_files = [
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json"
        ]
        
        missing_files = []
        for file_name in required_files:
            if not (model_dir / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            print(f"Warning: Missing files: {missing_files}")
            return False
        
        print("✅ Model download completed successfully!")
        print(f"Model size: {sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / 1024**3:.2f} GB")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

if __name__ == "__main__":
    success = download_model()
    if not success:
        sys.exit(1)
    print("Model download script completed successfully.")