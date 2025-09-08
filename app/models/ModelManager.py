from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import PeftModel, LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset
import chromadb
from sentence_transformers import SentenceTransformer
import os
import uuid
import json
import time
import logging
import asyncio
import shutil
from typing import Optional, List, Dict, Union, Any
from contextlib import asynccontextmanager
from threading import Lock
from datetime import datetime
import psutil
import gc
import numpy as np
from pathlib import Path

# Enhanced Model Manager with memory optimization and caching
class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_adapter = None
        self.current_avatar = "base"
        self.model_lock = Lock()
        self._model_name = "meta-llama/Llama-3.2-1B-Instruct"
        self._adapters_dir = "./adapters"
        self._training_data_dir = "./training_data"
        self._cache_dir = "./model_cache"
        self.adapter_cache = {}
        self.training_sessions = {}
        
    def get_memory_usage(self) -> Dict:
        """Get detailed memory usage statistics"""
        if torch.cuda.is_available():
            return {
                # "gpu_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                # "gpu_cached_gb": torch.cuda.memory_reserved() / 1024**3,
                # "gpu_max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
                "system_memory_gb": psutil.virtual_memory().used / 1024**3,
                "system_memory_percent": psutil.virtual_memory().percent
            }
        return {
            "system_memory_gb": psutil.virtual_memory().used / 1024**3,
            "system_memory_percent": psutil.virtual_memory().percent
        }
    
    def optimize_memory(self):
        """Perform memory optimization"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    def load_base_model(self):
        """Load base model with enhanced error handling"""
        with self.model_lock:
            try:
                # Clear existing model
                if self.model is not None:
                    del self.model
                    self.optimize_memory()
                
                # bnb_config = BitsAndBytesConfig(
                #     load_in_4bit=True,
                #     bnb_4bit_quant_type="nf4",
                #     bnb_4bit_compute_dtype=torch.bfloat16,
                #     bnb_4bit_use_double_quant=True
                # )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self._model_name,
                    # quantization_config=bnb_config,
                    device_map="cpu",
                    trust_remote_code=True,
                    token=os.getenv("HUGGINGFACE_TOKEN"),
                    torch_dtype=torch.float16,
                    cache_dir=self._cache_dir
                )
                
                if self.tokenizer is None:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self._model_name, 
                        trust_remote_code=True, 
                        token=os.getenv("HUGGINGFACE_TOKEN"),
                        cache_dir=self._cache_dir
                    )
                    
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.current_adapter = None
                logger.info("Base model loaded successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load base model: {str(e)}")
                raise RuntimeError(f"Failed to load base model: {str(e)}")
    
    def load_adapter(self, avatar: str) -> bool:
        """Load adapter with caching and improved error handling"""
        with self.model_lock:
            if self.current_adapter == avatar:
                return True
                
            # Check cache first
            if avatar in self.adapter_cache:
                logger.info(f"Loading {avatar} from cache")
                self.model = self.adapter_cache[avatar]
                self.current_adapter = avatar
                return True
            
            if self.current_adapter is not None:
                self.load_base_model()
            
            adapter_path = os.path.join(self._adapters_dir, avatar)
            
            if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
                try:
                    self.model = PeftModel.from_pretrained(self.model, adapter_path)
                    self.current_adapter = avatar
                    logger.info(f"QLoRA adapter for {avatar} loaded successfully")
                    return True
                except Exception as e:
                    logger.error(f"Failed to load adapter for {avatar}: {str(e)}")
                    self.load_base_model()
                    return False
            else:
                logger.info(f"No adapter found for {avatar}, using base model")
                return False

    def get_available_avatars(self) -> List[Dict]:
        """Get list of available avatars with metadata"""
        avatars = []
        
        # Add base avatar
        avatars.append({
            "name": "base",
            "has_adapter": False,
            "has_training_data": False,
            "created_at": None,
            "size_mb": 0
        })
        
        # Scan adapter directory
        if os.path.exists(self._adapters_dir):
            for avatar_dir in os.listdir(self._adapters_dir):
                adapter_path = os.path.join(self._adapters_dir, avatar_dir)
                if os.path.isdir(adapter_path):
                    config_path = os.path.join(adapter_path, "adapter_config.json")
                    has_adapter = os.path.exists(config_path)
                    
                    training_data_path = os.path.join(self._training_data_dir, avatar_dir, "dataset.jsonl")
                    has_training_data = os.path.exists(training_data_path)
                    
                    created_at = None
                    size_mb = 0
                    if has_adapter:
                        created_at = os.path.getctime(config_path)
                        size_mb = sum(os.path.getsize(os.path.join(adapter_path, f)) 
                                    for f in os.listdir(adapter_path)) / 1024 / 1024
                    
                    avatars.append({
                        "name": avatar_dir,
                        "has_adapter": has_adapter,
                        "has_training_data": has_training_data,
                        "created_at": created_at,
                        "size_mb": round(size_mb, 2)
                    })
        
        return avatars


    def ensure_base_model_loaded(self):
        """Ensure base model is loaded, load if not"""
        if self.model is None or self.tokenizer is None:
            logger.info("Base model not loaded, loading now...")
            self.load_base_model()
            return True
        return False
    
    def create_empty_adapter(self, avatar: str) -> bool:
        """Create an empty adapter for avatar without training"""
        try:
            adapter_path = os.path.join(self._adapters_dir, avatar)
            os.makedirs(adapter_path, exist_ok=True)
            
            # Ensure base model is loaded
            if self.current_adapter is not None:
                self.load_base_model()
            
            # Create LoRA configuration with default parameters
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=0.1,
                task_type="CAUSAL_LM"
            )
            
            with self.model_lock:
                # Apply PEFT model
                self.model = get_peft_model(self.model, lora_config)
                
                # Save the adapter (even though untrained)
                self.model.save_pretrained(adapter_path)
                self.tokenizer.save_pretrained(adapter_path)
                
                # Load the adapter back to ensure it's properly attached
                self.model = PeftModel.from_pretrained(self.model.base_model, adapter_path)
                self.current_adapter = avatar
            
            logger.info(f"Empty adapter created and attached for {avatar}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create empty adapter for {avatar}: {str(e)}")
            # Fallback to base model
            self.load_base_model()
            return False
    
    def attach_existing_adapter(self, avatar: str) -> bool:
        """Attach existing adapter to base model"""
        try:
            adapter_path = os.path.join(self._adapters_dir, avatar)
            
            # Ensure we start with base model
            if self.current_adapter is not None:
                self.load_base_model()
            
            with self.model_lock:
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
                self.current_adapter = avatar
            
            logger.info(f"Existing adapter attached for {avatar}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to attach existing adapter for {avatar}: {str(e)}")
            return False

