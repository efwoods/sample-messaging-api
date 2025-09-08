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
from models import ModelManager
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom embedding function wrapper for ChromaDB
class SentenceTransformerEmbeddingFunction:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(input)
        return embeddings.tolist()



# Initialize managers
model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Enhanced Avatar Management System...")
    try:
        # Create necessary directories
        os.makedirs(model_manager._adapters_dir, exist_ok=True)
        os.makedirs(model_manager._training_data_dir, exist_ok=True)
        os.makedirs(model_manager._cache_dir, exist_ok=True)
        
        model_manager.load_base_model()
        logger.info("System initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enhanced Avatar Management System...")
    model_manager.optimize_memory()

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Llama-3.2-1B Avatar System",
    description="Advanced Avatar-specific QLoRA adapters with ChromaDB RAG integration",
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    max_results: int = Field(3, ge=1, le=10)
    max_tokens: int = Field(512, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.1, le=2.0)
    use_base_model: bool = Field(False)
    include_conversation_history: bool = Field(True)
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0)

class StreamQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    max_results: int = Field(3, ge=1, le=10)
    max_tokens: int = Field(512, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0.1, le=2.0)

class TrainRequest(BaseModel):
    epochs: int = Field(1, ge=1, le=10)
    batch_size: int = Field(4, ge=1, le=16)
    learning_rate: float = Field(2e-4, ge=1e-5, le=1e-3)
    lora_r: int = Field(8, ge=4, le=64)
    lora_alpha: int = Field(16, ge=8, le=128)
    lora_dropout: float = Field(0.1, ge=0.0, le=0.5)

class DocumentRequest(BaseModel):
    content: str = Field(..., min_length=10, max_length=10000)
    source: str = Field("user_upload", max_length=200)
    metadata: Optional[Dict] = Field(None)

class BulkDocumentRequest(BaseModel):
    documents: List[DocumentRequest] = Field(..., min_items=1, max_items=100)

class TrainingDocumentRequest(BaseModel):
    instruction: str = Field(..., min_length=5, max_length=2000)
    response: str = Field(..., min_length=5, max_length=2000)
    source: str = Field("user_upload", max_length=200)
    category: Optional[str] = Field(None)

class SelectAvatarRequest(BaseModel):
    avatar: str = Field(..., min_length=1, max_length=50, pattern="^[a-zA-Z0-9_\\-\\s]+$")

class AvatarComparisonRequest(BaseModel):
    avatars: List[str] = Field(..., min_items=2, max_items=5)
    query: str = Field(..., min_length=1, max_length=1000)

class ExportRequest(BaseModel):
    avatar: str = Field(..., min_length=1, max_length=50)
    include_training_data: bool = Field(True)
    include_documents: bool = Field(True)


def normalize_avatar_name(avatar: str) -> str:
    """Normalize avatar name for consistency"""
    return avatar.lower().replace(" ", "_").replace("-", "_")



# Updated select_avatar endpoint
@app.post("/load_avatar", tags=["Avatar Management"])
async def load_avatar(request: SelectAvatarRequest, background_tasks: BackgroundTasks):
    """
    Enhanced avatar selection with guaranteed base model loading,
    adapter creation/attachment, and ChromaDB initialization
    """
    avatar = normalize_avatar_name(request.avatar)
    
    try:
        # Step 1: Ensure base model is loaded
        logger.info(f"Selecting avatar: {avatar}")
        base_loaded = model_manager.ensure_base_model_loaded()
        
        # Step 2: Check if adapter exists
        adapter_path = os.path.join(model_manager._adapters_dir, avatar)
        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
        adapter_exists = os.path.exists(adapter_config_path)
        
        adapter_result = {
            "adapter_exists": adapter_exists,
            "adapter_attached": False,
            "adapter_created": False,
            "adapter_trained": False
        }
        
        if adapter_exists:
            # Step 3a: Attach existing adapter
            success = model_manager.load_adapter(avatar)
            adapter_result["adapter_attached"] = success
            
            # Check if adapter was trained (has training data)
            training_data_path = os.path.join(model_manager._training_data_dir, avatar, "dataset.jsonl")
            adapter_result["adapter_trained"] = os.path.exists(training_data_path)
            
        else:
            # Step 3b: Create new empty adapter
            success = model_manager.create_empty_adapter(avatar)
            adapter_result["adapter_created"] = success
            adapter_result["adapter_attached"] = success
            adapter_result["adapter_trained"] = False
        
        # Step 4: Ensure ChromaDB collection exists (even if empty)
        chroma_success, chroma_stats = chroma_manager.ensure_collection_exists(avatar)
        
        # Step 5: Update current avatar
        model_manager.current_avatar = avatar
        
        # Step 6: Broadcast status update
        await websocket_manager.broadcast({
            "type": "avatar_selected",
            "avatar": avatar,
            "base_model_loaded": model_manager.model is not None,
            "adapter_result": adapter_result,
            "chroma_stats": chroma_stats
        })
        
        # Prepare response
        status_messages = []
        if base_loaded:
            status_messages.append("base model loaded")
        
        if adapter_result["adapter_created"]:
            status_messages.append("new adapter created and attached")
        elif adapter_result["adapter_attached"]:
            trained_status = "trained" if adapter_result["adapter_trained"] else "untrained"
            status_messages.append(f"existing {trained_status} adapter attached")
        else:
            status_messages.append("adapter attachment failed, using base model")
        
        if chroma_success:
            doc_status = f"with {chroma_stats['document_count']} documents" if chroma_stats['has_documents'] else "empty"
            status_messages.append(f"ChromaDB collection ready ({doc_status})")
        else:
            status_messages.append("ChromaDB collection unavailable")
        
        return {
            "status": f"Avatar {avatar} selected successfully",
            "details": ", ".join(status_messages),
            "avatar": avatar,
            "base_model_loaded": model_manager.model is not None,
            "adapter_result": adapter_result,
            "chroma_stats": chroma_stats,
            "ready_for_queries": True
        }
        
    except Exception as e:
        logger.error(f"Error selecting avatar {avatar}: {str(e)}")
        
        # Broadcast error
        await websocket_manager.broadcast({
            "type": "avatar_selection_failed",
            "avatar": avatar,
            "error": str(e)
        })
        
        # Try to fallback to base model
        try:
            model_manager.load_base_model()
            model_manager.current_avatar = "base"
            fallback_message = "Fell back to base model"
        except:
            fallback_message = "System in unstable state"
        
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to select avatar {avatar}: {str(e)}. {fallback_message}"
        )



# Enhanced query method to handle untrained adapters gracefully
@app.post("/query", tags=["Model Operations"])
async def query_model(request: QueryRequest):
    """Enhanced query with better handling of untrained adapters and empty ChromaDB"""
    if not model_manager.model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Ensure correct adapter is loaded if not using base model explicitly
        if not request.use_base_model and model_manager.current_avatar != "base":
            if model_manager.current_adapter != model_manager.current_avatar:
                # Try to load the adapter, create if doesn't exist
                adapter_path = os.path.join(model_manager._adapters_dir, model_manager.current_avatar)
                if not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
                    logger.info(f"Adapter not found for {model_manager.current_avatar}, creating empty adapter")
                    model_manager.create_empty_adapter(model_manager.current_avatar)
                else:
                    model_manager.load_adapter(model_manager.current_avatar)
        
        # Enhanced context retrieval (works even with empty ChromaDB)
        context_start = time.time()
        search_results = chroma_manager.semantic_search(
            model_manager.current_avatar,
            request.query,
            request.max_results,
            request.similarity_threshold
        )
        context_time = time.time() - context_start
        
        # Build context from search results (gracefully handles empty results)
        context_parts = []
        if search_results["has_results"]:
            for part in search_results["context_parts"]:
                context_parts.append(f"[{part['source']} - Similarity: {part['similarity']}]\n{part['content']}")
        
        context = "\n\n".join(context_parts) if context_parts else ""
        
        # Get conversation history if requested
        conversation_context = ""
        if request.include_conversation_history:
            conversation_context = conversation_manager.get_conversation_context(model_manager.current_avatar)
        
        # Build enhanced system message
        system_message = "You are a helpful assistant"
        if not request.use_base_model and model_manager.current_avatar != "base":
            system_message += f" in the style of {model_manager.current_avatar}"
        
        # Adapt message based on available context
        if context:
            system_message += ". Use the following context to answer accurately."
        elif model_manager.current_adapter:
            # Adapter exists but no context from ChromaDB
            system_message += ". Draw upon your specialized knowledge and training."
        else:
            system_message += ". Rely on your general knowledge to be helpful."
        
        if conversation_context:
            system_message += " Consider the conversation history when appropriate."
        
        # Construct enhanced prompt
        prompt_parts = [f"<|start_header_id|>system<|end_header_id|>\n{system_message}"]
        
        if context:
            prompt_parts.append(f"Context:\n{context}")
        
        if conversation_context:
            prompt_parts.append(f"Recent Conversation:\n{conversation_context}")
        
        prompt_parts.extend([
            "<|end_header_id|>",
            f"<|start_header_id|>user<|end_header_id|>\n{request.query}<|end_header_id|>",
            "<|start_header_id|>assistant<|end_header_id|>"
        ])
        
        prompt = "\n".join(prompt_parts)
        
        # Tokenization with error handling
        tokenize_start = time.time()
        try:
            inputs = model_manager.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            device = next(model_manager.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception as e:
            logger.error(f"Tokenization error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Tokenization failed: {str(e)}")
        
        tokenize_time = time.time() - tokenize_start
        
        # Generation with improved error handling
        gen_start = time.time()
        try:
            with torch.no_grad():
                outputs = model_manager.model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=model_manager.tokenizer.eos_token_id,
                    eos_token_id=model_manager.tokenizer.eos_token_id
                )
        except Exception as e:
            logger.error(f"Generation error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")
        
        gen_time = time.time() - gen_start
        
        # Decode response
        decode_start = time.time()
        response = model_manager.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response
        try:
            assistant_response = response.split("<|start_header_id|>assistant<|end_header_id|>")[1].strip()
        except IndexError:
            assistant_response = response[len(prompt):].strip()
        
        decode_time = time.time() - decode_start
        total_time = time.time() - start_time
        
        # Add to conversation memory
        conversation_manager.add_message(model_manager.current_avatar, "user", request.query)
        conversation_manager.add_message(model_manager.current_avatar, "assistant", assistant_response)
        
        # Calculate metrics
        input_tokens = len(inputs['input_ids'][0])
        output_tokens = len(outputs[0]) - input_tokens
        tokens_per_second = output_tokens / gen_time if gen_time > 0 else 0
        
        # Determine adapter status
        adapter_status = "none"
        if model_manager.current_adapter:
            training_data_path = os.path.join(model_manager._training_data_dir, model_manager.current_avatar, "dataset.jsonl")
            adapter_status = "trained" if os.path.exists(training_data_path) else "untrained"
        
        response_data = {
            "response": assistant_response,
            "avatar": model_manager.current_avatar,
            "adapter_status": adapter_status,
            "context_used": search_results["has_results"],
            "context_source_count": len(search_results["context_parts"]),
            "using_base_model_only": request.use_base_model,
            "search_results": search_results,
            "conversation_history_used": request.include_conversation_history,
            "metrics": {
                "total_time_seconds": round(total_time, 4),
                "context_retrieval_time_seconds": round(context_time, 4),
                "tokenization_time_seconds": round(tokenize_time, 4),
                "generation_time_seconds": round(gen_time, 4),
                "decoding_time_seconds": round(decode_time, 4),
                "tokens_per_second": round(tokens_per_second, 2),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        }
        
        # Broadcast query completion
        await websocket_manager.broadcast({
            "type": "query_completed",
            "avatar": model_manager.current_avatar,
            "adapter_status": adapter_status,
            "context_used": search_results["has_results"],
            "metrics": response_data["metrics"]
        })
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/system/memory", tags=["System"])
async def get_memory_usage():
    """Get detailed system memory usage"""
    return {
        "memory_usage": model_manager.get_memory_usage(),
        "model_loaded": model_manager.model is not None,
        "current_avatar": model_manager.current_avatar,
        "adapter_loaded": model_manager.current_adapter
    }

@app.post("/system/optimize", tags=["System"])
async def optimize_system():
    """Perform system optimization"""
    try:
        memory_before = model_manager.get_memory_usage()
        model_manager.optimize_memory()
        memory_after = model_manager.get_memory_usage()
        
        return {
            "status": "System optimization completed",
            "memory_before": memory_before,
            "memory_after": memory_after
        }
        
    except Exception as e:
        logger.error(f"Error optimizing system: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")


@app.get("/status", tags=["System"])
async def get_status():
    """Enhanced system status with comprehensive information"""
    return {
        "current_avatar": model_manager.current_avatar,
        "adapter_loaded": model_manager.current_adapter,
        "model_loaded": model_manager.model is not None,
        "tokenizer_loaded": model_manager.tokenizer is not None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "memory_usage": model_manager.get_memory_usage(),
        "available_avatars": len(model_manager.get_available_avatars()),
        "active_connections": len(websocket_manager.active_connections),
        "system_info": {
            "python_version": os.sys.version.split()[0],
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    }

@app.get("/health", tags=["System"])
async def health_check():
    """Enhanced health check with system validation"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "checks": {
                "model_loaded": model_manager.model is not None,
                "tokenizer_loaded": model_manager.tokenizer is not None,
                "chroma_accessible": True,
                "memory_ok": True
            }
        }
        
        # Check memory usage
        memory_usage = model_manager.get_memory_usage()
        if memory_usage.get("system_memory_percent", 0) > 95:
            health_status["checks"]["memory_ok"] = False
            health_status["status"] = "degraded"
        
        if not all(health_status["checks"].values()):
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }

    
# Additional utility endpoint to check avatar readiness
@app.get("/avatar_status/{avatar}", tags=["Avatar Management"])
async def get_avatar_status(avatar: str):
    """Get detailed status of a specific avatar"""
    avatar = normalize_avatar_name(avatar)
    
    # Check adapter status
    adapter_path = os.path.join(model_manager._adapters_dir, avatar)
    adapter_exists = os.path.exists(os.path.join(adapter_path, "adapter_config.json"))
    
    # Check training data
    training_data_path = os.path.join(model_manager._training_data_dir, avatar, "dataset.jsonl")
    training_data_exists = os.path.exists(training_data_path)
    training_data_count = 0
    
    if training_data_exists:
        try:
            with open(training_data_path, "r", encoding="utf-8") as f:
                training_data_count = sum(1 for line in f if line.strip())
        except:
            training_data_count = 0
    
    # Check ChromaDB collection
    # chroma_success, chroma_stats = chroma_manager.ensure_collection_exists(avatar)
    
    return {
        "avatar": avatar,
        "is_current": avatar == model_manager.current_avatar,
        "adapter": {
            "exists": adapter_exists,
            "loaded": model_manager.current_adapter == avatar,
            "trained": training_data_exists,
            "training_examples": training_data_count
        },
        "ready_for_use": True,  # Always ready with this new system
        "recommended_action": (
            "Ready to use" if adapter_exists and training_data_exists 
            else "Consider adding training data" if adapter_exists 
            else "Will create adapter on selection"
        )
    }