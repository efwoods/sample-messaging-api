from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from core.config import settings
from core.logging import logger
from core.monitoring import metrics
from core.s3_instance import init_s3_session, close_s3_session
from core.redis_instance import get_redis_client, close_redis_client
from db.database import db
from classes.ModelManager import ModelManager
import asyncio
import json
import os
import torch
from typing import Optional
from uuid import uuid4
import motor.motor_asyncio
import datetime
from huggingface_hub import login

app = FastAPI()
model_manager = ModelManager()

class QueryRequest(BaseModel):
    user_id: str
    avatar_id: Optional[str] = None
    query: str
    use_context: bool = False
    max_new_tokens: int = 50

@app.on_event("startup")
async def startup_event():
    """Initialize all services and preload the model on startup."""
    try:
        # Log system information
        logger.info("🚀 Starting application initialization...")
        logger.info(f"Python environment: {os.getenv('PYTHON_ENV', 'development')}")
        
        # Log GPU/CPU configuration
        use_gpu = os.getenv("USE_GPU", "0") == "1"
        force_cpu = os.getenv("FORCE_CPU", "0") == "1"
        logger.info(f"GPU Configuration - USE_GPU: {use_gpu}, FORCE_CPU: {force_cpu}")
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"CUDA available with {gpu_count} GPU(s)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            logger.info(f"CUDA_VISIBLE_DEVICES: {os.getenv('CUDA_VISIBLE_DEVICES', 'Not set')}")
        else:
            logger.info("CUDA not available - running on CPU")
        
        # Initialize Hugging Face login (if token available)
        if hasattr(settings, 'HUGGINGFACE_TOKEN') and settings.HUGGINGFACE_TOKEN:
            try:
                login(token=settings.HUGGINGFACE_TOKEN)
                logger.info("✅ Logged into Hugging Face Hub successfully.")
            except Exception as e:
                logger.warning(f"⚠️ Failed to login to Hugging Face Hub: {e}")
        
        # Initialize S3 and database connections
        logger.info("Initializing external services...")
        await init_s3_session()
        await db.db_connect()
        logger.info("✅ External services initialized successfully.")
        
        # **PRELOAD THE MODEL - This is the key addition**
        logger.info("Starting model preload...")
        await model_manager.preload_model_on_startup()
        logger.info("✅ Model preload completed successfully.")
        
        # Update metrics
        metrics.app_starts.inc()
        
        # Log final status
        device = model_manager.device
        model_loaded = model_manager.model is not None
        logger.info(f"🎉 Application startup completed successfully!")
        logger.info(f"   - Model loaded: {model_loaded}")
        logger.info(f"   - Device: {device}")
        logger.info(f"   - Ready for inference!")
        
    except Exception as e:
        logger.error(f"❌ Application startup failed: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        # Log additional context for debugging
        if torch.cuda.is_available():
            logger.error(f"   CUDA available: True")
            logger.error(f"   GPU count: {torch.cuda.device_count()}")
        else:
            logger.error(f"   CUDA available: False")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown of all services."""
    logger.info("🔄 Starting application shutdown...")
    
    try:
        # Close external connections
        await close_s3_session()
        await close_redis_client()
        await db.db_disconnect()
        
        # Clear GPU memory if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("✅ GPU memory cleared")
        
        logger.info("✅ Application shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint that verifies model and GPU status."""
    model_status = "loaded" if model_manager.model is not None else "not_loaded"
    device = model_manager.device
    
    # GPU-specific health info
    gpu_info = {}
    if torch.cuda.is_available() and device.startswith("cuda"):
        try:
            gpu_info = {
                "gpu_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1e9:.2f}GB",
                "gpu_memory_cached": f"{torch.cuda.memory_reserved(0) / 1e9:.2f}GB",
                "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB"
            }
        except Exception as e:
            gpu_info = {"gpu_error": str(e)}
    else:
        gpu_info = {"gpu_available": False}
    
    health_data = {
        "status": "healthy",
        "model_status": model_status,
        "device": device,
        "environment": {
            "use_gpu": os.getenv("USE_GPU", "0"),
            "force_cpu": os.getenv("FORCE_CPU", "0"),
            "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        },
        **gpu_info
    }
    
    return health_data

@app.get("/device-info")
async def device_info():
    """Endpoint to get detailed device and model information."""
    info = {
        "current_device": model_manager.device,
        "model_loaded": model_manager.model is not None,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "gpus": []
        })
        
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            info["gpus"].append({
                "index": i,
                "name": gpu_props.name,
                "total_memory": f"{gpu_props.total_memory / 1e9:.2f}GB",
                "allocated_memory": f"{torch.cuda.memory_allocated(i) / 1e9:.2f}GB" if i == 0 else "N/A",
                "cached_memory": f"{torch.cuda.memory_reserved(i) / 1e9:.2f}GB" if i == 0 else "N/A"
            })
    
    return info

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Handle query with optional avatar adapter and vectorstore context."""
    # Check if model is loaded
    if model_manager.model is None:
        logger.error("Model is not loaded. This should not happen after startup.")
        raise HTTPException(status_code=503, detail="Model is not ready for inference")
    
    # Log inference request
    logger.info(f"Processing query for user {request.user_id}, avatar {request.avatar_id}")
    logger.debug(f"Query: {request.query[:100]}...")  # Log first 100 chars
    
    # Store query in MongoDB
    conversation_id = str(uuid4())
    query_doc = {
        "_id": conversation_id,
        "avatar_id": request.avatar_id,
        "user_id": request.user_id,
        "type": "text",
        "message": request.query,
        "media": [],
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "sender": "user"
    }
    await db.avatars.insert_one(query_doc)
    logger.info(f"Stored query in MongoDB: {conversation_id}")
    metrics.db_operations.inc()

    # Push query to Redis cache
    redis_client = await get_redis_client()
    cache_key = f"chat:{request.user_id}:{request.avatar_id or 'default'}:full_messages"
    await redis_client.lpush(cache_key, json.dumps(query_doc))
    await redis_client.expire(cache_key, 3600)  # 1 hour TTL
    logger.info(f"Pushed query to Redis cache: {cache_key}")
    metrics.redis_operations_total.inc()

    # Generate response
    try:
        logger.info(f"Starting model inference on device: {model_manager.device}")
        start_time = datetime.datetime.utcnow()
        
        response_data = await model_manager.generate_response(
            user_input=request.query,
            user_id=request.user_id,
            avatar_id=request.avatar_id,
            use_context=request.use_context,
            max_new_tokens=request.max_new_tokens
        )
        
        inference_time = (datetime.datetime.utcnow() - start_time).total_seconds()
        logger.info(f"Model inference completed in {inference_time:.2f}s")
        
        # Add timing information to response
        response_data["inference_time_seconds"] = inference_time
        
    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # Log GPU memory status if applicable
        if torch.cuda.is_available() and model_manager.device.startswith("cuda"):
            try:
                allocated = torch.cuda.memory_allocated(0) / 1e9
                cached = torch.cuda.memory_reserved(0) / 1e9
                logger.error(f"GPU memory at error - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
            except Exception as gpu_e:
                logger.error(f"Could not get GPU memory info: {gpu_e}")
        
        metrics.model_errors.inc()
        raise HTTPException(status_code=500, detail=f"Model inference failed: {str(e)}")

    # Store response in MongoDB
    response_doc = {
        "_id": str(uuid4()),
        "avatar_id": request.avatar_id,
        "user_id": request.user_id,
        "type": "text",
        "message": response_data["response"],
        "media": [],
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "sender": "avatar",
        "conversation_id": conversation_id,
        "metadata": {
            "device": response_data.get("device", model_manager.device),
            "adapter_used": response_data.get("adapter_used", False),
            "context_used": response_data.get("context_used", False),
            "inference_time_seconds": response_data.get("inference_time_seconds", 0)
        }
    }
    await db.avatars.insert_one(response_doc)
    logger.info(f"Stored response in MongoDB: {response_doc['_id']}")
    metrics.db_operations.inc()

    # Push response to Redis stream
    stream_key = f"stream:{request.user_id}:{request.avatar_id or 'default'}"
    await redis_client.xadd(stream_key, {"message": json.dumps(response_doc)})
    await redis_client.publish("chat_updates", json.dumps(response_doc))
    logger.info(f"Pushed response to Redis stream: {stream_key}")
    metrics.redis_operations_total.inc()

    return response_data

@app.get("/cache-stream")
async def cache_stream():
    """Stream Redis pub/sub updates for chat messages."""
    async def event_generator():
        redis_client = await get_redis_client()
        pubsub = redis_client.pubsub()
        await pubsub.subscribe("chat_updates")
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = json.loads(message["data"])
                    yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(0.01)
        finally:
            await pubsub.unsubscribe()

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/clear-gpu-cache")
async def clear_gpu_cache():
    """Endpoint to manually clear GPU cache (useful for debugging)."""
    if not torch.cuda.is_available():
        return {"message": "CUDA not available", "success": False}
    
    try:
        before_allocated = torch.cuda.memory_allocated(0) / 1e9
        before_cached = torch.cuda.memory_reserved(0) / 1e9
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        after_allocated = torch.cuda.memory_allocated(0) / 1e9
        after_cached = torch.cuda.memory_reserved(0) / 1e9
        
        logger.info(f"GPU cache cleared - Before: {before_allocated:.2f}GB allocated, {before_cached:.2f}GB cached")
        logger.info(f"GPU cache cleared - After: {after_allocated:.2f}GB allocated, {after_cached:.2f}GB cached")
        
        return {
            "message": "GPU cache cleared successfully",
            "success": True,
            "memory_before": {
                "allocated_gb": round(before_allocated, 2),
                "cached_gb": round(before_cached, 2)
            },
            "memory_after": {
                "allocated_gb": round(after_allocated, 2),
                "cached_gb": round(after_cached, 2)
            },
            "memory_freed": {
                "allocated_gb": round(before_allocated - after_allocated, 2),
                "cached_gb": round(before_cached - after_cached, 2)
            }
        }
    except Exception as e:
        logger.error(f"Failed to clear GPU cache: {e}")
        return {"message": f"Failed to clear GPU cache: {str(e)}", "success": False}