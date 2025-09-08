# main.py
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
        # Initialize Hugging Face login (if token available)
        if hasattr(settings, 'HUGGINGFACE_TOKEN') and settings.HUGGINGFACE_TOKEN:
            login(token=settings.HUGGINGFACE_TOKEN)
            logger.info("Logged into Hugging Face Hub successfully.")
        
        # Initialize S3 and database connections
        await init_s3_session()
        await db.db_connect()
        
        # **PRELOAD THE MODEL - This is the key addition**
        await model_manager.preload_model_on_startup()
        
        # Update metrics
        metrics.app_starts.inc()
        
        logger.info("🚀 Application startup completed successfully. Model is ready for inference!")
        
    except Exception as e:
        logger.error(f"❌ Application startup failed: {e}")
        raise e

@app.on_event("shutdown")
async def shutdown_event():
    await close_s3_session()
    await close_redis_client()
    await db.db_disconnect()

@app.get("/health")
async def health_check():
    """Health check endpoint that also verifies model is loaded."""
    model_status = "loaded" if model_manager.model is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "device": model_manager.device
    }

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """Handle query with optional avatar adapter and vectorstore context."""
    # Check if model is loaded
    if model_manager.model is None:
        logger.error("Model is not loaded. This should not happen after startup.")
        raise HTTPException(status_code=503, detail="Model is not ready for inference")
    
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
        response_data = await model_manager.generate_response(
            user_input=request.query,
            user_id=request.user_id,
            avatar_id=request.avatar_id,
            use_context=request.use_context,
            max_new_tokens=request.max_new_tokens
        )
    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        metrics.model_errors.inc()
        raise HTTPException(status_code=500, detail="Model inference failed")

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
        "conversation_id": conversation_id
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