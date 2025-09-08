Please update the ModelManager and main.py to perform this behavior.

I need to be able to send a query to the query endpoint. The adapter name is the name of the avatar_id. The adapter is stored in s3. I need to lazy load the adapter. I need to cache recently used adapters in the container locally. This api needs to have the /query endpoint allow for the avatar_id to be sent as an optional parameter. If the current adapter is not the adapter for the avatar_id, then it will check the cache. If the adapter is cached then it will attach the adapter and complete the query. if not cached, it will check the s3 endpoint. If available in s3 it will pull and attach the adapter as well as cache the adapter in recent use. If the adapter is not available in s3, the base model will be queried and a response will be appended indicating that the adapter was not used. There needs to be an optional boolean to enable context via a vectorstore. When true, an external query to a chroma_db vectorstore is made and the response is appended to the model response. (the model will use the context)

For example, the context is first sent to the vectorstore via http, then the result is added to the prompt, tokenized, and the model is then queried. This endpoint will use an http client for chroma_db that will need to be added at a later time. 

Here is an example of the use:

def generate_with_context(user_input: str, top_k: int = 10, max_new_tokens: int = 50) -> str:
    # Embed query and retrieve from vector DB
    query_vec = embedder.encode(user_input).tolist()
    results = collection.query(query_embeddings=[query_vec], n_results=top_k)
    print(f"results: {results}")

    docs = results.get("documents", [[]])[0]
    if not docs:
        context = "No relevant context found."
    else:
        # Optionally truncate context length for model input token limits
        # Here we join and limit length (e.g., first 1000 chars)
        context = "\n".join(docs)
        context = context[:1000]

    # Improved prompt with explicit instruction and clear delimiters
    prompt = f"""You are the person in the contextual statements. Use the context to answer the question briefly and only once.

Context:
{context}

Q: {user_input}
A:"""

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(device)

    # Generate answer
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Set True to enable sampling
            # temperature=0.7,  # Uncomment if do_sample=True
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract text after last "A:" in case prompt or output has multiple
    answer = decoded.split("A:")[-1].strip()
    return answer
generate_with_context(user_input=input())


----

The query endpoint needs to save the user query in mongo_db, push the query into the redis cache, perform inference as described above to produce a response, store the avatar response in mongo_db for the avatar_conversation and push the new response to the redis stream. It needs to provide a separate endpoint for Server Side Events that will update the frontend with every message in the redis cache.

Here is a trivial example:
#### Backend (FastAPI with Redis Pub/Sub)
```python
# server.py
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import redis
import json
import asyncio

app = FastAPI()
redis_client = redis.Redis(host='redis', port=6379, decode_responses=True)

# Endpoint to set cache and publish update
@app.post("/set-cache/{key}")
async def set_cache(key: str, value: str):
    redis_client.set(key, value)
    redis_client.publish("cache_updates", json.dumps({"key": key, "value": value}))
    return {"status": "set"}

# SSE endpoint to stream Redis Pub/Sub updates
@app.get("/cache-stream")
async def cache_stream():
    async def event_generator():
        pubsub = redis_client.pubsub()
        pubsub.subscribe("cache_updates")
        try:
            while True:
                message = pubsub.get_message(ignore_subscribe_messages=True, timeout=30.0)
                if message and message["type"] == "message":
                    data = json.loads(message["data"])
                    yield f"data: {json.dumps(data)}\n\n"
                await asyncio.sleep(0.01)  # Yield control to event loop
        finally:
            pubsub.unsubscribe()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Example Avatar Object in MongoDB
_id: ObjectId('68af1a01608eb5e33d6f16ae')
avatar_id:"246876a7-8328-4a19-93e2-38ea259ad59f"
user_id:"27df12d9-9881-4369-bd75-e5c9538b0ea2"
name:"ev0ra"
description:"Music, Games, DJ, IRL & constant shenaniganry. Silly goose energy, com…"
created_at:2025-08-27T14:45:21.546+00:00
icon:"users/27df12d9-9881-4369-bd75-e5c9538b0ea2/avatars/246876a7-8328-4a19-…"
files: Array (empty)
messages: Array (empty)

# Example User Object in MongoDB
_id:ObjectId('68a946f2887bd9474a4c4fa4')
user_id:"27df12d9-9881-4369-bd75-e5c9538b0ea2"
username:"string"
email:"user@example.com"
password:"$2b$12$.8pKXfep7qWhMaKuaG9zO.eRs802JIsHL8.twXWP3QEYTjDceTGvy"
created_at:2025-08-23T04:43:30.342+00:00
last_login:2025-09-04T20:03:27.694+00:00
currently_logged_in:false
personal_image:"users/27df12d9-9881-4369-bd75-e5c9538b0ea2/image/future-of-humanity.pn…"
neural_nexus_api_key:null
grok_api_key:null
enable_grok_imagine:false
elevenlabs_api_key:null
enable_elevenlabs:false
api_usage:Object
billing_history:Array (empty)
credit_card:null
avatars:Array (8)
last_used_avatar:"246876a7-8328-4a19-93e2-38ea259ad59f"

# Example Avatar Conversations Object in MongoDB
## I will need the avatar conversations to be traceable threads per user per avatar
_id:"a8bbc654-7a5e-4606-822e-b7c28c4a495e"
avatar_id:"03631b03-2607-4a64-a3a6-f6ada35adf6c"
user_id:"27df12d9-9881-4369-bd75-e5c9538b0ea2"
type:"text"
message:"Hey Shivon!"
media:Array (empty)
timestamp:2025-08-26T02:43:08.303+00:00
sender:"user"


# S3_persistence Structure:
This is the S3 Persistence Structure
users/{user_id}/
├── vectorstore/                   # User-level vectorstore (chroma_db)
├── avatars/{avatar_id}/
│   ├── vectorstore_data/          # Avatar-specific context data (preprocessed)
│   ├── adapters/                  # QLoRA adapter files (the actual Adapter is stored here)
│   ├── adapters/training_data/    # Training data for fine-tuning (preprocessed for the LoRA Adapter)
|   └── media/                     # Unprocessed media for a specific avatar (audio/video/images/documents)
|── image/                         # User-level personal image
|── *{other_potential_user_level_folders}  # Other potential user-level folders such as billing & account information

This is the tree:

.
├── app
│   ├── api
│   │   └── __init__.py
│   ├── core
│   │   ├── config.py
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   ├── monitoring.py
│   │   ├── redis_instance.py
│   │   └── s3_instance.py
│   ├── db
│   │   ├── database.py
│   │   ├── __init__.py
│   │   └── schema
│   │       ├── avatar.py
│   │       └── __init__.py
│   ├── main.py
│   ├── models
│   │   ├── __init__.py
│   │   └── ModelManager.py
│   └── service
├── Docker
│   ├── docker-compose.gpu.yml
│   └── Dockerfile.gpu
├── docker-compose.prod.yml
├── docker-compose.yml
├── Dockerfile
├── Dockerfile.dev
├── elon_musk.md
├── elon_q_and_a.txt
├── elon_speech.txt
├── LICENSE
├── notes.md
├── prompt.md
├── __pycache__
│   └── app.cpython-310.pyc
├── README.md
└── requirements.txt


---

This is the s3_instance:

"""
Fully asynchronous S3 client management for Neural Nexus Database API.
Uses aioboto3 exclusively for all S3 operations.
"""

import aioboto3
from fastapi import HTTPException
from core.config import settings
from core.logging import logger
from core.monitoring import metrics
import os
from fastapi import UploadFile, HTTPException
from pathlib import Path
from botocore.exceptions import ClientError

from core.config import settings

# Configuration
BUCKET_NAME = settings.BUCKET_NAME
MESSAGE_CACHE_TTL_SECONDS = 3600  # 1 hour cache

# Global S3 session for async operations
_s3_session = None


async def init_s3_session():
    """Initialize aioboto3 session with credentials from settings."""
    global _s3_session
    try:
        _s3_session = aioboto3.Session(
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
        async with _s3_session.client("s3") as s3_client:
            await s3_client.list_buckets()
        logger.info("S3 session initialized successfully")
        metrics.s3_connections.inc()
    except Exception as e:
        logger.error(f"Failed to initialize S3 session: {e}")
        metrics.s3_connection_errors.inc()
        raise HTTPException(status_code=500, detail="Failed to connect to S3")


def get_s3_session():
    """Get the initialized aioboto3 session."""
    if _s3_session is None:
        logger.error("S3 session not initialized")
        metrics.s3_connection_errors.inc()
        raise HTTPException(status_code=500, detail="S3 session not initialized")
    return _s3_session


async def get_s3_client():
    """Async context manager that yields an S3 client."""
    session = get_s3_session()
    return session.client("s3")  # aioboto3 client is async-compatible

async def close_s3_session():
    """Cleanup for S3 session."""
    global _s3_session
    _s3_session = None
    logger.info("S3 session closed")


async def upload_image_to_s3(file: UploadFile, s3_key: str, user_id: str) -> str:
    """Upload a file to S3 asynchronously and return its URL."""
    async with await get_s3_client() as s3_client:
        allowed_extensions = {".png", ".jpg", ".jpeg", ".gif"}
        max_file_size = 5 * 1024 * 1024  # 5MB
        file_ext = os.path.splitext(file.filename)[1].lower()

        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail="Invalid image format. Allowed: png, jpg, jpeg, gif",
            )
        if file.size > max_file_size:
            raise HTTPException(status_code=400, detail="File size exceeds 5MB limit")

        try:
            await s3_client.upload_fileobj(
                file.file,
                BUCKET_NAME,
                s3_key,
                ExtraArgs={"ContentType": file.content_type},
            )
            s3_url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
            metrics.s3_operations_success.inc()
            logger.info(f"Uploaded image to S3 for user {user_id}: {s3_key}")
            return s3_url
        except ClientError as e:
            logger.error(f"S3 upload failed for user {user_id}: {e}")
            metrics.s3_operations_errors.inc()
            raise HTTPException(status_code=503, detail="Failed to upload image to S3")


async def delete_user_image(user_id: str, user_image_url: str):
    """Delete the previous personal image from S3 asynchronously if it exists."""
    async with await get_s3_client() as s3_client:
        if user_image_url:
            try:
                s3_key = user_image_url.replace(
                    f"https://{BUCKET_NAME}.s3.amazonaws.com/", ""
                )
                await s3_client.delete_object(Bucket=BUCKET_NAME, Key=s3_key)
                logger.info(f"Deleted old image from S3: {s3_key}")
                metrics.s3_operations_success.inc()
            except ClientError as e:
                logger.warning(f"Failed to delete old image {s3_key}: {e}")
                metrics.s3_operations_errors.inc()


async def generate_presigned_url(key: str, expiration: int = 3600) -> str:
    """Generate a presigned URL for an S3 object asynchronously."""
    async with await get_s3_client() as s3_client:
        try:
            url = await s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": BUCKET_NAME, "Key": key},
                ExpiresIn=expiration,
            )
            logger.info(f"Generated presigned URL for key {key}")
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL for {key}: {e}")
            metrics.s3_operations_errors.inc()
            raise HTTPException(
                status_code=503, detail="Failed to generate presigned URL"
            )


async def upload_dir_to_s3(local_path: Path, s3_prefix: str):
    """Upload a directory to S3 asynchronously."""
    async with await get_s3_client() as s3_client:
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file = os.path.join(root, file)
                rel_path = os.path.relpath(local_file, local_path)
                key = f"{s3_prefix}{rel_path}"
                try:
                    await s3_client.upload_file(local_file, BUCKET_NAME, key)
                    logger.info(f"Uploaded {local_file} to S3 as {key}")
                    metrics.s3_operations_success.inc()
                except ClientError as e:
                    logger.error(f"Failed to upload {local_file} to S3: {e}")
                    metrics.s3_operations_errors.inc()


async def download_s3_to_dir(s3_prefix: str, local_path: Path):
    """Download all objects with the given S3 prefix to a local directory asynchronously."""
    async with await get_s3_client() as s3_client:
        local_path.mkdir(parents=True, exist_ok=True)
        paginator = s3_client.get_paginator("list_objects_v2")
        try:
            async for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=s3_prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    rel_key = key[len(s3_prefix) :]
                    local_file = local_path / rel_key
                    local_file.parent.mkdir(parents=True, exist_ok=True)
                    await s3_client.download_file(BUCKET_NAME, key, str(local_file))
                    logger.info(f"Downloaded {key} to {local_file}")
                    metrics.s3_operations_success.inc()
        except ClientError as e:
            logger.error(f"Failed to download S3 prefix {s3_prefix}: {e}")
            metrics.s3_operations_errors.inc()


async def delete_s3_folder(prefix: str):
    """Delete all objects with the given S3 prefix asynchronously."""
    async with await get_s3_client() as s3_client:
        paginator = s3_client.get_paginator("list_objects_v2")
        try:
            async for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
                if "Contents" in page:
                    delete_keys = {
                        "Objects": [{"Key": obj["Key"]} for obj in page["Contents"]]
                    }
                    await s3_client.delete_objects(
                        Bucket=BUCKET_NAME, Delete=delete_keys
                    )
                    logger.info(f"Deleted S3 objects with prefix {prefix}")
                    metrics.s3_operations_success.inc()
        except ClientError as e:
            logger.error(f"Failed to delete S3 prefix {prefix}: {e}")
            metrics.s3_operations_errors.inc()


async def get_presigned_url(
    s3_key: str, bucket_name: str = BUCKET_NAME, expiration: int = 3600
) -> str:
    """
    Generate a temporary URL to access a private S3 object asynchronously.
    """
    async with await get_s3_client() as s3_client:
        try:
            url = await s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket_name, "Key": s3_key},
                ExpiresIn=expiration,
            )
            logger.info(f"Generated presigned URL for {s3_key}")
            metrics.s3_operations_success.inc()
            return url
        except ClientError as e:
            logger.error(f"Error generating presigned URL for {s3_key}: {e}")
            metrics.s3_operations_errors.inc()
            raise HTTPException(
                status_code=503, detail="Failed to generate presigned URL"
            )


def get_cache_key(user_id: str, avatar_id: str) -> str:
    """Generate a Redis cache key for user and avatar."""
    return f"chat:{user_id}:{avatar_id}:full_messages"


async def upload_document_to_s3(
    content: bytes, s3_key: str, content_type: str, s3_client
) -> str:
    """Upload document content to S3 asynchronously."""
    try:
        await s3_client.put_object(
            Bucket=BUCKET_NAME, Key=s3_key, Body=content, ContentType=content_type
        )
        logger.info(f"Uploaded document to S3: {s3_key}")
        metrics.s3_operations_success.inc()
        return s3_key
    except ClientError as e:
        logger.error(f"Failed to upload document to S3: {s3_key}, error: {e}")
        metrics.s3_operations_errors.inc()
        raise HTTPException(
            status_code=500, detail=f"Failed to upload document: {str(e)}"
        )


async def delete_document_from_s3(s3_key: str, s3_client):
    """Delete document from S3 asynchronously."""
    try:
        await s3_client.delete_object(Bucket=BUCKET_NAME, Key=s3_key)
        logger.info(f"Deleted document from S3: {s3_key}")
        metrics.s3_operations_success.inc()
    except ClientError as e:
        logger.error(f"Failed to delete document from S3: {s3_key}, error: {e}")
        metrics.s3_operations_errors.inc()


async def upload_file_to_s3(content: bytes, key: str, content_type: str, s3_client):
    """Upload file to S3 asynchronously."""
    try:
        await s3_client.put_object(
            Bucket=BUCKET_NAME, Key=key, Body=content, ContentType=content_type
        )
        logger.info(f"Uploaded {key} to S3")
        metrics.s3_operations_success.inc()
    except ClientError as e:
        logger.error(f"Failed to upload {key} to S3: {e}")
        metrics.s3_operations_errors.inc()
        raise HTTPException(status_code=503, detail=f"S3 upload failed: {str(e)}")



# Additional helper functions to add to your codebase:

async def ensure_user_vectorstore_exists(user_id: str, s3_client):
    """Ensure user-level vectorstore directory exists."""
    vectorstore_path = f"users/{user_id}/vectorstore/.keep"
    try:
        await s3_client.head_object(Bucket=BUCKET_NAME, Key=vectorstore_path)
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            # Directory doesn't exist, create it
            await upload_file_to_s3(b"", vectorstore_path, "text/plain", s3_client)
            logger.info(f"Created user vectorstore directory for {user_id}")
        else:
            raise e

async def ensure_avatar_directories_exist(user_id: str, avatar_id: str, s3_client):
    """Ensure all avatar directories exist with proper structure."""
    base_path = f"users/{user_id}/avatars/{avatar_id}"
    
    required_paths = [
        f"{base_path}/vectorstore_data/.keep",
        f"{base_path}/adapters/.keep", 
        f"{base_path}/adapters/training_data/.keep"
    ]
    
    for path in required_paths:
        try:
            await s3_client.head_object(Bucket=BUCKET_NAME, Key=path)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                await upload_file_to_s3(b"", path, "text/plain", s3_client)
                logger.info(f"Created directory: {path}")
            else:
                raise e

# ======================================================================

# ChromaDB integration paths - Use these in your vectorstore operations:

def get_user_vectorstore_path(user_id: str) -> str:
    """Get S3 path for user-level vectorstore."""
    return f"users/{user_id}/vectorstore/"

def get_avatar_vectorstore_path(user_id: str, avatar_id: str) -> str:
    """Get S3 path for avatar-specific vectorstore data."""
    return f"users/{user_id}/avatars/{avatar_id}/vectorstore_data/"

def get_avatar_adapter_path(user_id: str, avatar_id: str) -> str:
    """Get S3 path for avatar adapters."""
    return f"users/{user_id}/avatars/{avatar_id}/adapters/"

def get_avatar_training_data_path(user_id: str, avatar_id: str) -> str:
    """Get S3 path for avatar training data."""
    return f"users/{user_id}/avatars/{avatar_id}/adapters/training_data/"


----

# app/core/redis_instance.py
import asyncio
import redis.asyncio as redis
from core.config import settings
from core.logging import logger
from core.monitoring import metrics

import json
from typing import Optional
_redis_client = None
_redis_lock = asyncio.Lock()


async def get_redis_client():
    global _redis_client
    async with _redis_lock:
        if _redis_client is None:
            try:
                # Remote Connection
                _redis_client = redis.Redis(
                    host=settings.REDIS_HOST,
                    port=settings.REDIS_PORT,
                    username=settings.REDIS_USERNAME,
                    password=settings.REDIS_PASSWORD,
                    decode_responses=True,
                    ssl=True,  # if your connection uses TLS (check if the URL starts with rediss://)
                )
                # Test connection
                await _redis_client.ping()
                logger.info("Redis client initialized.")
                metrics.db_connection_status.labels(database="redis").set(1)
            except Exception as e:
                logger.error(f"Failed to initialize Redis client: {e}")
                metrics.db_connection_status.labels(database="redis").set(0)
                _redis_client = None
                raise RuntimeError(
                    f"Redis initialization failed: {e}"
                )  # Raise exception to propagate error
    return _redis_client


async def close_redis_client():
    global _redis_client
    if _redis_client is not None:
        try:
            await _redis_client.close()
            logger.info("Redis client closed.")
            _redis_client = None
            metrics.db_connection_status.labels(database="redis").set(0)
        except Exception as e:
            logger.error(f"Failed to close Redis client: {e}")


async def update_avatar_cache_after_creation(
    redis_client, 
    user_id: str, 
    avatar_id: str, 
    name: str, 
    description: str, 
    icon_url: Optional[str]
):
    """Update all relevant avatar cache entries after creating a new avatar."""
    try:
        # Create the new avatar object that matches AvatarPerUser structure
        new_avatar_data = {
            "avatar_id": avatar_id,
            "name": name,
            "description": description,
            "icon": icon_url,
        }
        
        # Find all existing cache keys for this user
        cache_pattern = f"avatars:{user_id}:all:*"
        cache_keys = await redis_client.keys(cache_pattern)
        
        for cache_key in cache_keys:
            try:
                # Get existing cached data
                cached_data = await redis_client.get(cache_key)
                if cached_data:
                    existing_avatars = json.loads(cached_data)
                    
                    # Parse the cache key to get skip and limit values
                    key_parts = cache_key.split(':')
                    if len(key_parts) >= 5:
                        skip = int(key_parts[3])
                        limit = int(key_parts[4])
                        
                        # Add new avatar to the beginning (most recent first)
                        # Since your query sorts by created_at ascending, new avatar should go at the end
                        # But for better UX, you might want to show newest first
                        existing_avatars.append(new_avatar_data)
                        
                        # If we exceed the limit, remove the first item (oldest)
                        if len(existing_avatars) > limit:
                            existing_avatars = existing_avatars[-limit:]  # Keep the last 'limit' items
                        
                        # Update cache with new data
                        await redis_client.setex(
                            cache_key,
                            3600,  # Same TTL as original
                            json.dumps(existing_avatars)
                        )
                        
                        logger.info(f"Updated cache {cache_key} with new avatar {avatar_id}")
                        
            except (json.JSONDecodeError, ValueError, IndexError) as e:
                # If we can't parse/update a specific cache entry, just delete it
                logger.warning(f"Failed to update cache {cache_key}, deleting: {e}")
                await redis_client.delete(cache_key)
                
        metrics.redis_operations_total.inc()
        
    except Exception as e:
        logger.error(f"Failed to update avatar cache after creation: {e}")
        # Don't raise exception here - cache update failure shouldn't break avatar creation


async def clear_user_cache(redis_client, user_id: str):
    """Clear all cache entries for a specific user."""
    try:
        # Define cache patterns for this user
        cache_patterns = [
            f"avatars:{user_id}:*",          # All avatar-related cache
            f"user:{user_id}:*",             # User profile cache
            f"messages:{user_id}:*",         # Message cache
            f"files:{user_id}:*",            # File cache
            f"conversations:{user_id}:*",    # Conversation cache
            # Add other user-specific cache patterns as needed
        ]
        
        keys_to_delete = []
        
        # Collect all keys matching the patterns
        for pattern in cache_patterns:
            matching_keys = await redis_client.keys(pattern)
            keys_to_delete.extend(matching_keys)
        
        # Delete all collected keys
        if keys_to_delete:
            deleted_count = await redis_client.delete(*keys_to_delete)
            logger.info(f"Cleared {deleted_count} cache entries for user {user_id}")
            metrics.redis_operations_total.inc()
        else:
            logger.info(f"No cache entries found for user {user_id}")
            
    except Exception as e:
        logger.error(f"Failed to clear cache for user {user_id}: {e}")
        metrics.redis_errors.inc()
        # Don't raise exception - cache clearing failure shouldn't break logout

---

This is the mongoDB


from core.config import settings
from core.logging import logger
from core.monitoring import metrics
from motor.motor_asyncio import AsyncIOMotorClient
from db.database import db

class Database:
    def __init__(self):
        # MongoDB
        self.mongo_client = None
        self.mongo_db = None

        # ChromaDB - Global instances
        self.chroma_client = None
        self.training_documents_collection = None
        self.processed_media_collection = None

        # Unique identifier for this database instance
        self._id = id(self)

    def get_id(self):
        return self._id

    # Properties for Collections
    @property
    def users(self):
        if self.mongo_db is None:  # <-- Fix here
            raise RuntimeError("MongoDB not initialized")
        return self.mongo_db["users"]

    @property
    def avatars(self):
        if self.mongo_db is None:  # <-- Fix here
            raise RuntimeError("MongoDB not initialized")
        return self.mongo_db["avatars"]

    async def init_mongodb():
        try:
            # Remote Connection:
            db.mongo_client = AsyncIOMotorClient(settings.MONGO_URI)

            db.mongo_db = db.mongo_client[settings.MONGO_DB]
            await db.mongo_client.admin.command("ping")
            logger.info("MongoDB client initialized.")
            metrics.db_connection_status.labels(database="mongodb").set(1)
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB: {e}")
            db.mongo_client = None
            db.mongo_db = None
            metrics.db_connection_status.labels(database="mongodb").set(0)


    async def db_connect():
        logger.debug("Database.connect() called")
        await init_mongodb()
        # init_chroma
        logger.debug(f"Database connection pools: mongo_client={db.mongo_client}")
        logger.debug(f"[connect] Database instance id: {id(db.get_id())}")


    async def db_disconnect():
        if db.mongo_client:
            db.mongo_client.close()
        metrics.db_connection_status.labels(database="mongodb").set(0)


# Global database instance
db = Database()
