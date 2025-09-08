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