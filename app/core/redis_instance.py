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