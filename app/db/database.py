
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
