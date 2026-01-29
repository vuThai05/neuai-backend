"""Configuration settings for MongoDB connection."""

import os
from pymongo import MongoClient

MONGO_URI = os.environ.get("MONGODB_URI", "")
MONGO_DB_NAME = "chatbotNeu"


def get_mongo_client() -> MongoClient:
    """Get MongoDB client instance."""
    if not MONGO_URI:
        raise RuntimeError(
            "MONGODB_URI environment variable is not set. "
            "Please configure it in your .env file."
        )
    return MongoClient(MONGO_URI)
