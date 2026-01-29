"""Configuration settings for MongoDB connection."""

import os
from pymongo import MongoClient

MONGO_URI = os.environ.get(
    "MONGODB_URI",
    "mongodb+srv: ###",
)
MONGO_DB_NAME = "chatbotNeu"

def get_mongo_client() -> MongoClient:
    """Get MongoDB client instance."""
    return MongoClient(MONGO_URI)
