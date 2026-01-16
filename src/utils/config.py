"""Configuration settings for MongoDB connection."""

from pymongo import MongoClient

MONGO_URI = "mongodb+srv://Neuaiagent_db:rjuhWX6fKqcgVthx@cluster0-newproject.7uk6vxn.mongodb.net/?appName=Cluster0-Newproject"
MONGO_DB_NAME = "chatbotNeu"


def get_mongo_client() -> MongoClient:
    """Get MongoDB client instance."""
    return MongoClient(MONGO_URI)

