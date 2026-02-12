"""Repository layer for managing chat logs in MongoDB."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo.collection import Collection

from src.utils.config import MONGO_DB_NAME, get_mongo_client

from .models import ChatLog, ChatMessage, SourceRef


class ChatLogRepository:
    """Handle all chat log persistence to MongoDB."""

    COLLECTION_NAME = "chatlogs"

    def __init__(self) -> None:
        """Initialize repository with MongoDB connection."""
        self.client = get_mongo_client()
        self.db = self.client[MONGO_DB_NAME]
        self.collection: Collection = self.db[self.COLLECTION_NAME]

    def _generate_id(self, prefix: str) -> str:
        """Generate ID with prefix and unique suffix."""
        unique_suffix = str(uuid.uuid4())[:8]
        return f"{prefix}-{unique_suffix}"

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format with timezone."""
        return datetime.now(timezone.utc).isoformat()

    def create_conversation(self, first_user_message: str) -> ChatLog:
        """Create a new conversation with the first user message."""
        now = self._get_current_timestamp()
        conv_id = self._generate_id("conv")
        
        # Use first 50 chars of message as title
        title = first_user_message[:50]
        if len(first_user_message) > 50:
            title += "..."

        chat_log = ChatLog(
            id=conv_id,
            title=title,
            created_at=now,
            updated_at=now,
            messages=[
                ChatMessage(
                    id=self._generate_id("msg"),
                    type="user",
                    content=first_user_message,
                    timestamp=now,
                    refs=None,
                )
            ],
        )

        # Insert into MongoDB
        self.collection.insert_one(self._model_to_dict(chat_log))
        return chat_log

    def add_assistant_message(
        self,
        conversation_id: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatLog:
        """Add assistant response to existing conversation."""
        now = self._get_current_timestamp()
        
        # Convert sources to SourceRef objects
        refs = None
        if sources:
            refs = [
                SourceRef(
                    link=src.get("link", ""),
                    text=src.get("text", ""),
                    score=src.get("score"),
                    dense_score=src.get("dense_score"),
                )
                for src in sources
            ]

        assistant_message = ChatMessage(
            id=self._generate_id("msg"),
            type="assistant",
            content=content,
            timestamp=now,
            refs=refs,
        )

        # Update conversation in MongoDB
        result = self.collection.update_one(
            {"id": conversation_id},
            {
                "$push": {"messages": self._message_to_dict(assistant_message)},
                "$set": {"updated_at": now},
            },
        )

        if result.matched_count == 0:
            raise ValueError(f"Conversation {conversation_id} not found")

        # Fetch and return updated conversation
        return self.get_conversation(conversation_id)

    def add_user_message(
        self, conversation_id: str, content: str
    ) -> ChatLog:
        """Add user message to existing conversation."""
        now = self._get_current_timestamp()

        user_message = ChatMessage(
            id=self._generate_id("msg"),
            type="user",
            content=content,
            timestamp=now,
            refs=None,
        )

        # Update conversation in MongoDB
        result = self.collection.update_one(
            {"id": conversation_id},
            {
                "$push": {"messages": self._message_to_dict(user_message)},
                "$set": {"updated_at": now},
            },
        )

        if result.matched_count == 0:
            raise ValueError(f"Conversation {conversation_id} not found")

        return self.get_conversation(conversation_id)

    def get_conversation(self, conversation_id: str) -> Optional[ChatLog]:
        """Retrieve a conversation by ID."""
        doc = self.collection.find_one({"id": conversation_id})
        if not doc:
            return None
        return self._dict_to_model(doc)

    def list_conversations(self, limit: int = 50) -> List[ChatLog]:
        """List recent conversations."""
        docs = (
            self.collection.find()
            .sort("updated_at", -1)
            .limit(limit)
        )
        return [self._dict_to_model(doc) for doc in docs]

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        result = self.collection.delete_one({"id": conversation_id})
        return result.deleted_count > 0

    @staticmethod
    def _model_to_dict(chat_log: ChatLog) -> Dict[str, Any]:
        """Convert ChatLog model to MongoDB document."""
        return {
            "id": chat_log.id,
            "title": chat_log.title,
            "created_at": chat_log.created_at,
            "updated_at": chat_log.updated_at,
            "messages": [
                ChatLogRepository._message_to_dict(msg)
                for msg in chat_log.messages
            ],
        }

    @staticmethod
    def _message_to_dict(message: ChatMessage) -> Dict[str, Any]:
        """Convert ChatMessage model to MongoDB document."""
        msg_dict = {
            "id": message.id,
            "type": message.type,
            "content": message.content,
            "timestamp": message.timestamp,
        }
        if message.refs:
            msg_dict["refs"] = [
                {
                    "link": ref.link,
                    "text": ref.text,
                    "score": ref.score,
                    "dense_score": ref.dense_score,
                }
                for ref in message.refs
            ]
        else:
            msg_dict["refs"] = None
        return msg_dict

    @staticmethod
    def _dict_to_model(doc: Dict[str, Any]) -> ChatLog:
        """Convert MongoDB document to ChatLog model."""
        messages = []
        for msg_doc in doc.get("messages", []):
            refs = None
            if msg_doc.get("refs"):
                refs = [
                    SourceRef(
                        link=ref.get("link", ""),
                        text=ref.get("text", ""),
                        score=ref.get("score"),
                        dense_score=ref.get("dense_score"),
                    )
                    for ref in msg_doc["refs"]
                ]
            
            message = ChatMessage(
                id=msg_doc.get("id", ""),
                type=msg_doc.get("type", ""),
                content=msg_doc.get("content", ""),
                timestamp=msg_doc.get("timestamp", ""),
                refs=refs,
            )
            messages.append(message)

        return ChatLog(
            id=doc.get("id", ""),
            title=doc.get("title"),
            created_at=doc.get("created_at", ""),
            updated_at=doc.get("updated_at", ""),
            messages=messages,
        )
