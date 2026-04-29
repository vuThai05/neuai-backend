"""Repository layer for managing chat logs in MongoDB."""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo.collection import Collection

from src.utils.config import MONGO_DB_NAME, get_mongo_client

from .models import ChatLog, ChatMessage, DecisionLog, SourceRef


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

        self.collection.insert_one(self._model_to_dict(chat_log))
        return chat_log

    def add_assistant_message(
        self,
        conversation_id: str,
        content: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        decision: Optional[Dict[str, Any]] = None,
    ) -> ChatLog:
        """Add assistant response (with optional sources and decision) to an existing conversation."""
        now = self._get_current_timestamp()

        refs: Optional[List[SourceRef]] = None
        if sources:
            refs = [self._coerce_source_ref(src) for src in sources]

        decision_model: Optional[DecisionLog] = None
        if decision:
            decision_model = DecisionLog(
                route=str(decision.get("route", "")),
                reason_code=str(decision.get("reason_code", "")),
                reason=str(decision.get("reason", "")),
                confidence=float(decision.get("confidence", 0.0)),
            )

        assistant_message = ChatMessage(
            id=self._generate_id("msg"),
            type="assistant",
            content=content,
            timestamp=now,
            refs=refs,
            decision=decision_model,
        )

        result = self.collection.update_one(
            {"id": conversation_id},
            {
                "$push": {"messages": self._message_to_dict(assistant_message)},
                "$set": {"updated_at": now},
            },
        )

        if result.matched_count == 0:
            raise ValueError(f"Conversation {conversation_id} not found")

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
    def _coerce_source_ref(src: Dict[str, Any]) -> SourceRef:
        """Build a `SourceRef` from either the new schema or a legacy record."""
        if any(key in src for key in ("snippet", "url", "type", "title")):
            return SourceRef(
                type=str(src.get("type", "internal")),
                title=str(src.get("title", "")),
                url=src.get("url"),
                snippet=str(src.get("snippet", "")),
                score=src.get("score"),
            )

        # Legacy shape from pre-AI-Agent records: {link, text, score, dense_score}.
        return SourceRef(
            type="internal",
            title="",
            url=src.get("link"),
            snippet=str(src.get("text", "")),
            score=src.get("score"),
        )

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
        msg_dict: Dict[str, Any] = {
            "id": message.id,
            "type": message.type,
            "content": message.content,
            "timestamp": message.timestamp,
        }

        if message.refs:
            msg_dict["refs"] = [
                {
                    "type": ref.type,
                    "title": ref.title,
                    "url": ref.url,
                    "snippet": ref.snippet,
                    "score": ref.score,
                }
                for ref in message.refs
            ]
        else:
            msg_dict["refs"] = None

        if message.decision:
            msg_dict["decision"] = {
                "route": message.decision.route,
                "reason_code": message.decision.reason_code,
                "reason": message.decision.reason,
                "confidence": message.decision.confidence,
            }
        else:
            msg_dict["decision"] = None

        return msg_dict

    @staticmethod
    def _dict_to_model(doc: Dict[str, Any]) -> ChatLog:
        """Convert MongoDB document to ChatLog model.

        Tolerates legacy records that used the old `{link, text, score, dense_score}`
        source format and have no `decision` field.
        """
        messages: List[ChatMessage] = []
        for msg_doc in doc.get("messages", []):
            refs: Optional[List[SourceRef]] = None
            raw_refs = msg_doc.get("refs")
            if raw_refs:
                refs = [ChatLogRepository._coerce_source_ref(ref) for ref in raw_refs]

            decision: Optional[DecisionLog] = None
            raw_decision = msg_doc.get("decision")
            if raw_decision:
                decision = DecisionLog(
                    route=str(raw_decision.get("route", "")),
                    reason_code=str(raw_decision.get("reason_code", "")),
                    reason=str(raw_decision.get("reason", "")),
                    confidence=float(raw_decision.get("confidence", 0.0)),
                )

            messages.append(
                ChatMessage(
                    id=msg_doc.get("id", ""),
                    type=msg_doc.get("type", ""),
                    content=msg_doc.get("content", ""),
                    timestamp=msg_doc.get("timestamp", ""),
                    refs=refs,
                    decision=decision,
                )
            )

        return ChatLog(
            id=doc.get("id", ""),
            title=doc.get("title"),
            created_at=doc.get("created_at", ""),
            updated_at=doc.get("updated_at", ""),
            messages=messages,
        )
