from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ConversationTurn:
    """A single user/assistant conversation turn."""

    user_message: str
    assistant_message: str
    timestamp: datetime = field(
        default_factory=datetime.utcnow
    )

    def __post_init__(self) -> None:
        if not isinstance(self.user_message, str):
            raise TypeError(
                "user_message must be a string."
            )

        if not self.user_message.strip():
            raise ValueError(
                "user_message cannot be empty."
            )

        if not isinstance(self.assistant_message, str):
            raise TypeError(
                "assistant_message must be a string."
            )

        if not self.assistant_message.strip():
            raise ValueError(
                "assistant_message cannot be empty."
            )

        if not isinstance(self.timestamp, datetime):
            raise TypeError(
                "timestamp must be a datetime."
            )


@dataclass(frozen=True)
class MemoryItem:
    """A persistent memory item."""

    content: str
    memory_type: str = "fact"
    metadata: dict[str, Any] = field(
        default_factory=dict
    )
    created_at: datetime = field(
        default_factory=datetime.utcnow
    )

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError(
                "content must be a string."
            )

        if not self.content.strip():
            raise ValueError(
                "content cannot be empty."
            )

        if not isinstance(self.memory_type, str):
            raise TypeError(
                "memory_type must be a string."
            )

        if not self.memory_type.strip():
            raise ValueError(
                "memory_type cannot be empty."
            )

        if not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary."
            )

        if not isinstance(self.created_at, datetime):
            raise TypeError(
                "created_at must be a datetime."
            )