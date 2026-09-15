from typing import Any

from ragforge.memory.config import MemoryConfig
from ragforge.memory.long_term import LongTermMemory
from ragforge.memory.models import (
    ConversationTurn,
    MemoryItem,
)
from ragforge.memory.short_term import ShortTermMemory
from ragforge.memory.store import (
    InMemoryStore,
    MemoryStore,
)


class MemoryManager:
    """
    Unified interface for short-term and long-term memory.
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        store: MemoryStore | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else MemoryConfig()
        )

        if not isinstance(
            self.config,
            MemoryConfig,
        ):
            raise TypeError(
                "config must be a MemoryConfig."
            )

        self.short_term = ShortTermMemory(
            max_turns=(
                self.config.short_term_max_turns
            )
        )

        self.long_term = LongTermMemory(
            store=(
                store
                if store is not None
                else InMemoryStore()
            ),
            max_items=(
                self.config.long_term_max_items
            ),
        )

    def add_turn(
        self,
        user_message: str,
        assistant_message: str,
    ) -> ConversationTurn:
        return self.short_term.add_turn(
            user_message=user_message,
            assistant_message=assistant_message,
        )

    def remember(
        self,
        content: str,
        memory_type: str = "fact",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        return self.long_term.add(
            content=content,
            memory_type=memory_type,
            metadata=metadata,
        )

    def recall(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[MemoryItem]:
        return self.long_term.search(
            query=query,
            top_k=top_k,
        )

    def get_conversation_context(
        self,
        limit: int | None = None,
    ) -> str:
        return self.short_term.build_context(
            limit=limit
        )

    def build_memory_context(
        self,
        query: str,
        top_k: int = 5,
        recent_turns: int | None = None,
    ) -> str:
        """
        Build a combined context from recent conversation
        and relevant long-term memories.
        """

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        sections: list[str] = []

        conversation = (
            self.get_conversation_context(
                limit=recent_turns
            )
        )

        if conversation:
            sections.append(
                "Recent Conversation:\n"
                + conversation
            )

        memories = self.recall(
            query=query,
            top_k=top_k,
        )

        if memories:
            memory_text = "\n".join(
                f"- {memory.content}"
                for memory in memories
            )

            sections.append(
                "Relevant Memories:\n"
                + memory_text
            )

        return "\n\n".join(sections)

    def clear_short_term(self) -> None:
        self.short_term.clear()

    def clear_long_term(self) -> None:
        self.long_term.clear()

    def clear_all(self) -> None:
        self.clear_short_term()
        self.clear_long_term()