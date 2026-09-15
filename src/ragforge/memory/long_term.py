from typing import Any

from ragforge.memory.models import MemoryItem
from ragforge.memory.store import MemoryStore


class LongTermMemory:
    """Manages persistent memory items."""

    def __init__(
        self,
        store: MemoryStore,
        max_items: int = 100,
    ) -> None:
        if not isinstance(store, MemoryStore):
            raise TypeError(
                "store must implement MemoryStore."
            )

        if max_items <= 0:
            raise ValueError(
                "max_items must be greater than 0."
            )

        self.store = store
        self.max_items = max_items

    def add(
        self,
        content: str,
        memory_type: str = "fact",
        metadata: dict[str, Any] | None = None,
    ) -> MemoryItem:
        memory = MemoryItem(
            content=content,
            memory_type=memory_type,
            metadata=(
                metadata
                if metadata is not None
                else {}
            ),
        )

        self.store.add(memory)

        self._enforce_limit()

        return memory

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[MemoryItem]:
        return self.store.search(
            query=query,
            top_k=top_k,
        )

    def list_all(self) -> list[MemoryItem]:
        return self.store.list_all()

    def clear(self) -> None:
        self.store.clear()

    def _enforce_limit(self) -> None:
        memories = self.store.list_all()

        if len(memories) <= self.max_items:
            return

        excess = len(memories) - self.max_items

        # Keep the newest memories.
        memories = memories[excess:]

        self.store.clear()

        for memory in memories:
            self.store.add(memory)