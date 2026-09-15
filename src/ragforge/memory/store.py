from abc import ABC, abstractmethod
from typing import Any
import re

from ragforge.memory.models import MemoryItem


class MemoryStore(ABC):
    """Abstract interface for persistent memory storage."""

    @abstractmethod
    def add(self, memory: MemoryItem) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[MemoryItem]:
        raise NotImplementedError

    @abstractmethod
    def list_all(self) -> list[MemoryItem]:
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        raise NotImplementedError


class InMemoryStore(MemoryStore):
    """
    Simple in-memory memory store.

    This is intentionally backend-independent so that a
    persistent/vector implementation can replace it later.
    """

    def __init__(self) -> None:
        self._memories: list[MemoryItem] = []

    def add(self, memory: MemoryItem) -> None:
        if not isinstance(memory, MemoryItem):
            raise TypeError(
                "memory must be a MemoryItem."
            )

        self._memories.append(memory)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[MemoryItem]:
        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        if not isinstance(top_k, int):
            raise TypeError(
                "top_k must be an integer."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        def tokenize(text: str) -> set[str]:
            return set(
                re.findall(
                    r"\b\w+\b",
                    text.lower(),
                )
            )

        query_terms = tokenize(query)

        scored: list[
            tuple[int, int, MemoryItem]
        ] = []

        for index, memory in enumerate(
            self._memories
        ):
            content_terms = tokenize(
                memory.content
            )

            overlap = len(
                query_terms & content_terms
            )

            if overlap > 0:
                scored.append(
                    (
                        overlap,
                        -index,
                        memory,
                    )
                )

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        return [
            item[2]
            for item in scored[:top_k]
        ]

    def list_all(self) -> list[MemoryItem]:
        return list(self._memories)

    def clear(self) -> None:
        self._memories.clear()