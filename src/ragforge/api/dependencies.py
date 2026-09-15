from typing import Any

from ragforge.memory.manager import MemoryManager


class APIContainer:
    def __init__(
        self,
        llm: Any | None = None,
        rag_pipeline: Any | None = None,
        agent: Any | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        self.llm = llm
        self.rag_pipeline = rag_pipeline
        self.agent = agent
        self.memory_manager = (
            memory_manager
            if memory_manager is not None
            else MemoryManager()
        )


_container: APIContainer | None = None


def get_container() -> APIContainer:
    global _container

    if _container is None:
        from ragforge.api.application import (
            create_api_container,
        )

        _container = create_api_container()

    return _container


def set_container(
    container: APIContainer,
) -> None:
    global _container

    if not isinstance(container, APIContainer):
        raise TypeError(
            "container must be an APIContainer."
        )

    _container = container