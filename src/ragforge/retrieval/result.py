from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    """
    Represents one result returned by vector similarity search.
    """

    id: str
    text: str
    score: float
    metadata: dict[str, Any]