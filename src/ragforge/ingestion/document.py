from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """
    Canonical representation of an ingested document.

    Attributes:
        text:
            Extracted document text.

        source:
            Original source identifier, usually a file path.

        metadata:
            Additional information about the document.
    """

    text: str
    source: str
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Validate the document."""

        if not isinstance(self.text, str):
            raise TypeError(
                "Document text must be a string."
            )

        if not self.text.strip():
            raise ValueError(
                "Document text cannot be empty."
            )

        if not isinstance(self.source, str):
            raise TypeError(
                "Document source must be a string."
            )

        if not self.source.strip():
            raise ValueError(
                "Document source cannot be empty."
            )

        if not isinstance(self.metadata, dict):
            raise TypeError(
                "Document metadata must be a dictionary."
            )