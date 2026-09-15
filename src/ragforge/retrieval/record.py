from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class VectorRecord:
    """
    Stores one embedded document chunk.
    """

    id: str
    text: str
    vector: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            raise TypeError("id must be a string.")

        if not self.id.strip():
            raise ValueError("id cannot be empty.")

        if not isinstance(self.text, str):
            raise TypeError("text must be a string.")

        if not self.text.strip():
            raise ValueError("text cannot be empty.")

        if not isinstance(self.vector, np.ndarray):
            raise TypeError(
                "vector must be a NumPy array."
            )

        if self.vector.ndim != 1:
            raise ValueError(
                "vector must be one-dimensional."
            )

        if not isinstance(self.metadata, dict):
            raise TypeError(
                "metadata must be a dictionary."
            )