import numpy as np
from sentence_transformers import SentenceTransformer

from ragforge.embeddings.base import EmbeddingModel
from ragforge.embeddings.config import EmbeddingConfig


class SentenceTransformerEmbedding(
    EmbeddingModel
):
    """
    Sentence-Transformers based embedding model.

    The model converts text into dense semantic vectors.
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else EmbeddingConfig()
        )

        self.model = SentenceTransformer(
            self.config.model_name,
            device=self.config.device,
        )

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""

        dimension = self.model.get_embedding_dimension()

        if dimension is None:
            raise RuntimeError(
                "Unable to determine embedding dimension."
            )

        return int(dimension)

    def encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """
        Encode a batch of texts into dense vectors.
        """

        if not isinstance(texts, list):
            raise TypeError(
                "texts must be a list of strings."
            )

        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        if not all(
            isinstance(text, str)
            for text in texts
        ):
            raise TypeError(
                "All texts must be strings."
            )

        if any(
            not text.strip()
            for text in texts
        ):
            raise ValueError(
                "Texts cannot contain empty strings."
            )

        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=(
                self.config.normalize_embeddings
            ),
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )