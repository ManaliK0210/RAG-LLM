from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """
    Configuration for the RAGForge embedding model.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize_embeddings: bool = True
    device: str = "cpu"