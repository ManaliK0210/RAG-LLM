from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the RAGForge language model."""

    vocab_size: int = 1000
    context_length: int = 128
    embedding_dim: int = 128

    num_heads: int = 4
    num_layers: int = 4

    dropout: float = 0.1

    batch_size: int = 32
    learning_rate: float = 3e-4
    epochs: int = 10

    device: str = "cpu"