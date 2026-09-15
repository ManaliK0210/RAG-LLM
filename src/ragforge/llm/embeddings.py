import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Converts token IDs into dense vector representations.

    Input:
        [batch_size, sequence_length]

    Output:
        [batch_size, sequence_length, embedding_dim]
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:

        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    """
    Learnable positional embeddings.

    Each position in the sequence gets its own
    learnable vector.
    """

    def __init__(
        self,
        max_sequence_length: int,
        embedding_dim: int,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=max_sequence_length,
            embedding_dim=embedding_dim,
        )

    def forward(
        self,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:

        positions = torch.arange(
            sequence_length,
            device=device,
        )

        return self.embedding(positions)

class InputEmbedding(nn.Module):
    """
    Combines token embeddings and positional embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        max_sequence_length: int,
        embedding_dim: int,
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.position_embedding = PositionalEmbedding(
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, sequence_length = token_ids.shape

        token_vectors = self.token_embedding(
            token_ids
        )

        position_vectors = self.position_embedding(
            sequence_length=sequence_length,
            device=token_ids.device,
        )

        position_vectors = position_vectors.unsqueeze(0)

        return token_vectors + position_vectors    