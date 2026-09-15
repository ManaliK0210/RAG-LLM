import math

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """
    Causal self-attention for a GPT-style language model.

    Input:
        [batch_size, sequence_length, embedding_dim]

    Output:
        [batch_size, sequence_length, embedding_dim]

    Causal masking ensures that a token cannot attend
    to future tokens.
    """

    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, sequence_length, embedding_dim = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1)

        scores = scores / math.sqrt(
            embedding_dim
        )

        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf"),
        )

        attention_weights = torch.softmax(
            scores,
            dim=-1,
        )

        attention_weights = self.dropout(
            attention_weights
        )

        output = attention_weights @ v

        output = self.output_projection(output)

        return output


class MultiHeadCausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Instead of performing one attention operation over the
    complete embedding dimension, the embedding is divided
    into multiple attention heads.

    Example:

        embedding_dim = 128
        num_heads = 4

        head_dim = 128 / 4 = 32

    Input:
        [batch_size, sequence_length, embedding_dim]

    Output:
        [batch_size, sequence_length, embedding_dim]
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Produce Q, K and V for all heads at once.
        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        # Combine the outputs from all attention heads.
        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, sequence_length, _ = x.shape

        # -------------------------------------------------
        # 1. Create Q, K and V
        # -------------------------------------------------

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Current shape:
        # [batch, sequence, embedding_dim]

        # -------------------------------------------------
        # 2. Split embedding dimension into heads
        # -------------------------------------------------

        q = q.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        k = k.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        v = v.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        # Move heads before sequence dimension:
        #
        # [batch, heads, sequence, head_dim]

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # -------------------------------------------------
        # 3. Calculate attention scores
        # -------------------------------------------------

        scores = q @ k.transpose(-2, -1)

        scores = scores / math.sqrt(
            self.head_dim
        )

        # Shape:
        # [batch, heads, sequence, sequence]

        # -------------------------------------------------
        # 4. Causal mask
        # -------------------------------------------------

        mask = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
            )
        )

        scores = scores.masked_fill(
            mask == 0,
            float("-inf"),
        )

        # -------------------------------------------------
        # 5. Convert scores into probabilities
        # -------------------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=-1,
        )

        attention_weights = self.dropout(
            attention_weights
        )

        # -------------------------------------------------
        # 6. Weighted combination of values
        # -------------------------------------------------

        output = attention_weights @ v

        # Shape:
        # [batch, heads, sequence, head_dim]

        # -------------------------------------------------
        # 7. Combine attention heads
        # -------------------------------------------------

        output = output.transpose(1, 2)

        # Shape:
        # [batch, sequence, heads, head_dim]

        output = output.contiguous().view(
            batch_size,
            sequence_length,
            self.embedding_dim,
        )

        # -------------------------------------------------
        # 8. Final linear projection
        # -------------------------------------------------

        output = self.output_projection(output)

        return output