import torch
import torch.nn as nn

from ragforge.llm.attention import MultiHeadCausalSelfAttention


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network used inside
    a Transformer block.

    The embedding dimension is expanded and then
    projected back down.

    Example:

        128 -> 512 -> 128
    """

    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.1,
        expansion_factor: int = 4,
    ):
        super().__init__()

        hidden_dim = embedding_dim * expansion_factor

        self.network = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(x)


class TransformerBlock(nn.Module):
    """
    A GPT-style Transformer block.

    Structure:

        x
        ↓
        LayerNorm
        ↓
        Multi-Head Causal Self-Attention
        ↓
        Residual Connection
        ↓
        LayerNorm
        ↓
        Feed-Forward Network
        ↓
        Residual Connection
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attention_norm = nn.LayerNorm(
            embedding_dim
        )

        self.attention = MultiHeadCausalSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.feed_forward_norm = nn.LayerNorm(
            embedding_dim
        )

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # ---------------------------------------------
        # Self-attention + residual connection
        # ---------------------------------------------

        attention_input = self.attention_norm(x)

        x = x + self.attention(
            attention_input
        )

        # ---------------------------------------------
        # Feed-forward network + residual connection
        # ---------------------------------------------

        feed_forward_input = self.feed_forward_norm(x)

        x = x + self.feed_forward(
            feed_forward_input
        )

        return x