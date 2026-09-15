import torch
import torch.nn as nn

from ragforge.llm.embeddings import InputEmbedding
from ragforge.llm.transformer import TransformerBlock


class RAGForgeLanguageModel(nn.Module):
    """
    Small GPT-style causal language model.

    The model predicts the next token at every position.

    Input:
        [batch_size, sequence_length]

    Output:
        [batch_size, sequence_length, vocab_size]
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embedding_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length

        # Token + positional embeddings
        self.embedding = InputEmbedding(
            vocab_size=vocab_size,
            max_sequence_length=context_length,
            embedding_dim=embedding_dim,
        )

        # Stack multiple Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Final normalization
        self.final_norm = nn.LayerNorm(
            embedding_dim
        )

        # Convert hidden representations into
        # vocabulary logits.
        self.language_model_head = nn.Linear(
            embedding_dim,
            vocab_size,
            bias=False,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:

        _, sequence_length = token_ids.shape

        if sequence_length > self.context_length:
            raise ValueError(
                "Sequence length cannot exceed "
                "context_length."
            )

        # ---------------------------------------------
        # Embeddings
        # ---------------------------------------------

        x = self.embedding(token_ids)

        # ---------------------------------------------
        # Transformer stack
        # ---------------------------------------------

        for block in self.transformer_blocks:
            x = block(x)

        # ---------------------------------------------
        # Final normalization
        # ---------------------------------------------

        x = self.final_norm(x)

        # ---------------------------------------------
        # Vocabulary prediction
        # ---------------------------------------------

        logits = self.language_model_head(x)

        return logits