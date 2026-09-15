from typing import Tuple

import torch
from torch.utils.data import Dataset


class LanguageModelDataset(Dataset):
    """
    Dataset for causal language-model training.

    Given a sequence of token IDs:

        [10, 20, 30, 40, 50]

    and a context length of 4, it produces:

        input:  [10, 20, 30, 40]
        target: [20, 30, 40, 50]

    The target is shifted one position to the right because
    the model learns to predict the next token.
    """

    def __init__(
        self,
        token_ids: torch.Tensor,
        context_length: int,
    ):
        if token_ids.ndim != 1:
            raise ValueError(
                "token_ids must be a 1D tensor."
            )

        if len(token_ids) <= context_length:
            raise ValueError(
                "Token sequence must be longer than "
                "context_length."
            )

        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self) -> int:
        """Number of training examples."""

        return len(self.token_ids) - self.context_length

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one input-target pair."""

        input_ids = self.token_ids[
            index:index + self.context_length
        ]

        target_ids = self.token_ids[
            index + 1:index + self.context_length + 1
        ]

        return input_ids, target_ids