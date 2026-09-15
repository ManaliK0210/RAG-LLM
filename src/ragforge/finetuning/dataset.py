from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class InstructionExample:
    """
    A single instruction-tuning example.
    """

    instruction: str
    response: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.instruction,
            str,
        ):
            raise TypeError(
                "instruction must be a string."
            )

        if not self.instruction.strip():
            raise ValueError(
                "instruction cannot be empty."
            )

        if not isinstance(
            self.response,
            str,
        ):
            raise TypeError(
                "response must be a string."
            )

        if not self.response.strip():
            raise ValueError(
                "response cannot be empty."
            )


class InstructionDataset(Dataset):
    """
    PyTorch dataset for instruction tuning.

    The dataset stores tokenized input sequences and
    next-token prediction targets.
    """

    def __init__(
        self,
        examples: list[InstructionExample],
        tokenizer,
        max_length: int = 128,
    ) -> None:
        if not examples:
            raise ValueError(
                "examples cannot be empty."
            )

        if max_length <= 1:
            raise ValueError(
                "max_length must be greater than 1."
            )

        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

        self._encoded = [
            self._encode(example)
            for example in examples
        ]

    def _encode(
        self,
        example: InstructionExample,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text = (
            "Instruction:\n"
            f"{example.instruction}\n\n"
            "Response:\n"
            f"{example.response}"
        )

        token_ids = self.tokenizer.encode(
            text,
            add_bos=True,
            add_eos=True,
        )

        token_ids = token_ids[
            : self.max_length
        ]

        if len(token_ids) < 2:
            raise ValueError(
                "Encoded example must contain "
                "at least two tokens."
            )

        input_ids = torch.tensor(
            token_ids[:-1],
            dtype=torch.long,
        )

        labels = torch.tensor(
            token_ids[1:],
            dtype=torch.long,
        )

        return input_ids, labels

    def __len__(self) -> int:
        return len(self._encoded)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._encoded[index]