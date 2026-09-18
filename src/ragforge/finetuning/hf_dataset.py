from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class HFInstructionExample:
    """Instruction/response pair for Hugging Face fine-tuning."""

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


class HFInstructionDataset(Dataset):
    """
    Tokenized instruction dataset for causal language modeling.
    """

    def __init__(
        self,
        examples: list[HFInstructionExample],
        tokenizer,
        max_length: int = 256,
    ) -> None:
        if not isinstance(
            examples,
            list,
        ):
            raise TypeError(
                "examples must be a list."
            )

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

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError(
                    "Tokenizer must define either "
                    "pad_token_id or eos_token_id."
                )

            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        self.items = [
            self._encode(example)
            for example in examples
        ]

    def _encode(
        self,
        example: HFInstructionExample,
    ) -> dict[str, torch.Tensor]:
        text = (
            "Instruction:\n"
            f"{example.instruction}\n\n"
            "Response:\n"
            f"{example.response}"
        )

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        input_ids = encoded[
            "input_ids"
        ].squeeze(0)

        attention_mask = encoded[
            "attention_mask"
        ].squeeze(0)

        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        return self.items[index]


class CausalLMCollator:
    """
    Pads variable-length causal-LM examples into a batch.
    """

    def __init__(
        self,
        tokenizer,
    ) -> None:
        self.tokenizer = tokenizer

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError(
                    "Tokenizer must define a padding token."
                )

            tokenizer.pad_token = (
                tokenizer.eos_token
            )

    def __call__(
        self,
        examples: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        input_ids = [
            example["input_ids"]
            for example in examples
        ]

        attention_masks = [
            example["attention_mask"]
            for example in examples
        ]

        labels = [
            example["labels"]
            for example in examples
        ]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0,
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels,
        }