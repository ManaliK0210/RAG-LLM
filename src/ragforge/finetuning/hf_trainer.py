from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ragforge.finetuning.config import LoRAConfig
from ragforge.finetuning.hf_dataset import (
    CausalLMCollator,
    HFInstructionDataset,
)
from ragforge.finetuning.hf_lora import (
    apply_lora,
    trainable_parameter_count,
)


@dataclass(frozen=True)
class HFLoraTrainingResult:
    """Summary of a Hugging Face LoRA training run."""

    epochs: int
    final_loss: float
    trainable_parameters: int


class HFLoraTrainer:
    """
    Lightweight CPU/GPU trainer for PEFT LoRA models.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: LoRAConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.config = (
            config
            if config is not None
            else LoRAConfig()
        )

        self.device = torch.device(
            device
        )

        self.model = apply_lora(
            model,
            self.config,
        )

        self.model.to(self.device)

        self.tokenizer = tokenizer

        self.collator = CausalLMCollator(
            tokenizer,
        )

    def train(
        self,
        dataset: HFInstructionDataset,
    ) -> HFLoraTrainingResult:
        """Train the LoRA adapter."""

        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collator,
        )

        trainable_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

        if not trainable_parameters:
            raise RuntimeError(
                "No trainable LoRA parameters were found."
            )

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.config.learning_rate,
        )

        self.model.train()

        final_loss = 0.0

        for _ in range(self.config.epochs):
            epoch_loss = 0.0
            batches = 0

            for batch in loader:
                batch = {
                    key: value.to(self.device)
                    for key, value in batch.items()
                }

                optimizer.zero_grad()

                outputs = self.model(
                    **batch
                )

                loss = outputs.loss

                if loss is None:
                    raise RuntimeError(
                        "Model did not return a training loss."
                    )

                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()
                batches += 1

            if batches == 0:
                raise RuntimeError(
                    "Training dataset produced no batches."
                )

            final_loss = (
                epoch_loss / batches
            )

        return HFLoraTrainingResult(
            epochs=self.config.epochs,
            final_loss=final_loss,
            trainable_parameters=(
                trainable_parameter_count(
                    self.model
                )
            ),
        )

    def save_adapter(
        self,
        output_directory: str | Path,
    ) -> None:
        """Save the trained PEFT adapter."""

        output_directory = Path(
            output_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model.save_pretrained(
            output_directory
        )

        self.tokenizer.save_pretrained(
            output_directory
        )