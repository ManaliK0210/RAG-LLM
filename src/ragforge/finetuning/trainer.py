from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from ragforge.finetuning.config import (
    LoRAConfig,
)
from ragforge.finetuning.lora import (
    count_trainable_parameters,
    freeze_model,
)


@dataclass
class TrainingResult:
    """
    Result of a fine-tuning run.
    """

    losses: list[float]
    trainable_parameters: int
    total_parameters: int


class LoRATrainer:
    """
    CPU-friendly trainer for LoRA-enabled models.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset,
        config: LoRAConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.dataset = dataset

        self.config = (
            config
            if config is not None
            else LoRAConfig()
        )

        self.device = torch.device(device)

        freeze_model(
            self.model
        )

        self._ensure_lora_parameters_trainable()

        self.model.to(
            self.device
        )

    def _ensure_lora_parameters_trainable(
        self,
    ) -> None:
        for name, parameter in (
            self.model.named_parameters()
        ):
            if "lora_" in name:
                parameter.requires_grad = True

    def train(self) -> TrainingResult:
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate,
        )

        trainable_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

        if not trainable_parameters:
            raise RuntimeError(
                "No trainable LoRA parameters found."
            )

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.config.learning_rate,
        )

        criterion = nn.CrossEntropyLoss()

        losses: list[float] = []

        self.model.train()

        for _ in range(
            self.config.epochs
        ):
            epoch_loss = 0.0
            batches = 0

            for input_ids, labels in dataloader:
                input_ids = input_ids.to(
                    self.device
                )

                labels = labels.to(
                    self.device
                )

                optimizer.zero_grad()

                logits = self.model(
                    input_ids
                )

                loss = criterion(
                    logits.reshape(
                        -1,
                        logits.size(-1),
                    ),
                    labels.reshape(-1),
                )

                loss.backward()

                optimizer.step()

                epoch_loss += loss.item()
                batches += 1

            losses.append(
                epoch_loss / batches
            )

        return TrainingResult(
            losses=losses,
            trainable_parameters=count_trainable_parameters(
                self.model
            ),
            total_parameters=count_trainable_parameters(
                self.model
            )
            + sum(
                parameter.numel()
                for parameter in self.model.parameters()
                if not parameter.requires_grad
            ),
        )

    @staticmethod
    def _collate(batch):
        max_length = max(
            item[0].size(0)
            for item in batch
        )

        input_ids = []
        labels = []

        for inputs, targets in batch:
            padding = max_length - inputs.size(0)

            input_ids.append(
                torch.cat(
                    [
                        inputs,
                        torch.zeros(
                            padding,
                            dtype=torch.long,
                        ),
                    ]
                )
            )

            labels.append(
                torch.cat(
                    [
                        targets,
                        torch.full(
                            (padding,),
                            -100,
                            dtype=torch.long,
                        ),
                    ]
                )
            )

        return (
            torch.stack(input_ids),
            torch.stack(labels),
        )