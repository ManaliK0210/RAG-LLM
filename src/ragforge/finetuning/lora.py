import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    """
    Linear layer with a Low-Rank Adaptation branch.

    The original linear layer remains frozen.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()

        if rank <= 0:
            raise ValueError(
                "rank must be greater than 0."
            )

        if alpha <= 0:
            raise ValueError(
                "alpha must be greater than 0."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be between 0 and 1."
            )

        self.linear = linear

        for parameter in self.linear.parameters():
            parameter.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_dropout = nn.Dropout(
            dropout
        )

        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                linear.in_features,
            )
        )

        self.lora_B = nn.Parameter(
            torch.empty(
                linear.out_features,
                rank,
            )
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(
            self.lora_A,
            a=math.sqrt(5),
        )

        nn.init.zeros_(
            self.lora_B
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        base_output = self.linear(x)

        adapted = self.lora_dropout(x)

        adapted = torch.matmul(
            adapted,
            self.lora_A.transpose(0, 1),
        )

        adapted = torch.matmul(
            adapted,
            self.lora_B.transpose(0, 1),
        )

        return (
            base_output
            + self.scaling * adapted
        )

    @property
    def trainable_parameters(self) -> int:
        return (
            self.lora_A.numel()
            + self.lora_B.numel()
        )


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """
    Count parameters participating in optimization.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def count_total_parameters(
    model: nn.Module,
) -> int:
    """
    Count all model parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def freeze_model(
    model: nn.Module,
) -> None:
    """
    Freeze every parameter in a model.
    """

    for parameter in model.parameters():
        parameter.requires_grad = False