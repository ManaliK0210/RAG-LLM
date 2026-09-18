from typing import Any

import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from ragforge.finetuning.config import LoRAConfig


def create_lora_config(
    config: LoRAConfig,
) -> LoraConfig:
    """Convert RAGForge LoRA configuration to PEFT configuration."""

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        bias="none",
    )


def apply_lora(
    model: Any,
    config: LoRAConfig | None = None,
) -> PeftModel:
    """
    Attach trainable LoRA adapters to a Hugging Face
    causal language model.
    """

    config = (
        config
        if config is not None
        else LoRAConfig()
    )

    peft_config = create_lora_config(config)

    peft_model = get_peft_model(
        model,
        peft_config,
    )

    return peft_model


def freeze_base_parameters(
    model: nn.Module,
) -> None:
    """
    Freeze all parameters except LoRA adapter parameters.
    """

    for name, parameter in model.named_parameters():
        parameter.requires_grad = (
            "lora_" in name
        )


def trainable_parameter_count(
    model: nn.Module,
) -> int:
    """Return the number of trainable parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def total_parameter_count(
    model: nn.Module,
) -> int:
    """Return the total number of parameters."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )