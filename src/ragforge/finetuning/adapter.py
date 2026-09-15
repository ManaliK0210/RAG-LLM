from pathlib import Path

import torch
from torch import nn


def save_lora_adapter(
    model: nn.Module,
    path: str | Path,
) -> None:
    """
    Save only LoRA parameters.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state = {
        name: parameter.detach().cpu()
        for name, parameter
        in model.named_parameters()
        if "lora_" in name
    }

    if not state:
        raise ValueError(
            "No LoRA parameters found in model."
        )

    torch.save(
        state,
        path,
    )


def load_lora_adapter(
    model: nn.Module,
    path: str | Path,
) -> None:
    """
    Load LoRA parameters into a model.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Adapter not found: {path}"
        )

    state = torch.load(
        path,
        map_location="cpu",
    )

    if not isinstance(
        state,
        dict,
    ):
        raise ValueError(
            "Invalid LoRA adapter state."
        )

    model_state = model.state_dict()

    missing = [
        name
        for name in state
        if name not in model_state
    ]

    if missing:
        raise ValueError(
            "Adapter contains parameters "
            "not present in the model."
        )

    model_state.update(state)

    model.load_state_dict(
        model_state
    )