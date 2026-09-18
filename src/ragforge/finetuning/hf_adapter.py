from pathlib import Path

from peft import PeftModel


def save_lora_adapter(
    model: PeftModel,
    output_directory: str | Path,
) -> None:
    """Save a PEFT LoRA adapter."""

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        output_directory
    )


def load_lora_adapter(
    base_model,
    adapter_directory: str | Path,
) -> PeftModel:
    """
    Load a previously trained LoRA adapter
    onto a base Hugging Face model.
    """

    adapter_directory = Path(
        adapter_directory
    )

    if not adapter_directory.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found: "
            f"{adapter_directory}"
        )

    return PeftModel.from_pretrained(
        base_model,
        adapter_directory,
    )