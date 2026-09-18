import torch

from ragforge.finetuning.hf_adapter import (
    load_lora_adapter,
)


def generate_with_lora(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """
    Generate text using a loaded LoRA model.
    """

    if not isinstance(
        prompt,
        str,
    ):
        raise TypeError(
            "prompt must be a string."
        )

    if not prompt.strip():
        raise ValueError(
            "prompt cannot be empty."
        )

    if max_new_tokens <= 0:
        raise ValueError(
            "max_new_tokens must be greater than 0."
        )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    device = next(
        model.parameters()
    ).device

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    model.eval()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[
        0
    ][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )


def generate_from_lora_adapter(
    base_model,
    tokenizer,
    adapter_directory,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> str:
    """Load a LoRA adapter and generate text."""

    model = load_lora_adapter(
        base_model=base_model,
        adapter_directory=adapter_directory,
    )

    return generate_with_lora(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
    )