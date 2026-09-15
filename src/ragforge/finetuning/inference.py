import torch
from torch import nn


@torch.no_grad()
def generate_with_model(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 20,
) -> torch.Tensor:
    """
    Generate tokens autoregressively from a model.
    """

    if max_new_tokens <= 0:
        raise ValueError(
            "max_new_tokens must be greater than 0."
        )

    model.eval()

    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        logits = model(
            generated
        )

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )

        generated = torch.cat(
            [
                generated,
                next_token,
            ],
            dim=1,
        )

    return generated