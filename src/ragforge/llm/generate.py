from pathlib import Path

import torch

from ragforge.llm.config import ModelConfig
from ragforge.llm.model import RAGForgeLanguageModel
from ragforge.llm.tokenizer import Tokenizer

def load_model(
    model_path: str | Path,
    tokenizer_path: str | Path,
):
    """
    Load a trained RAGForge model and its tokenizer.
    """

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    config = ModelConfig(
        **checkpoint["config"]
    )

    tokenizer = Tokenizer.load(
        tokenizer_path
    )

    model = RAGForgeLanguageModel(
        vocab_size=checkpoint["vocab_size"],
        context_length=config.context_length,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, tokenizer, config

def select_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
) -> torch.Tensor:
    """
    Select the next token from model logits.

    Supports:

    - Greedy decoding
    - Temperature sampling
    - Top-k sampling
    """

    if temperature <= 0:
        raise ValueError(
            "temperature must be greater than zero."
        )

    # -------------------------------------------------
    # Greedy decoding
    # -------------------------------------------------

    if greedy:
        return torch.argmax(
            logits,
            dim=-1,
            keepdim=True,
        )

    # -------------------------------------------------
    # Temperature
    # -------------------------------------------------

    logits = logits / temperature

    # -------------------------------------------------
    # Top-k filtering
    # -------------------------------------------------

    if top_k is not None:

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        top_k = min(
            top_k,
            logits.size(-1),
        )

        values, _ = torch.topk(
            logits,
            top_k,
            dim=-1,
        )

        minimum_value = values[
            :, -1
        ].unsqueeze(-1)

        logits = torch.where(
            logits < minimum_value,
            torch.full_like(
                logits,
                float("-inf"),
            ),
            logits,
        )

    # -------------------------------------------------
    # Convert logits to probabilities
    # -------------------------------------------------

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )

    # -------------------------------------------------
    # Sample
    # -------------------------------------------------

    return torch.multinomial(
        probabilities,
        num_samples=1,
    )


@torch.no_grad()
def generate_text(
    model: RAGForgeLanguageModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 30,
    temperature: float = 0.8,
    top_k: int | None = 10,
    greedy: bool = False,
) -> str:
    """
    Generate text from a trained language model.
    """

    if not prompt.strip():
        raise ValueError(
            "Prompt must contain at least one token."
        )

    token_ids = tokenizer.encode(
        prompt
    )

    if not token_ids:
        raise ValueError(
            "Prompt must contain at least one token."
        )

    input_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
    )

    for _ in range(max_new_tokens):

        # Keep only the supported context.
        input_context = input_ids[
            :, -model.context_length:
        ]

        # Predict next-token logits.
        logits = model(
            input_context
        )

        # We only need predictions for
        # the final position.
        next_token_logits = logits[
            :, -1, :
        ]

        # Select the next token.
        next_token = select_next_token(
            logits=next_token_logits,
            temperature=temperature,
            top_k=top_k,
            greedy=greedy,
        )

        # Append selected token.
        input_ids = torch.cat(
            [
                input_ids,
                next_token,
            ],
            dim=1,
        )

    return tokenizer.decode(
        input_ids[0].tolist()
    )

def generate_from_checkpoint(
    model_path: str | Path,
    tokenizer_path: str | Path,
    prompt: str,
    max_new_tokens: int = 30,
    temperature: float = 0.8,
    top_k: int | None = 10,
    greedy: bool = False,
) -> str:
    """
    Load a model and tokenizer, then generate text.
    """

    model, tokenizer, _ = load_model(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )

    return generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        greedy=greedy,
    )