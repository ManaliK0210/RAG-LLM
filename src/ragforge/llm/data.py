from pathlib import Path

import torch

from ragforge.llm.tokenizer import Tokenizer


def load_text(path: str | Path) -> str:
    """Load a text file."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Training file not found: {path}"
        )

    return path.read_text(
        encoding="utf-8"
    )


def create_tokenizer(
    text: str,
) -> Tokenizer:
    """Create a tokenizer from training text."""

    return Tokenizer(text)


def encode_text(
    tokenizer: Tokenizer,
    text: str,
) -> torch.Tensor:
    """Convert text into a PyTorch tensor."""

    token_ids = tokenizer.encode(
        text,
        add_bos=True,
        add_eos=True,
    )

    return torch.tensor(
        token_ids,
        dtype=torch.long,
    )