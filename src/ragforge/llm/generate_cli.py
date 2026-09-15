import argparse
from pathlib import Path

from ragforge.llm.generate import generate_from_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "ragforge_lm.pt"
)

TOKENIZER_FILE = (
    PROJECT_ROOT
    / "models"
    / "tokenizer.json"
)


def main() -> None:
    """Generate text with the RAGForge model."""

    parser = argparse.ArgumentParser(
        description="Generate text with RAGForge."
    )

    parser.add_argument(
        "prompt",
        type=str,
        help="Text prompt to start generation.",
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=30,
        help="Number of new tokens.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of highest-probability tokens to sample from.",
    )

    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Use greedy decoding instead of sampling.",
    )

    args = parser.parse_args()

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_FILE}"
        )

    if not TOKENIZER_FILE.exists():
        raise FileNotFoundError(
            f"Tokenizer not found: {TOKENIZER_FILE}"
        )

    generated_text = generate_from_checkpoint(
        model_path=MODEL_FILE,
        tokenizer_path=TOKENIZER_FILE,
        prompt=args.prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        greedy=args.greedy,
    )

    print()
    print("Generated text:")
    print(generated_text)


if __name__ == "__main__":
    main()