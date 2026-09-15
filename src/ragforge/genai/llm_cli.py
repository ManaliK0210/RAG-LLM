import argparse

from ragforge.genai.llm import (
    LLMConfig,
    PretrainedLLM,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a pretrained LLM."
    )

    parser.add_argument(
        "prompt",
        type=str,
    )

    parser.add_argument(
        "--tokens",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    config = LLMConfig(
        model_name="distilgpt2",
        device="cpu",
    )

    print("Loading pretrained LLM...")
    print(f"Model: {config.model_name}")
    print()

    llm = PretrainedLLM(config)

    generated = llm.generate(
        prompt=args.prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("Generated text:")
    print(generated)


if __name__ == "__main__":
    main()