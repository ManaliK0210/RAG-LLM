from pathlib import Path

from ragforge.llm.config import ModelConfig
from ragforge.llm.train import train_language_model


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAINING_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "training.txt"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "ragforge_lm.pt"
)


def main() -> None:
    """Train the RAGForge language model."""

    config = ModelConfig(
        vocab_size=1000,
        context_length=128,
        embedding_dim=128,
        num_heads=4,
        num_layers=4,
        dropout=0.1,
        batch_size=32,
        learning_rate=3e-4,
        epochs=10,
        device="cpu",
    )

    print("Starting RAGForge language-model training...")
    print(f"Training data: {TRAINING_FILE}")
    print(f"Model output: {MODEL_FILE}")
    print()

    train_language_model(
        training_path=TRAINING_FILE,
        model_path=MODEL_FILE,
        config=config,
    )


if __name__ == "__main__":
    main()