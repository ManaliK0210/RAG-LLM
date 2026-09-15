from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ragforge.llm.config import ModelConfig
from ragforge.llm.data import create_tokenizer, encode_text, load_text
from ragforge.llm.dataset import LanguageModelDataset
from ragforge.llm.model import RAGForgeLanguageModel


def train_language_model(
    training_path: str | Path,
    model_path: str | Path = "models/ragforge_lm.pt",
    tokenizer_path: str | Path = "models/tokenizer.json",
    config: ModelConfig | None = None,
) -> float:
    """
    Train the RAGForge language model and save its
    tokenizer alongside the model checkpoint.
    """

    if config is None:
        config = ModelConfig()

    # -------------------------------------------------
    # 1. Load training text
    # -------------------------------------------------

    text = load_text(training_path)

    # -------------------------------------------------
    # 2. Train tokenizer on the training corpus
    # -------------------------------------------------

    tokenizer = create_tokenizer(text)

    # -------------------------------------------------
    # 3. Encode training text
    # -------------------------------------------------

    token_ids = encode_text(
        tokenizer,
        text,
    )

    # -------------------------------------------------
    # 4. Create causal LM dataset
    # -------------------------------------------------

    dataset = LanguageModelDataset(
        token_ids=token_ids,
        context_length=config.context_length,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    # -------------------------------------------------
    # 5. Create model
    # -------------------------------------------------

    model = RAGForgeLanguageModel(
        vocab_size=tokenizer.vocab_size,
        context_length=config.context_length,
        embedding_dim=config.embedding_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )

    device = torch.device(config.device)

    model = model.to(device)

    # -------------------------------------------------
    # 6. Optimizer
    # -------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    # -------------------------------------------------
    # 7. Loss
    # -------------------------------------------------

    loss_function = nn.CrossEntropyLoss()

    # -------------------------------------------------
    # 8. Training
    # -------------------------------------------------

    model.train()

    final_loss = 0.0

    for epoch in range(config.epochs):

        total_loss = 0.0

        for input_ids, target_ids in dataloader:

            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optimizer.zero_grad()

            logits = model(input_ids)

            logits = logits.reshape(
                -1,
                tokenizer.vocab_size,
            )

            target_ids = target_ids.reshape(-1)

            loss = loss_function(
                logits,
                target_ids,
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        final_loss = (
            total_loss / len(dataloader)
        )

        print(
            f"Epoch {epoch + 1}/{config.epochs} "
            f"- Loss: {final_loss:.4f}"
        )

    # -------------------------------------------------
    # 9. Save tokenizer
    # -------------------------------------------------

    tokenizer_path = Path(
        tokenizer_path
    )

    tokenizer_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer.save(tokenizer_path)

    # -------------------------------------------------
    # 10. Save model checkpoint
    # -------------------------------------------------

    model_path = Path(model_path)

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": tokenizer.vocab_size,
            "config": config.__dict__,
        },
        model_path,
    )

    print(
        f"Model saved to: {model_path}"
    )

    print(
        f"Tokenizer saved to: {tokenizer_path}"
    )

    return final_loss