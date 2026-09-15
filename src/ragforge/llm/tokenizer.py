from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer
from tokenizers import models, pre_tokenizers, trainers


class Tokenizer:
    """
    BPE tokenizer used by RAGForge.

    The tokenizer learns subword units from the training corpus.

    Example:

        "unbelievable"

    may be represented using multiple subword tokens rather
    than requiring "unbelievable" to exist as one vocabulary item.
    """

    SPECIAL_TOKENS = [
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>",
    ]

    def __init__(
        self,
        text: str | None = None,
    ):
        self._tokenizer = HFTokenizer(
            models.BPE(
                unk_token="<UNK>"
            )
        )

        self._tokenizer.pre_tokenizer = (
            pre_tokenizers.Whitespace()
        )

        if text:
            self._train(text)

    def _train(
        self,
        text: str,
    ) -> None:
        """Train the BPE vocabulary."""

        trainer = trainers.BpeTrainer(
            vocab_size=1000,
            special_tokens=self.SPECIAL_TOKENS,
        )

        self._tokenizer.train_from_iterator(
            [text],
            trainer=trainer,
        )

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""

        return self._tokenizer.get_vocab_size()

    @property
    def token_to_id(self) -> dict[str, int]:
        """Return token → ID mapping."""

        return self._tokenizer.get_vocab()

    @property
    def id_to_token(self) -> dict[int, str]:
        """Return ID → token mapping."""

        return {
            token_id: self._tokenizer.id_to_token(token_id)
            for token_id in range(
                self.vocab_size
            )
        }

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Convert text into token IDs."""

        encoding = self._tokenizer.encode(
            text
        )

        token_ids = encoding.ids

        if add_bos:
            token_ids.insert(
                0,
                self._tokenizer.token_to_id(
                    "<BOS>"
                ),
            )

        if add_eos:
            token_ids.append(
                self._tokenizer.token_to_id(
                    "<EOS>"
                )
            )

        return token_ids

    def decode(
        self,
        token_ids: list[int],
    ) -> str:
        """Convert token IDs back into text."""

        return self._tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        )

    def save(
        self,
        path: str | Path,
    ) -> None:
        """Save the trained tokenizer."""

        self._tokenizer.save(
            str(path)
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "Tokenizer":
        """Load a previously trained tokenizer."""

        tokenizer = cls()

        tokenizer._tokenizer = (
            HFTokenizer.from_file(
                str(path)
            )
        )

        return tokenizer