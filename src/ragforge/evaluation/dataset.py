from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationSample:
    """
    A single RAG evaluation example.
    """

    question: str
    reference_answer: str
    relevant_document_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.question, str):
            raise TypeError(
                "question must be a string."
            )

        if not self.question.strip():
            raise ValueError(
                "question cannot be empty."
            )

        if not isinstance(
            self.reference_answer,
            str,
        ):
            raise TypeError(
                "reference_answer must be a string."
            )

        if not self.reference_answer.strip():
            raise ValueError(
                "reference_answer cannot be empty."
            )

        if not self.relevant_document_ids:
            raise ValueError(
                "At least one relevant document ID "
                "is required."
            )

        if any(
            not isinstance(doc_id, str)
            or not doc_id.strip()
            for doc_id in self.relevant_document_ids
        ):
            raise ValueError(
                "Relevant document IDs must be "
                "non-empty strings."
            )