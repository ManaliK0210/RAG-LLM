from ragforge.retrieval.result import SearchResult


def hit_at_k(
    results: list[SearchResult],
    relevant_document_ids: set[str],
    k: int,
) -> float:
    """
    Return 1.0 if at least one relevant document
    appears in the top-k results.
    """

    if k <= 0:
        raise ValueError(
            "k must be greater than 0."
        )

    if not isinstance(results, list):
        raise TypeError(
            "results must be a list."
        )

    if not isinstance(
        relevant_document_ids,
        set,
    ):
        raise TypeError(
            "relevant_document_ids must be a set."
        )

    retrieved_ids = {
        result.id
        for result in results[:k]
    }

    return float(
        bool(
            retrieved_ids
            & relevant_document_ids
        )
    )


def recall_at_k(
    results: list[SearchResult],
    relevant_document_ids: set[str],
    k: int,
) -> float:
    """
    Calculate recall@k.
    """

    if k <= 0:
        raise ValueError(
            "k must be greater than 0."
        )

    if not relevant_document_ids:
        raise ValueError(
            "relevant_document_ids cannot be empty."
        )

    if not isinstance(results, list):
        raise TypeError(
            "results must be a list."
        )

    retrieved_ids = {
        result.id
        for result in results[:k]
    }

    relevant_retrieved = (
        retrieved_ids
        & relevant_document_ids
    )

    return (
        len(relevant_retrieved)
        / len(relevant_document_ids)
    )


def reciprocal_rank(
    results: list[SearchResult],
    relevant_document_ids: set[str],
) -> float:
    """
    Calculate reciprocal rank of the first
    relevant retrieved document.

    Example:
        relevant document at rank 1 -> 1.0
        relevant document at rank 2 -> 0.5
        relevant document at rank 3 -> 0.333...
    """

    if not relevant_document_ids:
        raise ValueError(
            "relevant_document_ids cannot be empty."
        )

    if not isinstance(results, list):
        raise TypeError(
            "results must be a list."
        )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        if result.id in relevant_document_ids:
            return 1.0 / rank

    return 0.0