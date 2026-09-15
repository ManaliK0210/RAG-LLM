import re

from ragforge.retrieval.result import SearchResult


class HybridRetriever:
    """
    Combines semantic similarity and lexical matching.
    """

    def combine(
        self,
        query: str,
        semantic_results: list[SearchResult],
        top_k: int = 5,
        semantic_weight: float = 0.7,
    ) -> list[SearchResult]:
        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError(
                "semantic_weight must be between 0 and 1."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        query_terms = set(
            self._tokenize(query)
        )

        scored = []

        for result in semantic_results:
            document_terms = set(
                self._tokenize(result.text)
            )

            if query_terms:
                lexical_score = (
                    len(
                        query_terms
                        & document_terms
                    )
                    / len(query_terms)
                )
            else:
                lexical_score = 0.0

            combined_score = (
                semantic_weight * result.score
                + (1.0 - semantic_weight)
                * lexical_score
            )

            scored.append(
                (
                    combined_score,
                    result,
                )
            )

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            result
            for _, result in scored[:top_k]
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )