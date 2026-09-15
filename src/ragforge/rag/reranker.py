import re

from ragforge.retrieval.result import SearchResult


class KeywordReranker:
    """
    Lightweight lexical reranker.

    Scores candidates based on query-term overlap.
    """

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        if not isinstance(results, list):
            raise TypeError(
                "results must be a list."
            )

        query_terms = set(
            self._tokenize(query)
        )

        scored = []

        for result in results:
            document_terms = set(
                self._tokenize(result.text)
            )

            overlap = len(
                query_terms & document_terms
            )

            scored.append(
                (
                    overlap,
                    result.score,
                    result,
                )
            )

        scored.sort(
            key=lambda item: (
                item[0],
                item[1],
            ),
            reverse=True,
        )

        reranked = [
            item[2]
            for item in scored
        ]

        if top_k is not None:
            if top_k <= 0:
                raise ValueError(
                    "top_k must be greater than 0."
                )

            reranked = reranked[:top_k]

        return reranked

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(
            r"\b\w+\b",
            text.lower(),
        )