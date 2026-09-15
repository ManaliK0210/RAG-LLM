from ragforge.genai.llm import PretrainedLLM


class QueryRewriter:
    """
    Rewrites a user query into a clearer retrieval query.
    """

    def __init__(
        self,
        llm: PretrainedLLM,
    ) -> None:
        self.llm = llm

    def rewrite(
        self,
        query: str,
    ) -> str:
        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        prompt = (
            "Rewrite the following question into "
            "a concise search query.\n\n"
            f"Question: {query}\n\n"
            "Search query:"
        )

        rewritten = self.llm.generate(
            prompt
        ).strip()

        if not rewritten:
            raise RuntimeError(
                "Query rewriter returned an empty result."
            )

        return rewritten