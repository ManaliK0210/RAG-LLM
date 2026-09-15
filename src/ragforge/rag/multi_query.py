from ragforge.genai.llm import PretrainedLLM


class MultiQueryGenerator:
    """
    Generates multiple alternative formulations
    of a retrieval query.
    """

    def __init__(
        self,
        llm: PretrainedLLM,
        num_queries: int = 3,
    ) -> None:
        if num_queries <= 0:
            raise ValueError(
                "num_queries must be greater than 0."
            )

        self.llm = llm
        self.num_queries = num_queries

    def generate(
        self,
        query: str,
    ) -> list[str]:
        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        prompt = (
            "Generate alternative search queries "
            "for the following question.\n"
            f"Return exactly {self.num_queries} "
            "queries, one per line.\n\n"
            f"Question: {query}\n\n"
            "Queries:"
        )

        output = self.llm.generate(
            prompt
        )

        queries = [
            line.strip()
            for line in output.splitlines()
            if line.strip()
        ]

        if not queries:
            raise RuntimeError(
                "Multi-query generator returned "
                "no queries."
            )

        return queries[: self.num_queries]