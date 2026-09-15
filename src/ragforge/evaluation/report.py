from ragforge.evaluation.evaluator import (
    EvaluationResult,
)


class EvaluationReport:
    """
    Aggregates individual evaluation results
    into summary metrics.
    """

    def summarize(
        self,
        results: list[EvaluationResult],
    ) -> dict[str, float]:
        if not results:
            raise ValueError(
                "results cannot be empty."
            )

        summary = {
            "exact_match": self._mean(
                result.exact_match
                for result in results
            ),
            "token_f1": self._mean(
                result.token_f1
                for result in results
            ),
            "reciprocal_rank": self._mean(
                result.reciprocal_rank
                for result in results
            ),
        }

        k_values = results[0].hit_at_k.keys()

        for k in k_values:
            summary[
                f"hit_at_{k}"
            ] = self._mean(
                result.hit_at_k[k]
                for result in results
            )

            summary[
                f"recall_at_{k}"
            ] = self._mean(
                result.recall_at_k[k]
                for result in results
            )

        return summary

    @staticmethod
    def _mean(values) -> float:
        values = list(values)

        if not values:
            return 0.0

        return sum(values) / len(values)