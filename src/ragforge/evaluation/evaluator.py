from dataclasses import dataclass

from ragforge.evaluation.answer_metrics import (
    exact_match,
    token_f1,
)
from ragforge.evaluation.config import (
    EvaluationConfig,
)
from ragforge.evaluation.dataset import (
    EvaluationSample,
)
from ragforge.evaluation.retrieval_metrics import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
)
from ragforge.retrieval.result import SearchResult


@dataclass(frozen=True)
class EvaluationResult:
    """
    Evaluation metrics for one sample.
    """

    exact_match: float
    token_f1: float
    reciprocal_rank: float
    hit_at_k: dict[int, float]
    recall_at_k: dict[int, float]


class RAGEvaluator:
    """
    Evaluates both retrieval and generated answers.
    """

    def __init__(
        self,
        config: EvaluationConfig | None = None,
    ) -> None:
        self.config = (
            config
            if config is not None
            else EvaluationConfig()
        )

    def evaluate_sample(
        self,
        sample: EvaluationSample,
        results: list[SearchResult],
        prediction: str,
    ) -> EvaluationResult:
        if not isinstance(
            sample,
            EvaluationSample,
        ):
            raise TypeError(
                "sample must be an EvaluationSample."
            )

        if not isinstance(
            results,
            list,
        ):
            raise TypeError(
                "results must be a list."
            )

        if not isinstance(
            prediction,
            str,
        ):
            raise TypeError(
                "prediction must be a string."
            )

        relevant_ids = set(
            sample.relevant_document_ids
        )

        return EvaluationResult(
            exact_match=exact_match(
                prediction,
                sample.reference_answer,
            ),
            token_f1=token_f1(
                prediction,
                sample.reference_answer,
            ),
            reciprocal_rank=reciprocal_rank(
                results,
                relevant_ids,
            ),
            hit_at_k={
                k: hit_at_k(
                    results,
                    relevant_ids,
                    k,
                )
                for k in self.config.k_values
            },
            recall_at_k={
                k: recall_at_k(
                    results,
                    relevant_ids,
                    k,
                )
                for k in self.config.k_values
            },
        )

    def evaluate(
        self,
        samples: list[EvaluationSample],
        results: list[list[SearchResult]],
        predictions: list[str],
    ) -> list[EvaluationResult]:
        if not (
            len(samples)
            == len(results)
            == len(predictions)
        ):
            raise ValueError(
                "samples, results, and predictions "
                "must have the same length."
            )

        return [
            self.evaluate_sample(
                sample,
                sample_results,
                prediction,
            )
            for sample, sample_results, prediction
            in zip(
                samples,
                results,
                predictions,
            )
        ]