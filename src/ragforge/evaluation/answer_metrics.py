import re


def normalize_answer(text: str) -> str:
    """
    Normalize an answer for comparison.
    """

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string."
        )

    text = text.lower()

    text = re.sub(
        r"[^\w\s]",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def exact_match(
    prediction: str,
    reference: str,
) -> float:
    """
    Calculate normalized exact-match score.
    """

    return float(
        normalize_answer(prediction)
        == normalize_answer(reference)
    )


def token_f1(
    prediction: str,
    reference: str,
) -> float:
    """
    Calculate token-level F1 score.
    """

    prediction_tokens = normalize_answer(
        prediction
    ).split()

    reference_tokens = normalize_answer(
        reference
    ).split()

    if not prediction_tokens:
        return float(
            not reference_tokens
        )

    if not reference_tokens:
        return 0.0

    prediction_counts: dict[str, int] = {}

    for token in prediction_tokens:
        prediction_counts[token] = (
            prediction_counts.get(
                token,
                0,
            )
            + 1
        )

    reference_counts: dict[str, int] = {}

    for token in reference_tokens:
        reference_counts[token] = (
            reference_counts.get(
                token,
                0,
            )
            + 1
        )

    overlap = 0

    for token, count in prediction_counts.items():
        overlap += min(
            count,
            reference_counts.get(
                token,
                0,
            ),
        )

    if overlap == 0:
        return 0.0

    precision = (
        overlap
        / len(prediction_tokens)
    )

    recall = (
        overlap
        / len(reference_tokens)
    )

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )