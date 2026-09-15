import json
from typing import Any


class StructuredOutputError(ValueError):
    """Raised when generated output cannot be parsed as JSON."""


def parse_json_output(
    text: str,
) -> dict[str, Any] | list[Any]:
    """
    Parse model output as JSON.

    The model is expected to return a JSON object
    or JSON array.
    """

    text = text.strip()

    if not text:
        raise StructuredOutputError(
            "Model output is empty."
        )

    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError(
            "Model output is not valid JSON."
        ) from exc

    if not isinstance(result, (dict, list)):
        raise StructuredOutputError(
            "Structured output must be a JSON object "
            "or JSON array."
        )

    return result


def extract_json(
    text: str,
) -> dict[str, Any] | list[Any]:
    """
    Extract a JSON object or array from text that may
    contain additional natural-language content.

    Example:

        Here is the result:
        {"answer": "RAG uses retrieval."}

    becomes:

        {"answer": "RAG uses retrieval."}
    """

    text = text.strip()

    if not text:
        raise StructuredOutputError(
            "Model output is empty."
        )

    # First try parsing the complete response.
    try:
        return parse_json_output(text)
    except StructuredOutputError:
        pass

    decoder = json.JSONDecoder()

    # Look for the beginning of a JSON object or array.
    candidate_positions = [
        position
        for position, character in enumerate(text)
        if character in "{["
    ]

    for position in candidate_positions:
        candidate = text[position:]

        try:
            result, _ = decoder.raw_decode(
                candidate
            )
        except json.JSONDecodeError:
            continue

        if isinstance(result, (dict, list)):
            return result

    raise StructuredOutputError(
        "Could not find valid JSON in model output."
    )