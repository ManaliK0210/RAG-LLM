from typing import Any


def build_tool_schema(
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build an LLM-compatible tool schema.
    """

    if not isinstance(name, str):
        raise TypeError(
            "name must be a string."
        )

    if not name.strip():
        raise ValueError(
            "name cannot be empty."
        )

    if not isinstance(
        description,
        str,
    ):
        raise TypeError(
            "description must be a string."
        )

    if not description.strip():
        raise ValueError(
            "description cannot be empty."
        )

    if parameters is None:
        parameters = {}

    if not isinstance(
        parameters,
        dict,
    ):
        raise TypeError(
            "parameters must be a dictionary."
        )

    return {
        "name": name,
        "description": description,
        "parameters": parameters,
    }