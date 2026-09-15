from ragforge.genai.prompts import PromptTemplate


def build_zero_shot_prompt(
    task: str,
    input_text: str,
) -> str:
    """
    Build a zero-shot prompt.
    """

    prompt = PromptTemplate(
        template=(
            "Task:\n"
            "{task}\n\n"
            "Input:\n"
            "{input_text}\n\n"
            "Response:"
        )
    )

    return prompt.format(
        task=task,
        input_text=input_text,
    )


def build_few_shot_prompt(
    task: str,
    examples: list[tuple[str, str]],
    input_text: str,
) -> str:
    """
    Build a few-shot prompt using input/output examples.
    """

    if not examples:
        raise ValueError(
            "At least one example is required "
            "for few-shot prompting."
        )

    example_sections = []

    for index, (
        example_input,
        example_output,
    ) in enumerate(
        examples,
        start=1,
    ):
        example_sections.append(
            (
                f"Example {index}:\n"
                f"Input: {example_input}\n"
                f"Output: {example_output}"
            )
        )

    examples_text = "\n\n".join(
        example_sections
    )

    prompt = PromptTemplate(
        template=(
            "Task:\n"
            "{task}\n\n"
            "Examples:\n"
            "{examples}\n\n"
            "Input:\n"
            "{input_text}\n\n"
            "Output:"
        )
    )

    return prompt.format(
        task=task,
        examples=examples_text,
        input_text=input_text,
    )


def build_structured_prompt(
    system: str,
    task: str,
    context: str,
    input_text: str,
    output_format: str,
) -> str:
    """
    Build a production-style prompt with explicit
    system, task, context, input, and output sections.
    """

    prompt = PromptTemplate(
        template=(
            "System:\n"
            "{system}\n\n"
            "Task:\n"
            "{task}\n\n"
            "Context:\n"
            "{context}\n\n"
            "Input:\n"
            "{input_text}\n\n"
            "Output format:\n"
            "{output_format}\n\n"
            "Response:"
        )
    )

    return prompt.format(
        system=system,
        task=task,
        context=context,
        input_text=input_text,
        output_format=output_format,
    )

def build_constrained_prompt(
    system: str,
    task: str,
    input_text: str,
    output_constraints: list[str],
) -> str:
    """
    Build a prompt with explicit output constraints.

    Constraints tell the model what the response must
    look like, such as length, format, or required fields.
    """

    if not output_constraints:
        raise ValueError(
            "At least one output constraint is required."
        )

    constraints_text = "\n".join(
        f"- {constraint}"
        for constraint in output_constraints
    )

    prompt = PromptTemplate(
        template=(
            "System:\n"
            "{system}\n\n"
            "Task:\n"
            "{task}\n\n"
            "Input:\n"
            "{input_text}\n\n"
            "Output constraints:\n"
            "{constraints}\n\n"
            "Response:"
        )
    )

    return prompt.format(
        system=system,
        task=task,
        input_text=input_text,
        constraints=constraints_text,
    )