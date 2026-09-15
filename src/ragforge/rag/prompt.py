from ragforge.genai.prompts import PromptTemplate


RAG_PROMPT = PromptTemplate(
    template=(
        "You are a helpful assistant that answers "
        "questions using the provided context.\n\n"
        "Use only the information contained in the "
        "context to answer the question.\n"
        "If the context does not contain enough "
        "information to answer the question, say "
        "that you do not have enough information.\n\n"
        "Context:\n"
        "{context}\n\n"
        "Question:\n"
        "{question}\n\n"
        "Answer:"
    )
)


def build_rag_prompt(
    question: str,
    context: str,
) -> str:
    """
    Build a grounded RAG prompt.
    """

    if not isinstance(question, str):
        raise TypeError(
            "question must be a string."
        )

    if not question.strip():
        raise ValueError(
            "question cannot be empty."
        )

    if not isinstance(context, str):
        raise TypeError(
            "context must be a string."
        )

    return RAG_PROMPT.format(
        context=context,
        question=question,
    )