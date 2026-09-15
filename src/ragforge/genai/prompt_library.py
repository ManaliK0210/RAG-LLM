from ragforge.genai.prompts import PromptManager


def create_prompt_manager() -> PromptManager:
    """
    Create the default RAGForge prompt registry.
    """

    manager = PromptManager()

    manager.register(
        name="basic_qa",
        template=(
            "Answer the following question clearly "
            "and concisely.\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
    )

    manager.register(
        name="context_qa",
        template=(
            "Answer the question using only the "
            "provided context.\n\n"
            "Context:\n"
            "{context}\n\n"
            "Question:\n"
            "{question}\n\n"
            "Answer:"
        ),
    )

    manager.register(
        name="summarize",
        template=(
            "Summarize the following text while "
            "preserving its important information.\n\n"
            "Text:\n"
            "{text}\n\n"
            "Summary:"
        ),
    )

    return manager