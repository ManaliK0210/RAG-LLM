from typing import Any

from ragforge.genai.llm import (
    LLMConfig,
    PretrainedLLM,
)
from ragforge.genai.prompt_library import (
    create_prompt_manager,
)
from ragforge.genai.structured import (
    extract_json,
)


class GenAIService:
    """
    High-level interface connecting prompts
    with a pretrained language model.
    """

    def __init__(
        self,
        llm_config: LLMConfig | None = None,
    ):
        if llm_config is None:
            llm_config = LLMConfig()

        self.llm = PretrainedLLM(
            config=llm_config,
        )

        self.prompts = create_prompt_manager()

    def generate(
        self,
        prompt_name: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        **prompt_variables: str,
    ) -> str:
        """
        Render a registered prompt and generate
        a text response.
        """

        prompt = self.prompts.render(
            prompt_name,
            **prompt_variables,
        )

        return self.llm.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    def generate_structured(
        self,
        prompt_name: str,
        max_new_tokens: int = 100,
        temperature: float = 0.2,
        top_k: int = 20,
        **prompt_variables: str,
    ) -> dict[str, Any] | list[Any]:
        """
        Render a registered prompt, generate a response,
        and parse the result as structured JSON.
        """

        generated_text = self.generate(
            prompt_name=prompt_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            **prompt_variables,
        )

        return extract_json(
            generated_text
        )