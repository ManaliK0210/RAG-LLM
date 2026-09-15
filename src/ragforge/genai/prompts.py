from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """
    Reusable prompt template.

    A template contains named placeholders that are
    filled with runtime values.
    """

    template: str

    def format(self, **kwargs: str) -> str:
        """
        Fill the prompt template with runtime values.
        """

        try:
            return self.template.format(**kwargs)
        except KeyError as exc:
            missing_key = exc.args[0]

            raise ValueError(
                f"Missing prompt variable: {missing_key}"
            ) from exc


class PromptManager:
    """
    Central registry for RAGForge prompt templates.
    """

    def __init__(self) -> None:
        self._templates: dict[
            str,
            PromptTemplate,
        ] = {}

    def register(
        self,
        name: str,
        template: str,
    ) -> None:
        """
        Register a named prompt template.
        """

        if not name.strip():
            raise ValueError(
                "Prompt name cannot be empty."
            )

        if not template.strip():
            raise ValueError(
                "Prompt template cannot be empty."
            )

        self._templates[name] = PromptTemplate(
            template=template,
        )

    def get(
        self,
        name: str,
    ) -> PromptTemplate:
        """
        Retrieve a registered prompt template.
        """

        if name not in self._templates:
            raise KeyError(
                f"Prompt template not found: {name}"
            )

        return self._templates[name]

    def render(
        self,
        name: str,
        **kwargs: str,
    ) -> str:
        """
        Render a registered prompt with variables.
        """

        prompt = self.get(name)

        return prompt.format(**kwargs)