import re


class TextCleaner:
    """
    Cleans extracted document text before chunking.

    The cleaner removes unnecessary whitespace while
    preserving meaningful paragraph boundaries.
    """

    def clean(
        self,
        text: str,
    ) -> str:
        """
        Clean document text.
        """

        if not isinstance(text, str):
            raise TypeError(
                "Text must be a string."
            )

        text = text.strip()

        if not text:
            raise ValueError(
                "Text cannot be empty."
            )

        # Normalize spaces and tabs within lines.
        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        # Remove spaces around line breaks.
        text = re.sub(
            r" *\n *",
            "\n",
            text,
        )

        # Collapse excessive blank lines while
        # preserving paragraph separation.
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        return text.strip()