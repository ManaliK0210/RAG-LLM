from ragforge.memory.models import ConversationTurn


class ShortTermMemory:
    """Stores recent conversation turns."""

    def __init__(
        self,
        max_turns: int = 10,
    ) -> None:
        if max_turns <= 0:
            raise ValueError(
                "max_turns must be greater than 0."
            )

        self.max_turns = max_turns
        self._turns: list[
            ConversationTurn
        ] = []

    def add_turn(
        self,
        user_message: str,
        assistant_message: str,
    ) -> ConversationTurn:
        turn = ConversationTurn(
            user_message=user_message,
            assistant_message=assistant_message,
        )

        self._turns.append(turn)

        if len(self._turns) > self.max_turns:
            self._turns = self._turns[
                -self.max_turns:
            ]

        return turn

    def get_recent(
        self,
        limit: int | None = None,
    ) -> list[ConversationTurn]:
        if limit is not None:
            if limit <= 0:
                raise ValueError(
                    "limit must be greater than 0."
                )

            return self._turns[-limit:]

        return list(self._turns)

    def build_context(
        self,
        limit: int | None = None,
    ) -> str:
        turns = self.get_recent(limit)

        if not turns:
            return ""

        sections = []

        for turn in turns:
            sections.append(
                (
                    f"User: {turn.user_message}\n"
                    f"Assistant: {turn.assistant_message}"
                )
            )

        return "\n\n".join(sections)

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)