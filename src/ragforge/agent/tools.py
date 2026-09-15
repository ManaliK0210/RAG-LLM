import ast
import operator

from ragforge.agent.tool import (
    Tool,
    ToolResult,
)
from ragforge.agent.tool_schema import (
    build_tool_schema,
)


class CalculatorTool(Tool):
    """
    Safely evaluates basic arithmetic expressions.
    """

    name = "calculator"

    description = (
        "Calculate a basic arithmetic expression."
    )

    _operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
    }

    def run(
        self,
        expression: str,
    ) -> ToolResult:
        if not isinstance(
            expression,
            str,
        ):
            raise TypeError(
                "expression must be a string."
            )

        if not expression.strip():
            raise ValueError(
                "expression cannot be empty."
            )

        try:
            tree = ast.parse(
                expression,
                mode="eval",
            )

            result = self._evaluate(
                tree.body
            )

            return ToolResult(
                tool_name=self.name,
                output=result,
            )

        except (
            SyntaxError,
            ValueError,
            TypeError,
            ZeroDivisionError,
        ) as exc:
            return ToolResult(
                tool_name=self.name,
                output=None,
                success=False,
                error=str(exc),
            )

    def schema(self):
        return build_tool_schema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "Arithmetic expression "
                            "to calculate."
                        ),
                    }
                },
                "required": [
                    "expression"
                ],
            },
        )

    @classmethod
    def _evaluate(
        cls,
        node,
    ):
        if isinstance(
            node,
            ast.Constant,
        ):
            if isinstance(
                node.value,
                bool,
            ):
                raise ValueError(
                    "Boolean values are not allowed."
                )

            if not isinstance(
                node.value,
                (int, float),
            ):
                raise ValueError(
                    "Only numeric values are allowed."
                )

            return node.value

        if isinstance(
            node,
            ast.UnaryOp,
        ):
            operation = cls._operators.get(
                type(node.op)
            )

            if operation is None:
                raise ValueError(
                    "Unsupported unary operator."
                )

            return operation(
                cls._evaluate(node.operand)
            )

        if isinstance(
            node,
            ast.BinOp,
        ):
            operation = cls._operators.get(
                type(node.op)
            )

            if operation is None:
                raise ValueError(
                    "Unsupported binary operator."
                )

            return operation(
                cls._evaluate(node.left),
                cls._evaluate(node.right),
            )

        raise ValueError(
            "Unsupported expression."
        )


class TextSearchTool(Tool):
    """
    Simple in-memory keyword search tool.

    This provides a generic search interface that
    can later be replaced by the RAG retriever.
    """

    name = "text_search"

    description = (
        "Search a collection of text documents "
        "using keyword matching."
    )

    def __init__(
        self,
        documents: list[str],
    ) -> None:
        if not documents:
            raise ValueError(
                "documents cannot be empty."
            )

        self.documents = documents

    def run(
        self,
        query: str,
    ) -> ToolResult:
        if not isinstance(
            query,
            str,
        ):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        normalized_query = " ".join(
            query.lower().split()
        )

        matches = [
            document
            for document in self.documents
            if normalized_query
            in " ".join(document.lower().split())
        ]

        return ToolResult(
            tool_name=self.name,
            output=matches,
        )

    def schema(self):
        return build_tool_schema(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Text to search for."
                        ),
                    }
                },
                "required": [
                    "query"
                ],
            },
        )