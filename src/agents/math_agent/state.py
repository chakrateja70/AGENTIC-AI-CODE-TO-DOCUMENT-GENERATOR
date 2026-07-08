from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class MathAgentState(TypedDict):
    """Conversation state for the calculator-backed math agent."""

    messages: Annotated[list, add_messages]
