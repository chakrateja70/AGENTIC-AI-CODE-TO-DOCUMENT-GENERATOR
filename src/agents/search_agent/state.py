from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class SearchAgentState(TypedDict):
    """Conversation state for the Tavily-backed search agent."""

    messages: Annotated[list, add_messages]
