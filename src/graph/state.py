from __future__ import annotations

from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages


class OrchestratorState(TypedDict):
    """Top-level conversation state shared across the routed sub-agents."""

    messages: Annotated[list, add_messages]
    route: Optional[Literal["search", "math"]]
