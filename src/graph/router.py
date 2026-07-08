from __future__ import annotations

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.llm.groq import get_llm


class Route(BaseModel):
    """The sub-agent that should handle the user's request."""

    destination: Literal["search", "math"] = Field(
        description=(
            "'search' when the request needs current facts, news, or anything "
            "requiring a web lookup. 'math' when the request needs a numeric "
            "calculation or arithmetic."
        )
    )


_ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Classify the user's request into exactly one route: "
            "'search' (needs up-to-date facts or a web lookup) or "
            "'math' (needs a numeric calculation).",
        ),
        ("human", "{query}"),
    ]
)


def classify(query: str) -> Literal["search", "math"]:
    router = _ROUTER_PROMPT | get_llm().with_structured_output(Route)
    route: Route = router.invoke({"query": query})
    return route.destination
