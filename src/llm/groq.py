from __future__ import annotations

from functools import lru_cache

from langchain_groq import ChatGroq

from src.config.settings import settings


@lru_cache(maxsize=None)
def get_llm(temperature: float = 0.0) -> ChatGroq:
    """Return a cached ChatGroq (Llama) client for the configured model."""
    return ChatGroq(
        model=settings.GROQ_MODEL,
        api_key=settings.GROQ_API_KEY,
        temperature=temperature,
    )
