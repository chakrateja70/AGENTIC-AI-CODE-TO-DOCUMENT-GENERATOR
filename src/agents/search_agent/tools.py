from __future__ import annotations

from langchain_tavily import TavilySearch

from src.config.settings import settings

tavily_search = TavilySearch(
    max_results=5,
    tavily_api_key=settings.TAVILY_API_KEY,
)

TOOLS = [tavily_search]
