from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.agents.search_agent.state import SearchAgentState
from src.agents.search_agent.tools import TOOLS
from src.llm.groq import get_llm

SYSTEM_PROMPT = (
    "You are a research assistant. Use the tavily_search tool to look up "
    "current information on the web, then answer the user's question "
    "concisely based on what you found."
)


def _call_model(state: SearchAgentState) -> SearchAgentState:
    llm = get_llm().bind_tools(TOOLS)
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = llm.invoke(messages)
    return {"messages": [response]}


def build_graph():
    graph = StateGraph(SearchAgentState)
    graph.add_node("agent", _call_model)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


search_agent_graph = build_graph()


def run(query: str) -> str:
    result = search_agent_graph.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content
