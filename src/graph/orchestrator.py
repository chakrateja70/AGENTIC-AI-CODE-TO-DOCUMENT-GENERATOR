from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from src.agents.math_agent.graph import math_agent_graph
from src.agents.search_agent.graph import search_agent_graph
from src.graph.router import classify
from src.graph.state import OrchestratorState


def _route(state: OrchestratorState) -> OrchestratorState:
    query = state["messages"][-1].content
    return {"route": classify(query)}


def _select_agent(state: OrchestratorState) -> str:
    return "search_agent" if state["route"] == "search" else "math_agent"


def _run_search_agent(state: OrchestratorState) -> OrchestratorState:
    result = search_agent_graph.invoke({"messages": state["messages"]})
    return {"messages": [result["messages"][-1]]}


def _run_math_agent(state: OrchestratorState) -> OrchestratorState:
    result = math_agent_graph.invoke({"messages": state["messages"]})
    return {"messages": [result["messages"][-1]]}


def build_graph():
    graph = StateGraph(OrchestratorState)
    graph.add_node("router", _route)
    graph.add_node("search_agent", _run_search_agent)
    graph.add_node("math_agent", _run_math_agent)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        _select_agent,
        {"search_agent": "search_agent", "math_agent": "math_agent"},
    )
    graph.add_edge("search_agent", END)
    graph.add_edge("math_agent", END)

    return graph.compile()


orchestrator_graph = build_graph()


def run(query: str) -> str:
    result = orchestrator_graph.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content
