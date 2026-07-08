from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.agents.math_agent.state import MathAgentState
from src.agents.math_agent.tools import TOOLS
from src.llm.groq import get_llm

SYSTEM_PROMPT = (
    "You are a math assistant. Always use the calculator tool to compute "
    "the numeric result of a problem rather than doing arithmetic yourself, "
    "then state the final answer clearly."
)


def _call_model(state: MathAgentState) -> MathAgentState:
    llm = get_llm().bind_tools(TOOLS)
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    response = llm.invoke(messages)
    return {"messages": [response]}


def build_graph():
    graph = StateGraph(MathAgentState)
    graph.add_node("agent", _call_model)
    graph.add_node("tools", ToolNode(TOOLS))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


math_agent_graph = build_graph()


def run(query: str) -> str:
    result = math_agent_graph.invoke({"messages": [HumanMessage(content=query)]})
    return result["messages"][-1].content
