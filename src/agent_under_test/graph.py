"""Agent under test: one LLM node, one tool node, final_answer routed to END.

State carries the messages plus the fields the evals read. Nodes never raise:
provider errors land in `error`."""

import time
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from src.agent_under_test.ledger import AS_OF
from src.agent_under_test.tools import load_tools
from src.llm import make_chat, model_for
from src.pricing import cost_usd

MAX_STEPS = 8
PROMPTS = Path(__file__).parent / "prompts"


class ToolCall(BaseModel):
    name: str
    args: dict
    output: str


class AgentResult(BaseModel):
    answer: str
    refused: bool = False
    error: Literal["no_final_answer", "max_steps", "provider_error"] | None = None
    tool_calls: list[ToolCall] = []
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    pricing_known: bool = True
    latency_s: float = 0.0
    model: str = ""


class State(MessagesState):
    steps: int
    error: str | None


def system_prompt() -> str:
    return (PROMPTS / "system.md").read_text().replace("{as_of}", AS_OF)


def build_graph(seed: int | None = None):
    tools = load_tools()
    llm = make_chat("agent", seed=seed).bind_tools(tools)
    executor = ToolNode([t for t in tools if t.name != "final_answer"])

    async def agent(state: State) -> dict:
        try:
            msg = await llm.ainvoke(state["messages"])
        except Exception as e:  # noqa: BLE001 — provider errors are data, not crashes
            return {
                "messages": [AIMessage(content=f"provider error: {e}")],
                "error": "provider_error",
            }
        return {"messages": [msg], "steps": state.get("steps", 0) + 1}

    def route(state: State) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        if state.get("error") or not calls or any(c["name"] == "final_answer" for c in calls):
            return END
        if state.get("steps", 0) >= MAX_STEPS:
            return END
        return "tools"

    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", executor)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route)
    g.add_edge("tools", "agent")
    return g.compile()


def _collect(state: dict, latency: float, model: str) -> AgentResult:
    msgs = state["messages"]
    tin = tout = 0
    calls: list[ToolCall] = []
    outputs = {m.tool_call_id: str(m.content) for m in msgs if isinstance(m, ToolMessage)}
    for m in msgs:
        if isinstance(m, AIMessage):
            u = m.usage_metadata or {}
            tin += u.get("input_tokens", 0)
            tout += u.get("output_tokens", 0)
            for c in m.tool_calls:
                if c["name"] != "final_answer":
                    calls.append(
                        ToolCall(name=c["name"], args=c["args"], output=outputs.get(c["id"], ""))
                    )
    cost, known = cost_usd(model, tin, tout)
    base = dict(tool_calls=calls, input_tokens=tin, output_tokens=tout, cost_usd=cost,
                pricing_known=known, latency_s=round(latency, 2), model=model)  # fmt: skip
    last = msgs[-1]
    if state.get("error"):
        return AgentResult(answer=str(last.content), error="provider_error", **base)
    finals = [c for c in getattr(last, "tool_calls", []) or [] if c["name"] == "final_answer"]
    if finals:
        a = finals[0]["args"]
        return AgentResult(
            answer=str(a.get("answer", "")), refused=bool(a.get("refused", False)), **base
        )
    if getattr(last, "tool_calls", None):
        return AgentResult(answer="", error="max_steps", **base)
    return AgentResult(answer=str(last.content), error="no_final_answer", **base)


async def run(
    question: str, graph=None, seed: int | None = None, callbacks: list | None = None
) -> AgentResult:
    graph = graph or build_graph(seed)
    t0 = time.perf_counter()
    state = await graph.ainvoke(
        {
            "messages": [SystemMessage(system_prompt()), HumanMessage(question)],
            "steps": 0,
            "error": None,
        },
        config={"recursion_limit": 2 * MAX_STEPS + 4, "callbacks": callbacks or []},
    )
    return _collect(state, time.perf_counter() - t0, model_for("agent"))
