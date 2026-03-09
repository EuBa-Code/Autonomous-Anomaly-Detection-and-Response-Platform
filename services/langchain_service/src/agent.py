"""
agent.py
---------
LangGraph ReAct agent that orchestrates anomaly investigation.

Flow (matching architecture diagram):
  Tools → Agent workflow → Thread → (back to tools if needed) → final answer

The LLM is served by vLLM using the OpenAI-compatible endpoint.
Tool calls go to the MCP Server via tools.py.
"""

import json
import logging
import uuid
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from .config import get_config
from .tools import TOOLS

logger = logging.getLogger(__name__)


# ── Agent state ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    entity_id: str
    prediction_score: float
    features: dict[str, Any]
    retrieved_contexts: list[str]


# ── LLM (vLLM via OpenAI-compatible API) ─────────────────────────────────────

def _build_llm() -> ChatOpenAI:
    cfg = get_config()
    vcfg = cfg["vllm"]
    return ChatOpenAI(
        base_url=vcfg["base_url"],
        model=vcfg["model"],
        temperature=vcfg["temperature"],
        max_tokens=vcfg["max_tokens"],
        api_key="not-needed",           # vLLM doesn't require a real key
    ).bind_tools(TOOLS)


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert anomaly investigation agent.
Your job is to analyse anomalies detected by an ML model and produce a clear, 
structured investigation report.

You have access to the following tools:
- retrieve_context: general knowledge-base lookup
- explain_anomaly: explains why an entity is anomalous given its features
- lookup_entity_history: retrieves past behaviour of an entity
- get_remediation_actions: recommends actions to take for a given anomaly type

Always:
1. Start by calling explain_anomaly to understand the detected pattern.
2. Call lookup_entity_history to check the entity's past.
3. Call get_remediation_actions once you know the anomaly type.
4. Synthesise all retrieved context into a concise investigation report.

Your final answer must include:
- Anomaly summary (what was detected)
- Root cause analysis (why it triggered)
- Entity risk profile (history)
- Recommended actions (what to do next)
"""


# ── Graph nodes ───────────────────────────────────────────────────────────────

def agent_node(state: AgentState, llm: ChatOpenAI) -> dict:
    """Call the LLM — may produce a tool call or a final answer."""
    response = llm.invoke(state["messages"])
    logger.debug("agent_node: response type=%s", type(response).__name__)
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """Execute whatever tool the LLM requested."""
    tool_map = {t.name: t for t in TOOLS}
    last_message: AIMessage = state["messages"][-1]

    tool_messages = []
    collected_contexts: list[str] = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        logger.info("Calling tool '%s' with args: %s", tool_name, tool_args)

        if tool_name not in tool_map:
            result = f"Unknown tool: {tool_name}"
        else:
            try:
                result = tool_map[tool_name].invoke(tool_args)
            except Exception as exc:
                logger.exception("Tool '%s' raised an exception", tool_name)
                result = f"Tool error: {exc}"

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )
        collected_contexts.append(str(result))

    return {
        "messages": tool_messages,
        "retrieved_contexts": state.get("retrieved_contexts", []) + collected_contexts,
    }


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Route: if the last message has tool calls → execute them; else → done."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> tuple[Any, MemorySaver]:
    """
    Builds and compiles the LangGraph agent graph.

    Returns
    -------
    graph : CompiledGraph
    memory : MemorySaver  (in-memory checkpointer — swap for Redis in production)
    """
    llm = _build_llm()
    memory = MemorySaver()

    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("agent", lambda s: agent_node(s, llm))
    builder.add_node("tools", tool_node)

    # Edges
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    builder.add_edge("tools", "agent")    # loop back after tool execution

    cfg = get_config()
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=["tools"],   # uncomment for human-in-the-loop
    )
    logger.info("LangGraph agent compiled successfully")
    return graph, memory


# ── Public interface ──────────────────────────────────────────────────────────

class MCPClientAgent:
    """Thread-safe wrapper around the compiled LangGraph graph."""

    def __init__(self) -> None:
        self.graph, self.memory = build_graph()
        self._cfg = get_config()

    def investigate(
        self,
        entity_id: str,
        prediction_score: float,
        features: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Run a full anomaly investigation and return the result.

        Parameters
        ----------
        entity_id        : unique identifier of the anomalous entity
        prediction_score : score from the ML model
        features         : feature dict that caused the anomaly
        metadata         : optional extra context
        thread_id        : if provided, resumes an existing conversation thread

        Returns
        -------
        dict with keys: thread_id, conclusion, messages, retrieved_contexts
        """
        if thread_id is None:
            prefix = self._cfg["langgraph"]["thread_id_prefix"]
            thread_id = f"{prefix}-{uuid.uuid4().hex[:8]}"

        # Build the initial human message
        user_content = (
            f"Investigate anomaly for entity_id='{entity_id}'.\n"
            f"Prediction score: {prediction_score:.4f}\n"
            f"Features: {json.dumps(features, indent=2)}\n"
        )
        if metadata:
            user_content += f"Metadata: {json.dumps(metadata, indent=2)}\n"

        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ],
            "entity_id": entity_id,
            "prediction_score": prediction_score,
            "features": features,
            "retrieved_contexts": [],
        }

        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self._cfg["langgraph"]["recursion_limit"],
        }

        logger.info("Starting investigation thread_id=%s entity=%s", thread_id, entity_id)

        final_state = self.graph.invoke(initial_state, config=config)

        # Extract final assistant message
        conclusion = ""
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                conclusion = msg.content
                break

        serialized_messages = [
            {"role": self._msg_role(m), "content": str(m.content)}
            for m in final_state["messages"]
        ]

        logger.info("Investigation complete thread_id=%s", thread_id)

        return {
            "thread_id": thread_id,
            "entity_id": entity_id,
            "conclusion": conclusion,
            "messages": serialized_messages,
            "retrieved_contexts": final_state.get("retrieved_contexts", []),
            "status": "completed",
        }

    @staticmethod
    def _msg_role(msg) -> str:
        if isinstance(msg, HumanMessage):
            return "user"
        if isinstance(msg, AIMessage):
            return "assistant"
        if isinstance(msg, SystemMessage):
            return "system"
        if isinstance(msg, ToolMessage):
            return "tool"
        return "unknown"