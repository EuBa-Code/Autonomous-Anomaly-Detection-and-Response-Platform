"""
tools.py
---------
LangGraph-compatible tools that call the MCP Server REST API.

Each tool is a plain Python function decorated with @tool (langchain_core).
The MCP server is responsible for:
  - Query embedding (dense + sparse)
  - Hybrid search
  - Re-ranking
  - Returning contexts

These tools are injected into the LangGraph agent node.
"""

import logging
from functools import lru_cache
from typing import Annotated

import httpx
from langchain_core.tools import tool

from config import get_config

logger = logging.getLogger(__name__)


# ── HTTP client ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_http_client() -> httpx.Client:
    cfg = get_config()
    return httpx.Client(
        base_url=cfg["mcp_server"]["base_url"],
        timeout=cfg["mcp_server"]["timeout_seconds"],
    )


# ── Tool 1: retrieve_context ─────────────────────────────────────────────────

@tool
def retrieve_context(
    query: Annotated[str, "Natural-language query describing what to look up"],
    top_k: Annotated[int, "Number of context chunks to retrieve (default 5)"] = 5,
) -> str:
    """
    Retrieve relevant context from the knowledge base via the MCP server.
    Uses hybrid search (dense + sparse) with re-ranking.
    Returns a numbered list of context passages.
    """
    cfg = get_config()
    endpoint = cfg["mcp_server"]["endpoints"]["retrieve_context"]

    try:
        client = _get_http_client()
        response = client.post(
            endpoint,
            json={"query": query, "top_k": top_k},
        )
        response.raise_for_status()
        data = response.json()

        contexts: list[str] = data.get("contexts", [])
        scores: list[float] = data.get("scores", [])

        if not contexts:
            return "No relevant context found."

        formatted = "\n\n".join(
            f"[{i+1}] (score={scores[i]:.4f})\n{ctx}"
            for i, ctx in enumerate(contexts)
        )
        logger.info("retrieve_context: returned %d chunks for query='%s'", len(contexts), query[:80])
        return formatted

    except httpx.HTTPStatusError as exc:
        logger.error("MCP server returned HTTP %s: %s", exc.response.status_code, exc.response.text)
        return f"Error calling MCP server: HTTP {exc.response.status_code}"
    except Exception as exc:
        logger.exception("Unexpected error in retrieve_context")
        return f"Error: {exc}"


# ── Tool 2: explain_anomaly ──────────────────────────────────────────────────

@tool
def explain_anomaly(
    entity_id: Annotated[str, "The entity (user/transaction) ID that triggered the anomaly"],
    prediction_score: Annotated[float, "The anomaly score from the ML model"],
    features: Annotated[str, "JSON string of feature name→value pairs that the model used"],
) -> str:
    """
    Build a structured explanation query and retrieve relevant policy / pattern context
    to help explain *why* this entity is anomalous.
    Returns retrieved passages relevant to the anomaly pattern.
    """
    query = (
        f"Anomaly detected for entity {entity_id} with score {prediction_score:.4f}. "
        f"Key features: {features}. "
        "What patterns, policies, or known fraud indicators match this behaviour?"
    )
    return retrieve_context.invoke({"query": query, "top_k": 5})


# ── Tool 3: lookup_entity_history ────────────────────────────────────────────

@tool
def lookup_entity_history(
    entity_id: Annotated[str, "The entity ID to look up"],
) -> str:
    """
    Retrieve historical context and past incidents related to a specific entity.
    """
    query = f"Historical behaviour, past incidents, and risk profile of entity {entity_id}."
    return retrieve_context.invoke({"query": query, "top_k": 3})


# ── Tool 4: get_remediation_actions ──────────────────────────────────────────

@tool
def get_remediation_actions(
    anomaly_type: Annotated[str, "A brief description of the anomaly type or category"],
) -> str:
    """
    Retrieve recommended remediation or escalation actions for a given anomaly type.
    """
    query = f"Recommended actions, escalation procedures, and remediation steps for: {anomaly_type}."
    return retrieve_context.invoke({"query": query, "top_k": 3})


# ── Exported tool list ────────────────────────────────────────────────────────

TOOLS = [
    retrieve_context,
    explain_anomaly,
    lookup_entity_history,
    get_remediation_actions,
]