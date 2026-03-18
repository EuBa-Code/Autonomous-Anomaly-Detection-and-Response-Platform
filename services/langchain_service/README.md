# LangChain Service

## Overview

FastAPI service that hosts the **LangGraph ReAct agent** — the reasoning engine that investigates anomalies. It receives an anomaly description from the `if_anomaly_service`, calls the MCP server's `retrieve_context` tool to fetch relevant knowledge-base context, streams the agent's response token by token via SSE, and sends the final investigation summary to the operator via Slack.

## File Structure

```
services/langchain_service/
├── Dockerfile
├── config/
│   ├── __init__.py
│   └── config.py           # MCP server URI, vLLM URL, model name, Slack webhook
└── src/
    ├── __init__.py
    ├── app.py               # FastAPI app — /health + /chat/stream endpoint
    ├── agent.py             # LangGraph ReAct agent builder (MCP tools + vLLM)
    ├── notifier.py          # Slack webhook notification
    └── schemas.py           # ChatRequest Pydantic model
```

## End-to-End Flow

```
if_anomaly_service
  POST /chat/stream  {message, machine_id}
        │
        ▼
  FastAPI app.py
        │
        ├─► get_agent()  (lazy init, cached after first call)
        │       │
        │       ├─ MultiServerMCPClient  →  mcp_server:8020/mcp
        │       └─ ChatOpenAI  →  vllm:8000/v1  (Qwen2.5-7B-Instruct-GPTQ-Int4)
        │
        ├─► agent.astream_events()
        │       │
        │       ├─ on_tool_start  →  SSE event  "tool_start"
        │       ├─ on_tool_end    →  SSE event  "tool_end"
        │       └─ on_chat_model_stream → SSE event "token"  (streamed text)
        │
        ├─► SSE event "done"
        │
        └─► notify_operator(machine_id, full_response)
                │
                └─► Slack webhook  (fire-and-forget, non-blocking)
```

## Agent (`agent.py`)

A LangGraph **ReAct agent** built with `create_react_agent`:

| Component | Value |
|---|---|
| LLM | `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` via `ChatOpenAI` pointing at vLLM |
| Tools | `retrieve_context` loaded from the MCP server via `MultiServerMCPClient` |
| Transport | `streamable_http` to `http://mcp_server:8020/mcp` |
| Temperature | `0.2` (low — factual investigation, not creative) |
| Timeout | 120 s |
| Streaming | enabled |

The agent is initialised **lazily** on the first request and cached for the lifetime of the container. A double-checked lock (`asyncio.Lock`) prevents concurrent init races.

## SSE Event Types (`/chat/stream`)

| Event | Payload | When |
|---|---|---|
| `status` | `{"state": "started"}` | Request received |
| `tool_start` | `{"name": "<tool>"}` | Agent calls a tool |
| `tool_end` | `{"name": "<tool>", "output": "..."}` | Tool returns |
| `token` | `{"text": "<fragment>"}` | LLM streams a token |
| `done` | `{"ok": true}` | Agent finished |

## Slack Notifier (`notifier.py`)

After the agent finishes, `notify_operator()` posts to the configured Slack webhook:

```
🚨 Anomaly Detected — Machine `M_0001`
<full agent investigation summary>
```

- Called **after** the `done` SSE event so it never delays the stream
- Silently skipped if `SLACK_WEBHOOK_URL` is not set
- Errors are logged but never propagate to the caller

## Configuration (`config.py`)

| Variable | Default | Description |
|---|---|---|
| `MCP_SERVER_URI` | `http://mcp_server:8020/mcp` | FastMCP SSE endpoint |
| `VLLM_BASE_URL` | `http://vllm:8000/v1` | OpenAI-compatible vLLM API |
| `chat_model` | `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` | Model name sent in API requests |
| `SLACK_WEBHOOK_URL` | `None` | Slack incoming webhook (set in `.env`) |

## Request Schema

```python
class ChatRequest(BaseModel):
    message:    str            # Anomaly description forwarded by if_anomaly_service
    machine_id: str = "unknown"  # Used in the Slack notification
```

## Build & Run

```bash
# Build
docker build -f services/langchain_service/Dockerfile -t langchain_service:latest .

# Run
docker compose --profile online up langchain_service
```

Depends on `vllm` (healthy) before the agent can be initialised.