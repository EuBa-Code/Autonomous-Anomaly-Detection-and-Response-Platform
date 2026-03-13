# vLLM Service

## Overview

Local LLM inference server that exposes an **OpenAI-compatible REST API** backed by a quantised 7B model running on the host GPU. It is the language model endpoint consumed by the LangChain service when the agent investigates an anomaly.

## Base Image

```
vllm/vllm-openai:latest
```

Requires an NVIDIA GPU (`runtime: nvidia` in `compose.yaml`).

## Model

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` |
| Quantisation | GPTQ Int4 — ~4 GB VRAM footprint |
| Context length | 8192 tokens |
| dtype | bfloat16 |
| GPU memory utilization | 90% |

The model is pulled from HuggingFace on first start and cached at `./local_models/hf_cache` (bind-mounted from the host). Subsequent starts load directly from cache — no re-download.

## API

Served on port `8000` (mapped to `8222` on the host), fully compatible with the OpenAI `/v1/chat/completions` API:

```bash
# Health check
curl http://localhost:8222/health

# Inference
curl http://localhost:8222/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    "messages": [{"role": "user", "content": "Investigate this anomaly..."}]
  }'
```

Inside the Docker network the LangChain service reaches it at `http://vllm:8000/v1`.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HF_HOME` | `/root/.cache/huggingface` | HuggingFace model cache (bind-mounted volume) |
| `HUGGING_FACE_HUB_TOKEN` | — | Required for gated models; set in `.env` |
| `NVIDIA_VISIBLE_DEVICES` | `all` | GPU visibility |

## Tuning

Both parameters in the `CMD` are the first knobs to adjust if you hit VRAM limits:

| Flag | Current value | Effect of lowering |
|---|---|---|
| `--gpu-memory-utilization` | `0.90` | Frees VRAM headroom for other GPU processes |
| `--max-model-len` | `8192` | Reduces KV-cache size; limits max context window |

## Build & Run

```bash
# Build
docker build -f services/vllm_service/Dockerfile -t vllm_service:latest .

# Run
docker compose --profile online up vllm
```

The compose healthcheck polls `GET /health` every 20 seconds with a 120-second `start_period` — the model load takes 1–2 minutes on first boot from cache.