"""
config.py
----------
Loads config/config.yaml and merges environment variable overrides.
"""

import os
from functools import lru_cache
from pathlib import Path

import yaml


CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


@lru_cache(maxsize=1)
def get_config() -> dict:
    with open(CONFIG_PATH, "r") as fh:
        cfg = yaml.safe_load(fh)

    # Allow environment variable overrides for the most common settings
    _env_overrides(cfg)
    return cfg


def _env_overrides(cfg: dict) -> None:
    """Merge environment variables into the config dict (in-place)."""

    # vLLM
    if url := os.getenv("VLLM_BASE_URL"):
        cfg["vllm"]["base_url"] = url
    if model := os.getenv("VLLM_MODEL"):
        cfg["vllm"]["model"] = model

    # MCP Server
    if url := os.getenv("MCP_SERVER_URL"):
        cfg["mcp_server"]["base_url"] = url

    # RedPanda
    if brokers := os.getenv("REDPANDA_BROKERS"):
        cfg["redpanda"]["bootstrap_servers"] = brokers
    if topic := os.getenv("PREDICTIONS_TOPIC"):
        cfg["redpanda"]["topics"]["predictions"] = topic

    # MCP Client host/port
    if port := os.getenv("MCP_CLIENT_PORT"):
        cfg["mcp_client"]["port"] = int(port)