"""
anomaly_consumer.py
--------------------
Standalone RedPanda (Kafka) consumer.

This is the "If Anomaly True" node from the architecture diagram.
It lives in its own Docker container (or as a separate process within
the mcp_client container) and acts as the bridge between the
real-time inference pipeline and the MCP Client agent.

Flow:
  RedPanda (predictions topic)
      └─► [this consumer]
              ├─ anomaly == False  → skip / log
              └─ anomaly == True   → POST /investigate to MCP Client API
"""

import logging

from quixstreams import Application

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("anomaly_consumer")
