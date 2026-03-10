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
from quixstreams.models import TimestampType
from datetime import timezone, datetime
import httpx
from typing import Any

from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("anomaly_consumer")

def _parse_iso8601(ts: str) -> datetime:
    """Parse an ISO-8601 string into a timezone-aware datetime (UTC fallback)."""
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def _to_epoch_ms(ts: str) -> int:
    """Convert an ISO-8601 string to epoch milliseconds."""
    return int(_parse_iso8601(ts).timestamp() * 1000)

def _to_epoch_s(ts: str) -> int:
    return int(_parse_iso8601(ts).timestamp())

def timestamp_format(value: dict[str, Any]) -> dict[str, Any]:
    
    value['timestamp'] = _to_epoch_s(str(value.get('timestamp')))

    return value

def timestamp_extractor(
    value: Any,
    headers: Any,
    timestamp: float,
    timestamp_type: TimestampType,
) -> int:
    """
    QuixStreams timestamp extractor.

    Prefers the event timestamp embedded in the message payload so that
    sliding windows are aligned to when the event actually occurred, not
    when Kafka received it.  Falls back to the broker timestamp on any error.
    """
    try:
        if isinstance(value, dict) and "timestamp" in value:
            return _to_epoch_ms(value["timestamp"])
    except Exception as exc:
        logger.error(
            "Timestamp extractor error — falling back to broker timestamp: %s", exc
        )
    return int(timestamp)


def trigger_mcp_investigation(message: dict):
    """
    This function handles the side effect of calling the MCP API.
    """
    logger.info(f"🚨 ANOMALY DETECTED: Triggering investigation for {message.get('Machine_ID', 'unknown')}")
    
    MCP_CLIENT_URL: str = "http://langchain_service:8010"

    payload = {
        "message": (
            f"Investigate anomaly for machine {message.get('Machine_ID')}. "
            f"Score: {message.get('anomaly_score'):.3f}. "
            f"Features: {message.get('features')}"
        )
    }
    try:
        with httpx.Client(timeout=30) as client:
            r = client.post(f"{Config.MCP_CLIENT_URL}/chat/stream", json=payload)
            r.raise_for_status()
        logger.info("Investigation triggered successfully.")
    except Exception as e:
        logger.error(f"Failed to trigger MCP agent: {e}")

def main() -> None:
    logger.info('Starting Anomaly Consumer Service')

    app = Application(
        broker_address=Config.KAFKA_SERVER,
        consumer_group=Config.CONSUMER_GROUP,
        auto_offset_reset=Config.AUTO_OFFSET_RESET,
    )

    topic = app.topic(
        Config.TOPIC_PREDICTIONS, # Reads from predictions
        value_deserializer='json',
        timestamp_extractor=timestamp_extractor
    )

    sdf = app.dataframe(topic)

    # 1. Filter the stream: Only keep records where is_anomaly is 1
    # This replaces the 'if sdf["is_anomaly"] == 1:' block
    sdf = sdf[sdf['is_anomaly'] == 1]

    # 2. Execute the action for every anomaly found
    sdf.update(trigger_mcp_investigation)

    # Optional: Log the results to console for debugging
    sdf.print()

    app.run()

if __name__ == '__main__':
    main()