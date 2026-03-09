from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Kafka / Redpanda
    KAFKA_SERVER: str = "redpanda:9092"
    TOPIC_TELEMETRY: str = "telemetry-data"
    TOPIC_PREDICTIONS: str = "predictions"
    AUTO_OFFSET_RESET: str = "earliest"
    
    # MCP Client API
    # Assuming the MCP client is another service in the same network
    MCP_API_URL: str = "http://mcp_client:8000"

    class Config:
        env_file = ".env"

Config = Settings()