from pydantic_settings import BaseSettings
from typing import Final


class Config(BaseSettings):
    mcp_server_uri: str = 'http://mcp_server:8020/mcp'
    vllm_base_url: str = 'http://vllm:8000/v1'
    chat_model: str = 'Qwen/Qwen2.5-7B-Instruct'
    slack_webhook_url: str | None = None  


inference_settings: Final[Config] = Config()