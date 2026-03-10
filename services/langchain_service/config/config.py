from pydantic_settings import BaseSettings
from typing import Final

class Config(BaseSettings):
    mcp_server_uri: str = 'http://localhost:8020/mcp'
    chat_model: str = 'Qwen/Qwen3-8B-Instruct'

inference_settings: Final[Config] = Config()