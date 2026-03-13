from pydantic_settings import BaseSettings
from typing import Final

class Config(BaseSettings):
    qdrant_url: str = 'http://qdrant:6333'
    qdrant_collection: str = 'anomaly_rag'  
    qdrant_api_key: str | None = None
    embedding_model: str = 'BAAI/bge-m3'

retrieval_settings: Final[Config] = Config()