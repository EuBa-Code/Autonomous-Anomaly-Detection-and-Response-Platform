from pydantic_settings import BaseSettings
from typing import Final

class Config(BaseSettings):
    qdrant_url: str = 'http://qdrant:6334' 
    qdrant_collection: str = '/ingestion_rag_service/rag_files'
    qdrant_api_key: str | None = None
    embedding_model: str = 'BAAI/bge-m3'

ingestion_settings: Final[Config] = Config()