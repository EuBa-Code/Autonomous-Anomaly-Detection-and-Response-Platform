from pydantic_settings import BaseSettings
from typing import Final

class Config(BaseSettings):
    qdrant_url: str = 'http://qdrant:6333'
    qdrant_collection: str = 'machines_rag'
    qdrant_api_key: str | None = None
    data_dir: str = 'rag_files'
    embedding_model = 'BAAI/bge-m3'

ingestion_settings: Final[Config] = Config()