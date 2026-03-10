from pydantic_settings import BaseSettings
from typing import Final

class Config(BaseSettings):
    qdrant_url: str = 'http://qdrant:6334' # gRPC
    qdrant_collection: str = 'ingestion_rag_service'
    qdrant_api_key: str | None = None
    data_dir: str = '/ingestion_rag_service/rag_files'
    embedding_model: str = 'BAAI/bge-m3'

ingestion_settings: Final[Config] = Config()