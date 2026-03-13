from pydantic_settings import BaseSettings
from typing import Final

class Settings(BaseSettings):
    mongo_uri: str = "mongodb://admin:admin@mongodb:27017"
    mongo_db: str = "mcp_database"
    collection: str = "logs_agent"

retrieval_settings: Final[Settings] = Settings()
