from pymongo import MongoClient
from datetime import datetime, timezone
import os

from mongo_logger.config import retrieval_settings

class MongoLogger:
    def __init__(self):
        self.client = MongoClient(retrieval_settings.mongo_uri)
        self.db = self.client[retrieval_settings.mongo_db]
        self.logs_collection = self.db[retrieval_settings.collection]

    def log_query(self, machine_id: int):
        
        log_entry = {
            'timestamp': datetime.now(),
            'Machine_ID': machine_id,
        }

        self.logs_collection.insert_one(log_entry)
