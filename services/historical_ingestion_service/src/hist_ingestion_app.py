from services.historical_ingestion_service.src.dataloader import DataLoader
from config import IsolationForestConfig

dataloader = DataLoader(IsolationForestConfig.DATA_DIR)
dataloader.load_data(IsolationForestConfig.TRAIN_PATH)
