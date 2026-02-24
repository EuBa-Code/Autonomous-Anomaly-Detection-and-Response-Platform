import pandas as pd
from feast import FeatureStore
import logging
import time

logger = logging.getLogger(__name__)

class DataManager:
    """
    Responsible for loading data from Feast Feature Store.
    Isolates data loading logic from the rest of the pipeline.
    """
    def __init__(self, settings):
        """Constructor: saves settings and initializes Feast client"""
        self.s = settings
        self.store = FeatureStore(repo_path=self.s.feast_repo_path)
        
    def load_data(self) -> pd.DataFrame:
        """
        Carica i dati dal datalake ed esegue un point-in-time join con il Feast Offline Store.
        Se l'offline store è vuoto, effettua una callback e ritorna solo i dati del datalake.
        """
        t0 = time.time()
        logger.info("[DATA] Lettura entity_df dal percorso Spark (Datalake)...")
        
        # 1. Load the full entity_df from the datalake
        entity_df = pd.read_parquet(self.s.entity_df_path)
        
        # 2. Parse timestamp using the column name from settings, matching train.py logic
        ts_col = getattr(self.s, "event_timestamp_column", "timestamp")
        if ts_col in entity_df.columns:
            # Converts the timestamp to datetime and removes the timezone
            entity_df[ts_col] = pd.to_datetime(
                entity_df[ts_col], utc=True
            ).dt.tz_convert(None) 
            
        logger.info(f"[DATA] entity_df letto in {time.time()-t0:.2f}s — {len(entity_df)} righe")

        # 3. Feast Point-in-Time Join
        logger.info(f"[FEAST] Avvio get_historical_features (Point-in-Time join) con {self.s.feature_service_name}...")
        
        # Chunking per non saturare la RAM
        chunk_size = getattr(self.s, "feast_chunk_size", 50_000)
        chunks = [
            entity_df.iloc[i : i + chunk_size]
            for i in range(0, len(entity_df), chunk_size)
        ]

        parts = []
        try:
            # Get the feature service to read the feature schema
            feature_service = self.store.get_feature_service(self.s.feature_service_name)
            
            for idx, chunk in enumerate(chunks, 1):
                t_chunk = time.time()
                
                # Pass the FULL chunk (entity_df) to Feast
                part = self.store.get_historical_features(
                    entity_df=chunk,
                    features=feature_service
                ).to_df()
                
                # Check if Feast returned empty data (offline store is empty)
                if part.empty:
                    raise ValueError("Feast offline store returned an empty dataframe.")
                    
                parts.append(part)
                logger.info(f"[FEAST] Chunk {idx}/{len(chunks)} completato in {time.time()-t_chunk:.2f}s")

            training_df = pd.concat(parts, ignore_index=True)
            logger.info(f"[FEAST] Join completato in {time.time()-t0:.2f}s")
            return training_df

        except Exception as e:
            # 4. CALLBACK / FALLBACK: Offline store empty or unavailable
            logger.warning(f"[FEAST] Errore o Offline Store vuoto: {e}")
            logger.warning("[DATA] CALLBACK: Ritorno i dati esclusivi dal Datalake (entity_df).")
            return entity_df