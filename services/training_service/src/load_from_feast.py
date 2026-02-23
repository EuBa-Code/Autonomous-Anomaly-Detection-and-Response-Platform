import pandas as pd
from feast import FeatureStore
import logging

logger = logging.getLogger(__name__)

class DataManager:
    """
    Responsible for loading data from Feast Feature Store.
    Isolates data loading logic from the rest of the pipeline.
    """
    def __init__(self, settings):
        """Constructor: saves settings and initializes Feast client"""
        self.s = settings
        self.store = FeatureStore(repo_path=self.s.feast_repo_path)     # FeatureStore: client to communicate with Feast repo_path: path to Feast metadata (feature_store.yaml)
        
    def load_data(self) -> pd.DataFrame:
        """
        Carica i dati elaborati da PySpark tramite Feast per verificare performance e skew.
        Il caricamento avviene in chunk logici per non saturare la RAM.
        """
        import time
        t0 = time.time()
        logger.info("[FEAST] Inizializzazione FeatureStore...")
        logger.info(f"[FEAST] FeatureStore pronto in {time.time()-t0:.4f}s")

        t1 = time.time()
        logger.info("[FEAST] Lettura entity_df dal percorso Spark...")
        entity_df = pd.read_parquet(self.s.entity_df_path)
        logger.info(f"[FEAST] Actual columns: {entity_df.columns.tolist()}")
        entity_df = entity_df[["Machine_ID", "timestamp"]]

        # Mantieni UTC coerentemente
        entity_df["timestamp"] = pd.to_datetime(
            entity_df["timestamp"], utc=True
        )
        logger.info(f"[FEAST] entity_df letto in {time.time()-t1:.2f}s — {len(entity_df)} righe")

        t2 = time.time()
        logger.info(f"[FEAST] Avvio get_historical_features utilizzando il service: {self.s.feature_service_name}...")

        # --- CHUNKED LOADING: divide entity_df in batch per non saturare RAM ---
        chunk_size = getattr(self.s, "feast_chunk_size", 50_000)
        chunks = [
            entity_df.iloc[i : i + chunk_size]
            for i in range(0, len(entity_df), chunk_size)
        ]
        logger.info(f"[FEAST] Caricamento in {len(chunks)} chunk da {chunk_size} righe ciascuno")

        parts = []
        for idx, chunk in enumerate(chunks, 1):
            t_chunk = time.time()
            logger.info(f"[FEAST] Chunk {idx}/{len(chunks)} — {len(chunk)} righe...")
            part = self.store.get_historical_features(
                entity_df=chunk,
                features=self.store.get_feature_service(self.s.feature_service_name)
            ).to_df()
            parts.append(part)
            logger.info(f"[FEAST] Chunk {idx}/{len(chunks)} completato in {time.time()-t_chunk:.2f}s — {len(part)} righe caricate")

        training_df = pd.concat(parts, ignore_index=True)
        del parts  # libera subito la memoria dei chunk intermedi

        logger.info(f"[FEAST] get_historical_features completato in {time.time()-t2:.2f}s")
        logger.info(f"[FEAST] Totale load_data: {time.time()-t0:.2f}s — {len(training_df)} righe caricate")

        return training_df