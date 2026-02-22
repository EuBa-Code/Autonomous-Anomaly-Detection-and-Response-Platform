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
        """
        import time
        t0 = time.time()
        logger.info("[FEAST] Inizializzazione FeatureStore...")
        # Il FeatureStore viene già inizializzato in __init__
        logger.info(f"[FEAST] FeatureStore pronto in {time.time()-t0:.4f}s")

        t1 = time.time()
        logger.info("[FEAST] Lettura entity_df dal percorso Spark...")
        # Carichiamo solo le colonne minime necessarie per Feast
        entity_df = pd.read_parquet(self.s.entity_df_path)[["Machine_ID", "timestamp"]]
        
        # Rinomina per chiarezza verso Feast
        entity_df.rename(columns={"timestamp": "event_timestamp"}, inplace=True)
        
        # Mantieni UTC coerentemente
        entity_df["event_timestamp"] = pd.to_datetime(
            entity_df["event_timestamp"], utc=True
        )
        logger.info(f"[FEAST] entity_df letto in {time.time()-t1:.2f}s — {len(entity_df)} righe")

        t2 = time.time()
        logger.info(f"[FEAST] Avvio get_historical_features utilizzando il service: {self.s.feature_service_name}...")
        
        # Recupero delle feature storiche tramite Feast
        # Se si blocca qui, il problema è del Point-in-Time Join (Dask/Memory)
        training_df = self.store.get_historical_features(
            entity_df=entity_df,
            features=self.store.get_feature_service(self.s.feature_service_name)
        ).to_df()
        
        logger.info(f"[FEAST] get_historical_features completato in {time.time()-t2:.2f}s")
        logger.info(f"[FEAST] Totale load_data: {time.time()-t0:.2f}s — {len(training_df)} righe caricate")

        return training_df

"""    
        def prepare_features(self, df, drop_cols):
        
        Prepara features per il modello.
        Rimuove colonne non predittive (timestamp, ID, ecc.)
        
        Args:
            df: DataFrame completo
            drop_cols: Lista di colonne da escludere
        
        Returns:
            x: DataFrame con sole features
        
        x = df.drop(columns=[c for c in drop_cols if c in df.columns])
        logger.info(f"[DATA] Features prepared: {x.shape}")
        return x"""