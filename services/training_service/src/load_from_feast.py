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
        Carica dati da Feast Feature Store.
        Ritorna il dataset completo ordinato temporalmente.
        """
        logger.info("[DATA] Loading entity df and features from Feast")
        # Step 1: Carica entity_df dal parquet
        entity_df = pd.read_parquet(self.s.entity_df_path)
        
        # Step 2: Converti timestamp a UTC
        entity_df[self.s.event_timestamp_column] = pd.to_datetime(
            entity_df[self.s.event_timestamp_column], utc=True
        )

        # Step 3: Recupera feature service da Feast
        feature_service = self.store.get_feature_service(self.s.feature_service_name)

        # Step 4: Chiedi a Feast i dati storici per le entità e le features specificate
        df = self.store.get_historical_features(
            entity_df=entity_df,
            features=feature_service
        ).to_df()
        
        # Mantieni UTC coerentemente
        df[self.s.event_timestamp_column] = pd.to_datetime(
            df[self.s.event_timestamp_column], utc=True
        )
        
        logger.info(f"[DATA] Loaded {len(df)} rows from Feast")

        return df

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