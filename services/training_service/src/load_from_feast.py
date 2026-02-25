import pandas as pd
from feast import FeatureStore
import logging
import time
import glob
import os

logger = logging.getLogger(__name__)

class DataManager:
    """
    Two-phase data loading strategy:
    
    PHASE 1 (Training #1): Load ONLY from datalake
    - Simple, no Feast dependencies
    - Bootstrap the model with initial data
    
    PHASE 2+ (Training #2+): Load from datalake + Feast get_historical_features
    - Enrich datalake entities with historical features from Feast
    - Point-in-time accuracy for better model updates
    """
    def __init__(self, settings, training_number: int = 1):
        """
        Constructor: saves settings and training number
        
        Args:
            settings: Settings object with paths and configuration
            training_number: Which training run this is (1, 2, 3, etc)
        """
        self.s = settings
        self.training_number = training_number
        self.phase = "PHASE 1: Datalake Only" if training_number == 1 else "PHASE 2: Datalake + Feast"
        
        # Initialize Feast only if not first training
        if training_number > 1:
            try:
                self.store = FeatureStore(repo_path=self.s.feast_repo_path)
                logger.info(f"[FEAST] Initialized Feast store: {self.s.feast_repo_path}")
            except Exception as e:
                logger.warning(f"[FEAST] Failed to initialize Feast: {e}")
                self.store = None
        else:
            self.store = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data according to training phase.
        
        PHASE 1 (Training #1):
            - Load entity_df from datalake only
            - No Feast join
            
        PHASE 2+ (Training #2+):
            - Load entity_df from datalake
            - Join with Feast using get_historical_features
            - Result has entity data + historical features
        
        Returns:
            DataFrame with training data
        """
        logger.info(f"[DATA] {self.phase}")
        logger.info(f"[DATA] Training #{self.training_number}")
        
        if self.training_number == 1:
            return self._load_phase1_datalake_only()
        else:
            return self._load_phase2_datalake_with_feast()
    
    def _load_phase1_datalake_only(self) -> pd.DataFrame:
        """
        PHASE 1: Load data ONLY from datalake.
        Simple, no Feast complexity.
        """
        t0 = time.time()
        logger.info("[DATA] Loading data from the DataLake - No Feast Join ...")
        
        path = self.s.entity_df_path
        
        # Case 1: Single parquet file
        if path.endswith('.parquet') and os.path.isfile(path):
            logger.info(f"[DATA] Lettura file singolo: {path}")
            df = pd.read_parquet(path)
            logger.info(f"[DATA] File caricato in {time.time()-t0:.2f}s — {len(df)} righe")
            return df
        
        # Case 2: Directory with parquet files
        if os.path.isdir(path):
            logger.info(f"[DATA] Lettura directory: {path}")
            parquet_files = glob.glob(os.path.join(path, "**/*.parquet"), recursive=True)
            
            if not parquet_files:
                raise FileNotFoundError(
                    f"[DATA] ERRORE: Nessun file .parquet trovato in {path}"
                )
            
            logger.info(f"[DATA] Trovati {len(parquet_files)} file parquet")
            dfs = []
            for file in sorted(parquet_files):
                logger.info(f"[DATA] Caricamento: {os.path.basename(file)}")
                dfs.append(pd.read_parquet(file))
            
            df = pd.concat(dfs, ignore_index=True)
            logger.info(f"[DATA] Directory caricata in {time.time()-t0:.2f}s — {len(df)} righe")
            return df
        
        raise FileNotFoundError(
            f"[DATA] ERRORE: Percorso non trovato: {path}"
        )
    
    def _load_phase2_datalake_with_feast(self) -> pd.DataFrame:
        """
        PHASE 2: Load from datalake + enrich with Feast get_historical_features.
        Uses point-in-time join for accurate feature values.
        """
        t0 = time.time()
        logger.info("[DATA] Caricamento entity_df dal datalake...")
        
        # 1. Load entity_df from datalake
        entity_df = pd.read_parquet(self.s.entity_df_path)
        logger.info(f"[DATA] entity_df caricato: {len(entity_df)} righe")
        
        # 2. Parse timestamp
        ts_col = self.s.event_timestamp_column
        if ts_col in entity_df.columns:
            entity_df[ts_col] = pd.to_datetime(
                entity_df[ts_col], utc=True
            ).dt.tz_localize(None)
            logger.info(f"[DATA] Timestamp '{ts_col}' parsato")
        else:
            logger.warning(f"[DATA] Colonna timestamp '{ts_col}' non trovata")
        
        if entity_df.empty:
            logger.warning("[DATA] entity_df è vuoto!")
            return entity_df
        
        # 3. Feast Point-in-Time Join
        logger.info(f"[FEAST] Avvio get_historical_features con {self.s.feature_service_name}...")
        
        if self.store is None:
            logger.warning("[FEAST] Feast store not initialized, returning datalake data only")
            return entity_df
        
        chunk_size = self.s.feast_chunk_size
        chunks = [
            entity_df.iloc[i : i + chunk_size]
            for i in range(0, len(entity_df), chunk_size)
        ]
        
        parts = []
        try:
            feature_service = self.store.get_feature_service(self.s.feature_service_name)
            
            for idx, chunk in enumerate(chunks, 1):
                t_chunk = time.time()
                part = self.store.get_historical_features(
                    entity_df=chunk,
                    features=feature_service
                ).to_df()
                
                if part.empty:
                    raise ValueError("Feast offline store returned empty dataframe")
                
                parts.append(part)
                logger.info(f"[FEAST] Chunk {idx}/{len(chunks)} completato in {time.time()-t_chunk:.2f}s")
            
            training_df = pd.concat(parts, ignore_index=True)
            logger.info(f"[FEAST] Join completato in {time.time()-t0:.2f}s — {len(training_df)} righe")
            return training_df
        
        except Exception as e:
            logger.warning(f"[FEAST] Errore nella Feast join: {e}")
            logger.warning("[DATA] FALLBACK: Ritornando solo datalake data")
            return entity_df