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
            logger.info(f"[DATA] Reading Single Parquet: {path}")
            df = pd.read_parquet(path)
            logger.info(f"[DATA] File uploaded to {time.time()-t0:.2f}s — {len(df)} righe")
            return df
        
        # Case 2: Directory with parquet files
        if os.path.isdir(path):
            logger.info(f"[DATA] Reading Directory: {path}")
            parquet_files = glob.glob(os.path.join(path, "**/*.parquet"), recursive=True)
            
            if not parquet_files:
                raise FileNotFoundError(
                    f"[DATA] ERROR: No .parquet file found in {path}"
                )
            
            logger.info(f"[DATA] Found {len(parquet_files)} parquet files")
            dfs = []
            for file in sorted(parquet_files):
                logger.info(f"[DATA] Loading: {os.path.basename(file)}")
                dfs.append(pd.read_parquet(file))
            
            df = pd.concat(dfs, ignore_index=True)
            logger.info(f"[DATA] Directory uploaded to {time.time()-t0:.2f}s — {len(df)} righe")
            return df
        
        raise FileNotFoundError(
            f"[DATA] ERRORE: PATH NOT FOUND: {path}"
        )
    
    def _load_phase2_datalake_with_feast(self) -> pd.DataFrame:
        """
        PHASE 2: Load from datalake + enrich with Feast get_historical_features.
        Uses point-in-time join for accurate feature values.
        """
        t0 = time.time()
        logger.info("[DATA] Loading entity_df from DataLake...")
        
        # 1. Load entity_df from datalake
        entity_df = pd.read_parquet(self.s.entity_df_path)
        logger.info(f"[DATA] Loaded entity_df: {len(entity_df)} rows")
        
        # 2. Parse timestamp
        ts_col = self.s.event_timestamp_column
        if ts_col in entity_df.columns:
            entity_df[ts_col] = pd.to_datetime(
                entity_df[ts_col], utc=True
            ).dt.tz_localize(None)
            logger.info(f"[DATA] Parsed '{ts_col}' Timestamp")
        else:
            logger.warning(f"[DATA] Timestamp Column '{ts_col}' not found")
        
        if entity_df.empty:
            logger.warning("[DATA] entity_df IS EMPTY!")
            return entity_df
        
        # 3. Feast Point-in-Time Join
        logger.info(f"[FEAST] Starting get_historical_features with {self.s.feature_service_name}...")
        
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
                logger.info(f"[FEAST] Chunk {idx}/{len(chunks)} completed in {time.time()-t_chunk:.2f}s")
            
            training_df = pd.concat(parts, ignore_index=True)
            logger.info(f"[FEAST] Join completed in {time.time()-t0:.2f}s — {len(training_df)} rows")
            return training_df
        
        except Exception as e:
            logger.warning(f"[FEAST] ERROR: Feast Join failure: {e}")
            logger.warning("[DATA] FALLBACK: Returning only datalake data")
            return entity_df