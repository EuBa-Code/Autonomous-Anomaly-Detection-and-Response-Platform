import pandas as pd
import logging
import time
import os
import glob

logger = logging.getLogger(__name__)

class DataManager:
    """
    Simple data loader for offline store processed features.
    Directly loads parquet files from the offline store directory.
    No Feast integration needed - data is pre-processed.
    """
    def __init__(self, settings):
        """Constructor: saves settings"""
        self.s = settings
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from offline store directory.
        
        Handles three cases:
        1. Single parquet file: /path/to/data.parquet
        2. Directory with parquet: /path/to/dir/ (reads all .parquet files)
        3. Spark output: /path/to/dir/ (reads all part-*.parquet files)
        
        Returns:
            DataFrame with loaded data
        """
        t0 = time.time()
        path = self.s.offline_store_path
        
        logger.info(f"[DATA] Caricamento dati da offline store: {path}")
        
        # Case 1: Single parquet file
        if path.endswith('.parquet') and os.path.isfile(path):
            logger.info(f"[DATA] Lettura file singolo: {path}")
            df = pd.read_parquet(path)
            logger.info(f"[DATA] File caricato in {time.time()-t0:.2f}s — {len(df)} righe, {len(df.columns)} colonne")
            return df
        
        # Case 2 & 3: Directory (with parquet files or Spark output)
        if os.path.isdir(path):
            logger.info(f"[DATA] Lettura directory: {path}")
            
            # Try to find parquet files
            parquet_files = glob.glob(os.path.join(path, "**/*.parquet"), recursive=True)
            
            if not parquet_files:
                raise FileNotFoundError(
                    f"[DATA] ERRORE: Nessun file .parquet trovato in {path}\n"
                    f"Cartelle disponibili: {os.listdir(path) if os.path.isdir(path) else 'N/A'}"
                )
            
            logger.info(f"[DATA] Trovati {len(parquet_files)} file parquet")
            
            # Load all parquet files
            dfs = []
            for file in sorted(parquet_files):
                logger.info(f"[DATA] Caricamento: {os.path.basename(file)}")
                dfs.append(pd.read_parquet(file))
            
            # Concatenate all parts
            df = pd.concat(dfs, ignore_index=True)
            logger.info(f"[DATA] Directory caricata in {time.time()-t0:.2f}s — {len(df)} righe, {len(df.columns)} colonne")
            return df
        
        # Case 4: Path doesn't exist
        raise FileNotFoundError(
            f"[DATA] ERRORE CRITICO: Percorso non trovato: {path}\n"
            f"Assicurati che il percorso sia corretto e che i dati siano disponibili."
        )
    
    def _parse_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse timestamp column to datetime.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with parsed timestamp column
        """
        ts_col = self.s.event_timestamp_column
        
        if ts_col not in df.columns:
            logger.warning(
                f"[DATA] Colonna timestamp '{ts_col}' NON trovata.\n"
                f"Colonne disponibili: {df.columns.tolist()}"
            )
            return df
        
        try:
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True).dt.tz_localize(None)
            logger.info(f"[DATA] Timestamp '{ts_col}' parsato correttamente")
        except Exception as e:
            logger.warning(f"[DATA] Errore nel parsing timestamp: {e}")
        
        return df