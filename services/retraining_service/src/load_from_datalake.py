import pandas as pd
import logging
import time
import glob
import os

logger = logging.getLogger(__name__)


class DataManager:
    """
    Datalake-only data loading strategy.
    Loads training data directly from parquet files (single file or directory).
    No Feast / feature store dependencies.
    """

    def __init__(self, settings):
        """
        Args:
            settings: Settings object with paths and configuration.
        """
        self.s = settings

    def load_data(self) -> pd.DataFrame:
        """
        Load training data from the datalake (parquet).

        Supports:
            - Single .parquet file
            - Directory containing one or more .parquet files (recursive)

        Returns:
            pd.DataFrame with all loaded rows.

        Raises:
            FileNotFoundError: if the configured path does not exist or
                               contains no parquet files.
        """
        t0 = time.time()
        path = self.s.entity_df_path
        logger.info(f"[DATA] Loading data from datalake: {path}")

        # ── Case 1: single parquet file ──────────────────────────────────────
        if path.endswith(".parquet") and os.path.isfile(path):
            logger.info(f"[DATA] Reading single parquet file: {path}")
            df = pd.read_parquet(path)
            logger.info(
                f"[DATA] Loaded in {time.time() - t0:.2f}s — {len(df):,} rows"
            )
            return df

        # ── Case 2: directory with parquet files ─────────────────────────────
        if os.path.isdir(path):
            logger.info(f"[DATA] Reading parquet directory: {path}")
            parquet_files = sorted(
                glob.glob(os.path.join(path, "**/*.parquet"), recursive=True)
            )

            if not parquet_files:
                raise FileNotFoundError(
                    f"[DATA] No .parquet files found under: {path}"
                )

            logger.info(f"[DATA] Found {len(parquet_files)} parquet file(s)")
            dfs = []
            for fp in parquet_files:
                logger.info(f"[DATA]   Loading: {os.path.basename(fp)}")
                dfs.append(pd.read_parquet(fp))

            df = pd.concat(dfs, ignore_index=True)
            logger.info(
                f"[DATA] Loaded in {time.time() - t0:.2f}s — {len(df):,} rows"
            )
            return df

        raise FileNotFoundError(
            f"[DATA] Path not found or not a parquet file/directory: {path}"
        )