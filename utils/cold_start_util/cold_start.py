
from feast import FeatureStore
import logging
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

FEAST_REPO_PATH = "/cold_start_util/feature_store_service"
FEAST_FEATURE_VIEWS = "machine_batch_features"

def cold_start() -> None:
    """Push offline store rows into Redis via Feast materialize_incremental."""
    end_date = datetime.now(tz=timezone.utc)
    logger.info("Feast repo      : %s", FEAST_REPO_PATH)
    logger.info("Feature views   : %s", FEAST_FEATURE_VIEWS)
    logger.info("Materializing up to: %s", end_date.isoformat())

    store = FeatureStore(repo_path=FEAST_REPO_PATH)
    store.materialize_incremental(
        end_date=end_date,
        feature_views=[FEAST_FEATURE_VIEWS],
    )
    logger.info("✓ Materialization complete — Daily_Vibration_PeakMean_Ratio is now in Redis")

if __name__ == '__main__':
    cold_start()