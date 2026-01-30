import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

# --- CONFIGURATION IMPORT SETUP ---
# Ensure project root is in sys.path to find the config module
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.config import Config
# ----------------------------------

def split_streaming_training(input_path: Path = Config.SYNTHETIC_OUTPUT_PATH):
    """
    Loads the synthetic dataset, splits it into streaming (10%) and 
    historical (90%) sets, cleans the training data, and saves to parquet.
    """
    
    print(f"📂 Loading dataset from: {input_path}")
    if not input_path.exists():
        raise FileNotFoundError(f"The file {input_path} was not found. Run the generator script first.")

    # Load the dataset
    df = pd.read_parquet(input_path)

    # 1. SPLIT STREAMING vs HISTORICAL (90/10)
    split_index = int(len(df) * 0.9)
    model_development_data = df.iloc[:split_index]
    streaming_data = df.iloc[split_index:]

    # 2. SPLIT TRAINING AND TEST SETS (80/20 of the historical data)
    train_set, test_set = train_test_split(
        model_development_data, 
        test_size=0.2, 
        random_state=42,
        shuffle=True 
    )

    # 3. FILTER ANOMALIES FROM TRAINING SET
    # Only keep normal records (Is_Anomaly == 0) for the training set
    train_set_clean = train_set[train_set[Config.TARGET] == 0].copy()

    # 4. DROP UNNECESSARY COLUMNS
    try:
        # Clean Training Set: Remove target and metadata
        train_set_clean = train_set_clean.drop(columns=Config.DROP_COLUMNS)
        
        # Clean Streaming Data: Keep timestamp but remove labels
        streaming_data = streaming_data.drop(columns=['Is_Anomaly', 'Anomaly_Type'])
        
        # Clean Test Set: Remove metadata but KEEP Is_Anomaly/Anomaly_Type for evaluation
        test_set = test_set.drop(columns=['timestamp', 'Machine_ID'])
        
    except KeyError as e:
        print(f"⚠️ Warning: Some columns were already missing: {e}")

    # 5. SAVE TO PARQUET
    # Ensure directories exist inside the 'data' folder
    Config.STREAMING_DIR.mkdir(parents=True, exist_ok=True)
    Config.HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)

    # Define final paths
    streaming_file = Config.STREAMING_DIR / "streaming_data.parquet"
    train_file = Config.HISTORICAL_DIR / "train_set.parquet"
    test_file = Config.HISTORICAL_DIR / "test_set.parquet"

    # Save
    streaming_data.to_parquet(streaming_file, index=False)
    train_set_clean.to_parquet(train_file, index=False)
    test_set.to_parquet(test_file, index=False)

    print(f"\n✅ Data processed successfully:")
    print(f"   - Streaming data saved to: {streaming_file}")
    print(f"   - Training/Test sets saved to: {Config.HISTORICAL_DIR}")
    print(f"   - Rows in Clean Training Set: {len(train_set_clean)}")

    return streaming_data, train_set_clean, test_set
