import pandas as pd
import numpy as np
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
from sdv.evaluation.single_table import evaluate_quality
import sys
from pathlib import Path

# --- CONFIGURATION IMPORT SETUP ---
# Since this script is located in 'data/synthetic_data_creation',
# we need to add the project root to the Python path to import 'config'.
# We go up 3 levels: file -> synthetic_data_creation -> data -> ROOT
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from config import Config
# ----------------------------------

def generate_synthetic_data(num_rows=100000):
    print(f"📂 Loading data from: {Config.INPUT_DATA_PATH}")
    
    # Check if file exists to avoid crashes
    if not Config.INPUT_DATA_PATH.exists():
        raise FileNotFoundError(f"File not found: {Config.INPUT_DATA_PATH}")

    data = pd.read_csv(Config.INPUT_DATA_PATH)
    
    # Minimal cleaning required for SDV
    if 'Anomaly_Type' in data.columns:
        data['Anomaly_Type'] = data['Anomaly_Type'].fillna('None')

    # Training only on sensors (prevents OverflowError crash)
    # We exclude timestamp and ID from the learning process
    train_cols = [c for c in data.columns if c not in ['timestamp', 'Machine_ID']]
    metadata = Metadata.detect_from_dataframe(data[train_cols])
    
    print("🤖 Training synthesizer...")
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(data[train_cols])
    
    # Generate sensor values
    print(f"⚡ Generating {num_rows} rows...")
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    
    # LOGICAL CONSISTENCY FIX (Is_Anomaly vs Anomaly_Type)
    print("🔧 Fixing logical inconsistencies in Anomaly reporting...")
    
    # Get unique real anomalies excluding 'None'
    real_anomalies = [a for a in data['Anomaly_Type'].unique() if str(a) != 'None']
    
    # First step: If Is_Anomaly is 1 but Anomaly_Type is 'None' -> Assign a random real anomaly
    mask_fix_type = (synthetic_data['Is_Anomaly'] == 1) & (synthetic_data['Anomaly_Type'] == 'None')
    if mask_fix_type.any():
        synthetic_data.loc[mask_fix_type, 'Anomaly_Type'] = np.random.choice(real_anomalies, size=mask_fix_type.sum())
    
    # Second step: If Is_Anomaly is 0 -> Anomaly_Type MUST be 'None'
    mask_fix_zero = (synthetic_data['Is_Anomaly'] == 0)
    synthetic_data.loc[mask_fix_zero, 'Anomaly_Type'] = 'None'
    
    # Timeline synchronization (5 machines every 30 seconds)
    print("🕒 Synchronizing timestamps...")
    num_machines = 5
    num_steps = num_rows // num_machines
    
    start_time = pd.to_datetime(data['timestamp']).min()
    timeline = pd.date_range(start=start_time, periods=num_steps, freq='30s')
    
    # Assign timestamps and Machine IDs
    synthetic_data['timestamp'] = np.repeat(timeline, num_machines)[:num_rows]
    synthetic_data['Machine_ID'] = (['WM_01', 'WM_02', 'WM_03', 'WM_04', 'WM_05'] * num_steps)[:num_rows]
    
    # ========================================================================
    # CRITICAL: Convert timestamp to millisecond precision for PySpark
    # ========================================================================
    print("⚙️  Converting timestamp to PySpark-compatible format (millisecond precision)...")
    
    # Convert to datetime64[ms] - this is what PySpark can read natively
    synthetic_data['timestamp'] = synthetic_data['timestamp'].astype('datetime64[ms]')
    
    print(f"   Timestamp dtype: {synthetic_data['timestamp'].dtype}")
    print(f"   Timestamp range: {synthetic_data['timestamp'].min()} to {synthetic_data['timestamp'].max()}")
    
    # Quality evaluation
    full_metadata = Metadata.detect_from_dataframe(data)
    report = evaluate_quality(data, synthetic_data, full_metadata)
    print(f"\n✅ DATASET QUALITY: {report.get_score() * 100:.2f}%")
    
    # ========================================================================
    # Save with PySpark-compatible settings
    # ========================================================================
    print(f"\n💾 Saving to: {Config.SYNTHETIC_OUTPUT_PATH}")
    
    # Ensure output directory exists
    Config.SYNTHETIC_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with explicit Parquet settings for Spark compatibility
    synthetic_data.to_parquet(
        Config.SYNTHETIC_OUTPUT_PATH,
        engine='pyarrow',
        compression='snappy',
        index=False,
        # CRITICAL: Force millisecond timestamp coercion
        coerce_timestamps='ms',
        allow_truncated_timestamps=True,
        # Use Parquet version compatible with Spark
        version='2.6'
    )
    
    print(f"🚀 File created successfully: {Config.SYNTHETIC_OUTPUT_PATH}")
    
    # Verification
    file_size_mb = Config.SYNTHETIC_OUTPUT_PATH.stat().st_size / 1024 / 1024
    print(f"   File size: {file_size_mb:.2f} MB")
    print(f"   Rows: {len(synthetic_data):,}")
    print(f"   Columns: {len(synthetic_data.columns)}")
    
    # ========================================================================
    # Optional: Verify the file can be read back
    # ========================================================================
    print("\n🔬 Verifying file compatibility...")
    try:
        verify_df = pd.read_parquet(Config.SYNTHETIC_OUTPUT_PATH)
        print(f"✅ Verification successful!")
        print(f"   Loaded {len(verify_df):,} rows")
        print(f"   Timestamp dtype: {verify_df['timestamp'].dtype}")
        
        # Check if any precision was lost
        if len(verify_df) == len(synthetic_data):
            print("✅ All rows preserved")
        else:
            print(f"⚠️  Row count mismatch: {len(verify_df)} vs {len(synthetic_data)}")
            
    except Exception as e:
        print(f"⚠️  Verification failed: {e}")
    
    print("\n" + "="*70)
    print("✅ SYNTHETIC DATA GENERATION COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. This file is now compatible with PySpark (millisecond timestamps)")
    print("2. You can load it directly with PySpark without conversion")
    print("3. Use it in your historical_ingestion_service")
    

if __name__ == "__main__":
    # You can pass num_rows as command line argument
    import argparse
    parser = argparse.ArgumentParser(description='Generate synthetic data for PySpark')
    parser.add_argument('--rows', type=int, default=100000, help='Number of rows to generate')
    args = parser.parse_args()
    
    generate_synthetic_data(num_rows=args.rows)