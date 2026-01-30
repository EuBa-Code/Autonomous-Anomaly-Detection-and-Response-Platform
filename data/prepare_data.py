import logging
import sys
from data.create_datasets import split_streaming_training
from data.synthetic_data_creation import generate_synthetic_data

# Configure logging to track the execution flow
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_datasets():
    """
    Orchestrates the data preparation pipeline:
    1. Splits streaming data for training.
    2. (Optional/Future) Generates synthetic data.
    """
    try:
        logging.info('Creating synthethic data')
        generate_synthetic_data()
        logging.info("Data creation completed successfully")
        
        logging.info("Starting the data splitting process...")
        split_streaming_training()
        logging.info("Data splitting completed successfully.")

    except ImportError as e:
        logging.error(f"Module import failed: {e}. Check your folder structure.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during data preparation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Standard entry point check
    create_datasets()