import pandas as pd
import numpy as np
import time
from typing import Tuple
from mlflow.models import infer_signature, ModelSignature

def create_and_log_signature(x_sample: pd.DataFrame, model_pipe) -> ModelSignature:
    """
    Generate the model signature from the raw DataFrame.
    Always provide the original DataFrame (not the transformed array),
    since the production model will receive inputs as JSON via the API.
    """
    # Run a test prediction on a sample input to determine the model output
    sample_output = model_pipe.predict(x_sample.head(5))
    
    # Infer the model signature by mapping DataFrame column names to basic data types (integer, float, object/string)
    signature = infer_signature(
        model_input=x_sample.head(5), 
        model_output=sample_output
    )
    
    return signature

def measure_latency(pipe, x_data: pd.DataFrame) -> float:
    """
    Measure average per‑record latency in milliseconds.
    """
    start_time = time.time()
    _ = pipe.predict(x_data)
    total_time_ms = (time.time() - start_time) * 1000
    
    return total_time_ms / len(x_data)