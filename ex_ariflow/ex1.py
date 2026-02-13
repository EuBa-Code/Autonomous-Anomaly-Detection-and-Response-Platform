# dags/auto_training.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys
import os

# Add project root to path so Airflow can import 'src'
# Airflow mounts the volume at /opt/airflow/src, so we add /opt/airflow
sys.path.append("/opt/airflow")

from src.train import train_isolation_forest

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# DAG Definition
with DAG(
    'automatic_model_training',
    default_args=default_args,
    description='Anomaly Detection Automatic Training Pipeline',
    schedule_interval=timedelta(days=7), # Run every week
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlops', 'training']
) as dag:

    # Task 1: Check dependencies (Optional, useful for debugging in Docker)
    check_env = BashOperator(
        task_id='check_environment',
        bash_command='pip list | grep scikit-learn || echo "Scikit-learn not found"'
    )

    # Task 2: Training Execution
    # Calls the Python function we created
    training_task = PythonOperator(
        task_id='train_isolation_forest',
        python_callable=train_isolation_forest
    )

    # Flow definition
    check_env >> training_task

if __name__ == "__main__":
    dag.cli()
