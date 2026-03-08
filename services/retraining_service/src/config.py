from typing import Optional, Literal
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Classe per la gestione dei parametri di training
class TrainingArguments(BaseModel):
    test_size: float = Field(default=0.2)  # dimensione del test set
    random_state: int = 42  # random state per riproducibilità

    # iperparametri del modello
    rf_n_estimators: int = Field(default=300)   # N singoli alberi che compongono il random forest
    # strategia di gestione del class imbalance
    rf_class_weight: Optional[Literal["balanced", "balanced_subsample"]] = "balanced"
    rf_max_depth: Optional[int] = Field(default=None)


# Classe per la gestione delle variabili d'ambiente
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # MLFlow
    mlflow_tracking_uri: str = "http://mlflow:5000"   # url del server mlflow (http://nome_dell_container:porta)
    mlflow_experiment_name: str = "fraud_training"  # nome delle run di training che vogliamo vedere nella UI
    mlflow_model_name: str = "fraud_model"  # nome del modello ottenuto al termine delle run

    # Feast / data
    entity_df_path: str = "/datalake/entity_df_transactions.parquet"  # path all' entity_df
    feast_repo_path: str = "/feature_repo"  # path alla feature repo feast (all'interno del container di training)
    feature_service_name: str = "fraud_v1"  # nome della feature service feast

    target_column: str = "label"  # nome della colonna che contiene la label
    event_timestamp_column: str = "event_timestamp"  # nome della colonna che contiene il timestamp

    output_dir: str = "/app_out"  # path alla cartella nella quale salvare il modello alla fine del training
    # (riferendosi sempre al path all'interno del container)

    training: TrainingArguments = TrainingArguments()  # istanza dei parametri di training
