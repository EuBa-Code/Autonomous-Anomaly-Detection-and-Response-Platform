import os
import json
import pandas as pd  # per le elaborazioni ed i join
import logging  # stampare dei log all'interno del container

import mlflow  # gestione del training
import mlflow.sklearn  # estensione mlflow per scikit-learn
from feast import FeatureStore  # classe di implementazione del feature store

from sklearn.metrics import roc_auc_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from config import Settings  # classe dei settings dal file config.py

# settiamo ill livello di logging
# logging.WARNING
# logging.INFO
# logging.DEBUG
logging.basicConfig(level=logging.INFO)


def main():
    logging.info("[TRAIN] - starting App")  # logging.log_level(stampa) stampa dei log corrispondenti al livello settato
    s = Settings()  # istanziamo i settings

    # MLFlow
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)  # impostiamo l' url al server di tracking
    mlflow.set_experiment(s.mlflow_experiment_name)  # creiamo la run di riferimento

    logging.info("[TRAIN] - loading entity df")
    # Load entity df
    entity_df = pd.read_parquet(s.entity_df_path)  # lettura dell' entity df

    logging.info("[TRAIN] - correctly loaded entity df")
    logging.debug("[TRAIN] - parsing timestamp")

    entity_df[s.event_timestamp_column] = pd.to_datetime(
        entity_df[s.event_timestamp_column], utc=True).dt.tz_convert(None)  # conversione del timestamp in datetime

    logging.debug("[TRAIN] - timestamp parsed")

    logging.info("[TRAIN] - loading historical features")
    # Feast historical features (+ ODFV if in FeatureService)

    store = FeatureStore(repo_path=
                         s.feast_repo_path  # path alla repo dove c'è il file feature_store.yaml dentro il container
                         )  # istanza del feature store

    feature_service = store.get_feature_service(
        s.feature_service_name  # nome della feature service (fraud_v1)
    )  # otteniamo lo schema delle feature da leggere

    df = store.get_historical_features(  # leggo le feature storiche, effettuo il point-in-time join
        # entity df (contiene gli id delle entity, eventuali feature runtime (es: amount), event_timestamp e la label
        entity_df=entity_df,
        # schema delle feature da ottenere (feature service)
        features=feature_service
    ).to_df()  # trasforma il dataset ottenuto dal point-in-time join in pandas DataFrame

    logging.info("[TRAIN] - correctly loaded features df")

    # Pulizia del timestamp
    ts = s.event_timestamp_column  # estraggo la colonna timestamp
    if ts not in df.columns:  # verifico che nel dataframe sia presente la colonna timestamp
        raise ValueError(f"Missing timestamp column '{ts}' in historical features output.")

    df[ts] = pd.to_datetime(df[ts], utc=True, errors="raise")   # uniformare i timestamp nella colonna selezionata

    # ---- TEMPORAL SPLIT: train = passato, test = futuro ----
    # 1) - ordino in ordine crescente di timestamp le righe del dataset
    df = df.sort_values(ts).reset_index(drop=True)
    # 2) - stabiliamo l'indice al quale dividere il dataset
    cut = int(len(df) * (1.0 - s.training.test_size))
    # validiamo l'indice di split
    if cut <= 0 or cut >= len(df):  # verifichiamo che sia non minore di 0 e non maggiore del massimo di righe
        raise ValueError(f"Bad temporal split cut={cut} for len={len(df)}. Check test_size.")

    # 3) - effettuo lo split
    # - nel training set tutti gli elementi dalla prima riga a quella di split (cut)
    train_df = df.iloc[:cut]
    # - nel test set tutti gli elementi rimanenti (cut)
    test_df = df.iloc[cut:]

    # X/y
    y_train = train_df[s.target_column].astype(int)  # estraggo la colonna label dal train set
    y_test = test_df[s.target_column].astype(int)  # estraggo la colonna label dal test set

    drop_cols = [  # definisco le colonne da rimuovere
        s.target_column,  # colonna con la label (y)
        ts,  # colonna con il timestamp
        "created_timestamp",
        "card_id",
        "customer_id",
        "event_timestamp",
        "transaction_id",
        "merchant_id",
        "mcc",
    ]

    # definisco le X di training e test
    # -- prendo tutte le colonne che non sono nella lista di quelle da rimuovere
    x_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    x_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    # Pre - Processing
    # Basic preprocessing: numeric + categorical
    # - 1) stabilire quali sono le colonne categoriche e quali numeriche
    # --- num_cols: tutte le colonne con tipo di dato numerico o booleano
    # --- cat_cols: tutte le colonne rimanenti
    num_cols = x_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    cat_cols = [c for c in x_test.columns if c not in num_cols]

    # Pipeline di preprocessing
    pre = ColumnTransformer(  # stabilire le trasformazioni sulle colonne
        transformers=[
            # sintassi: (nome della trasformazione, trasformazione, colonne sulla quale applicarla)
            ("num", SimpleImputer(strategy="median"), num_cols),  # imputazione con la mediana per lee colonne numeriche
            ("cat", Pipeline([  # trasformazione a più step sequenziali
                ("imp", SimpleImputer(strategy="most_frequent")),  # imputazione con la moda (most_frequent)
                # encoding con tecnica One Hot Encoding
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Istanza del modello
    model = RandomForestClassifier(
        n_estimators=s.training.rf_n_estimators,  # settiamo i parametri a partire dalla classe di config
        max_depth=s.training.rf_max_depth,
        n_jobs=-1,
        random_state=s.training.random_state,
        class_weight=s.training.rf_class_weight,
    )

    # pipeline di modellazione
    # 1 - preprocessing
    # 2 - applicazione del modello
    pipe = Pipeline(
        [("pre", pre),  # step di pre processing (la pipeline di pre processing)
         ("model", model)  # modello predittivo (Random Forest)
         ]
    )

    os.makedirs(s.output_dir, exist_ok=True)  # creiamo per sicurezza la output directory

    with mlflow.start_run():  # iniziamo una run di training registrata dal tracking server MLFlow
        # impostiamo con mlflow.log_params un dizionario dei parametri che vogliamo loggare nella run
        mlflow.log_params({
            "split_type": "temporal_fraction",  # tipologia di split che abbiamo utilizzato (random o temporale)
            "feature_service": s.feature_service_name,  # il nome della feature serve utilizzata
            "rf_n_estimators": s.training.rf_n_estimators,
            "rf_max_depth": s.training.rf_max_depth if s.training.rf_max_depth is not None else "none",
            "test_size": s.training.test_size,
            "random_state": s.training.random_state,
        })

        # log time ranges (super utile per debug)
        mlflow.log_params({
            "train_ts_min": str(train_df[ts].min()),
            "train_ts_max": str(train_df[ts].max()),
            "test_ts_min": str(test_df[ts].min()),
            "test_ts_max": str(test_df[ts].max()),
        })

        pipe.fit(x_train, y_train)  # addestramento del modello sul training set
        pred = pipe.predict(x_test)  # predizione sul test et con il modello addestrato
        # peer poter effettuare il calcolo dele metriche

        # con mlflow.log_metric registro le metriche di interesse per il modello, calcolate sul test set
        mlflow.log_metric("f1",  # nome della metrica
                          float(f1_score(y_test, pred))  # calcolo del valore della metrica
                          )

        # roc_auc solo se binario e predict_proba disponibile
        if hasattr(pipe.named_steps["model"], "predict_proba") and y_test.nunique() > 1:
            proba = pipe.predict_proba(x_test)[:, 1]
            mlflow.log_metric("roc_auc", float(roc_auc_score(y_test, proba)))

        # artifact: schema feature raw
        schema = {
            "raw_feature_columns": list(x_train.columns)  # lista delle feature utilizzate
        }
        # path dove salvare l'artifact dellle feature
        schema_path = os.path.join(s.output_dir,
                                   "feature_schema.json"
                                   )

        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=2)  # salvo il json

        # log dell' artifact features per osservabilità
        mlflow.log_artifact(schema_path)

        # log + register model
        mlflow.sklearn.log_model(  # salviamo il modello nel model registry
            sk_model=pipe,  # salviamo ill modello e le trasformazioni ad esso connesse come artifact
            name="model",  # nome dell' artifact
            registered_model_name=s.mlflow_model_name,  # nome del modello
        )

    print("Training Completo (temporal split).")


if __name__ == "__main__":
    main()
