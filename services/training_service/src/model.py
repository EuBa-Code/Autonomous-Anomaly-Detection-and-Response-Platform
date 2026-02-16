from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

class ModelFactory:
    """
    Factory pattern: metodo statico che assembla una Pipeline
    complessa con preprocessing + modello.
    """
    @staticmethod
    def build_pipeline(num_cols, cat_cols, settings): # cat_cols = Cycle_Phase_ID
        """
        Costruisce Pipeline sklearn con:
        - ColumnTransformer (preprocessing parallelo per num/cat)
        - IsolationForest (anomaly detector)
        
        Args:
            num_cols: lista colonne numeriche
            cat_cols: lista colonne categoriche (Cycle_Phase_ID)
            settings: oggetto Settings con iperparametri
        
        Returns:
            Pipeline oggetto sklearn
        """
        pre = ColumnTransformer(                                # Trasforma in parallelo colonne numeriche e categoriche
            transformers=[
                ("num", Pipeline([
                    ("imp", SimpleImputer(strategy="median")),  # Imputazione con mediana per numeriche (robusta agli outlier)
                    ("scaler", StandardScaler())                # Standardizzazione (mean=0, std=1) per numeriche
                ]), num_cols),                                  
                ("cat", Pipeline([
                    ("imp", SimpleImputer(strategy="constant", fill_value="missing")), # Imputazione con "missing" per categoriche
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)), # One-Hot Encoding per categoriche, ignora categorie non viste in training
                ]), cat_cols),
            ],
            remainder="drop",                           # Drop altre colonne non specificate (es. timestamp, ID)
            verbose_feature_names_out=False,            # Nomi output più puliti (es. "num__feature1" → "feature1")
        )

        model = IsolationForest(
            n_estimators=settings.training.if_n_estimators, # Numero di alberi nell'ensemble (default 100)
            contamination=settings.training.contamination,  # Percentuale di anomalie attese (default 0.1 = 10%)
            random_state=settings.training.random_state,    # Seed per riproducibilità (default 42)
            n_jobs=-1,                                  # Usa tutti i core disponibili per velocizzare il training                                      
        )

        return Pipeline([("pre", pre),                  # Preprocessing come primo step
                         ("model", model)])             # Isolation Forest come secondo step
    
        # Workflow:
        #   pipe.fit(X) → pre.fit_transform(X) + model.fit(X_pre)
        #   pipe.predict(X) → pre.transform(X) + model.predict(X_pre)