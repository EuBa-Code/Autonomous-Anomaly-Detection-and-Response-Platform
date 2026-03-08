from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest

class ModelFactory:
    """
    Factory method (static):
    constructs a composite Pipeline consisting of 
    preprocessing steps and a model.
    """
    @staticmethod
    def build_pipeline(num_cols, cat_cols, settings): # cat_cols = Cycle_Phase_ID
        """
        sklearn Pipeline builder composed of:
        - ColumnTransformer (parallel preprocessing for numerical and categorical columns)
        - IsolationForest (anomaly detector)
        
        Args:
            num_cols: numerical columns list
            cat_cols: categorical columns list (Cycle_Phase_ID)
            settings: Settings Object containing iperparameters
        
        Returns:
            sklearn Pipeline instance
        """
        pre = ColumnTransformer(                                # Apply parallel transformations to numerical and categorical columns.
            transformers=[
                ("num", Pipeline([
                    ("imp", SimpleImputer(strategy="median")),  # Median imputation for numerical features (robust to outliers)
                    ("scaler", StandardScaler())                # Standardization (mean=0, std=1) for numerical features
                ]), num_cols),                                  
                ("cat", Pipeline([
                    ("imp", SimpleImputer(strategy="constant", fill_value="missing")), # Impute categorical features with the "missing" placeholder
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)), # One-Hot Encoding for categorical columns, ignore categories not seen during training
                ]), cat_cols),
            ],
            remainder="drop",                           # Drop other unspecified columns (e.g., timestamp, ID)
            verbose_feature_names_out=False,            # Rename output columns to simplified feature names (es. "num__feature1" → "feature1")
        )

        model = IsolationForest(
            n_estimators=settings.training.if_n_estimators, # Number of trees in the ensemble (default 100)
            contamination=settings.training.contamination,  # Expected anomaly rate (default 0.1 = 10%)
            random_state=settings.training.random_state,    # Seed for reproducibility (default 42)
            n_jobs=-1,                                      # Use all available CPU cores to speed up training                                     
        )

        return Pipeline([("pre", pre),                  # First step:   Preprocessing
                         ("model", model)])             # Second step:  Isolation Forest
    
        # Workflow:
        #   pipe.fit(X) → pre.fit_transform(X) + model.fit(X_pre)
        #   pipe.predict(X) → pre.transform(X) + model.predict(X_pre)