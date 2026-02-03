"""
Main module for training the Isolation Forest model
"""
import logging
import sys
from pathlib import Path
from typing import Optional
import pandas as pd

from config import IsolationForestConfig

from validation import ConfigValidator
from src.dataloader import DataLoader
from model import IsolationForestModel
from metrics import MetricsCalculator

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)

isf = IsolationForestConfig()

logger = logging.getLogger(__name__)

class TrainingPipeline:
    """Complete pipeline for model training"""
    def __init__(self, data_dir: Path = isf.DATA_DIR, model_dir: Path = isf.MODEL_DIR, metrics_dir: Path = isf.METRICS_DIR):
        """
        Initializes the training pipeline
        
        Args:
            data_dir: Directory containing historical data
            model_dir: Directory to save trained models
            metrics_dir: Directory to save metrics
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.metrics_dir = metrics_dir

        self.data_loader = DataLoader(data_dir) ############ Using the unedited historical data
        self.model = None
        self.validator = ConfigValidator(data_dir, model_dir, metrics_dir)
        self.metrics_calculator = MetricsCalculator()

    def validate_configuration(self) -> bool:
        """Validates the configuration
        
        Returns:
            bool: True if validation succeeded
        """
        logger.info("=" * 80)
        logger.info("CONFIGURATION VALIDATION")
        logger.info("=" * 80)
        
        is_valid, error_msg = self.validator.validate_all()
        
        if not is_valid:
            logger.error(f"Validation failed: {error_msg}")
            return False
        
        logger.info("Configuration successfully validated")
        return True
    
    def load_preprocess_data(self, filepattern: str = isf.TRAIN_FILENAME):
        """
        Loads and preprocesses the data
        
        Args:
            filepattern: Pattern for files to load
        
        Returns:
            pd.DataFrame: Preprocessed features (X)
        """    
        logger.info("=" * 80)
        logger.info("LOADING AND PREPROCESSING DATA")
        logger.info("=" * 80)
        
        # Load data
        data = self.data_loader.load_data(filepattern)

        # Preprocess (fit=True for training)
        X = self.data_loader.preprocess_data(data, fit=True)

        logger.info(f"Data ready: {X.shape[0]} samples, {X.shape[1]} features")

        return X
     
    def train_model(self, X_train: pd.DataFrame, model_params: dict = None):
        """
        Trains the Isolation Forest model
    
        Args:
            X_train: Training features
            model_params: Model parameters
    
        Returns:
            IsolationForestModel: Trained model
        """        
        logger.info("=" * 80)
        logger.info("TRAINING MODEL")
        logger.info("=" * 80)
        
        if model_params is None:
            model_params = ISOLATION_FOREST_PARAMS
        
        logger.info(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features")
        
        self.model = IsolationForestModel(**model_params)
        self.model.train(X_train)

        logger.info("Training completed successfully")
    
        return self.model
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: Optional[pd.Series] = None):
        """
        Evaluates the model on the test set
        
        Args:
            X_test: Test features (preprocessed)
            y_test: True labels (optional, for supervised metrics)
        
        Returns:
            Dict: Evaluation metrics
        """
        logger.info("=" * 80)
        logger.info("MODEL EVALUATION")
        logger.info("=" * 80)
        
        if self.model is None or not self.model.is_trained:
            raise RuntimeError("The model must be trained before evaluation")
        
        logger.info(f"Evaluating on {X_test.shape[0]} samples")
        
        y_pred = self.model.predict(X_test)
        scores = self.model.score_samples(X_test)
        
        X_test_original = self.data_loader.inverse_transform(X_test)
        
        logger.info(f"X_test shape after inverse_transform: {X_test_original.shape}")
        logger.info(f"X_test columns: {X_test_original.columns.tolist()}")
        logger.info(f"X_test sample (first row):\n{X_test_original.iloc[0]}")

        df_final = X_test_original.copy()
        df_final['anomaly_prediction'] = y_pred
        df_final['anomaly_score'] = scores

        if y_test is not None:
            df_final['Is_Anomaly'] = y_test.values

        df_final.to_csv(self.metrics_dir / "test_predictions.csv", index=False)
        logger.info("Predictions saved to test_predictions.csv")
        
        if y_test is not None:
            logger.info("Calculating supervised metrics with true labels...")
            metrics = self.metrics_calculator.calculate_metrics(
                y_true=y_test.values,
                y_pred=y_pred,
                scores=scores,
                X=X_test.values
            )
            
            self.metrics_calculator.print_summary()
        else:
            logger.info("No labels available, calculating unsupervised metrics...")
            metrics = self.metrics_calculator.calculate_unsupervised_metrics(
                y_pred=y_pred,
                scores=scores,
                X=X_test.values
            )

        return metrics
    
    def save_artifacts(self):
        """Saves model, scaler, and metrics"""
        logger.info("=" * 80)
        logger.info("SAVING ARTIFACTS")
        logger.info("=" * 80)
        
        if self.model is None or not self.model.is_trained:
            raise RuntimeError("The model must be trained before saving")

        model_path = self.model_dir / isf.MODEL_FILENAME
        self.model.save_model(model_path)
        
        scaler_path = self.model_dir / isf.SCALER_FILENAME
        self.data_loader.save_scaler(scaler_path)
        
        metrics_path = self.metrics_dir / isf.METRICS_FILENAME
        self.metrics_calculator.save_metrics(metrics_path)
        
        logger.info("All artifacts saved successfully")
    
    def run_training_evaluate(self,
                              train_file: str = isf.TRAIN_FILENAME,
                              test_file: str = isf.TEST_FILENAME,
                              model_params: dict = None):
        """
        Executes the full training and evaluation pipeline
        
        Args:
            train_file: Training file name
            test_file: Test file name
            model_params: Model parameters
        
        Returns:
            Dict: Evaluation metrics
        """
        try:
            logger.info("=" * 80)
            logger.info("STARTING ISOLATION FOREST TRAINING PIPELINE")
            logger.info("=" * 80)
            
            if not self.validate_configuration():
                raise RuntimeError("Configuration validation failed")
            
            logger.info("Loading training data...")
            X_train = self.load_preprocess_data(filepattern=train_file)
            
            self.train_model(X_train, model_params=model_params)
            
            logger.info("Loading test data...")
            data_test = self.data_loader.load_data(test_file)

            y_test = None
            if 'Is_Anomaly' in data_test.columns:
                y_test = data_test['Is_Anomaly']
                logger.info(f"'Is_Anomaly' labels found in test set")
                logger.info(f"Distribution: {y_test.value_counts().to_dict()}")
                anomaly_rate = y_test.mean()
                logger.info(f"  Anomaly rate: {anomaly_rate:.2%} ({int(y_test.sum())}/{len(y_test)})")
            else:
                logger.warning("'Is_Anomaly' column not found in test set")
                logger.warning("Only unsupervised metrics will be calculated")

            X_test = self.data_loader.preprocess_data(data_test, fit=False)
            
            logger.info(f"Original test columns: {data_test.columns.tolist()}")
            logger.info(f"X_test columns after preprocessing: {X_test.columns.tolist()}")
            logger.info(f"Scaler feature columns: {self.data_loader.feature_columns}")

            sample_idx = 0
            logger.info(f"\nSample {sample_idx} - ORIGINAL values:")
            for col in self.data_loader.feature_columns:
                if col in data_test.columns:
                    logger.info(f"  {col}: {data_test[col].iloc[sample_idx]}")

            logger.info(f"\nSample {sample_idx} - SCALED values:")
            logger.info(f"{X_test.iloc[sample_idx].to_dict()}")

            metrics = self.evaluate_model(X_test, y_test)
            
            self.save_artifacts()
            
            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error during pipeline execution: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    metrics = pipeline.run_training_evaluate()
    logger.info(f"Final metrics: {metrics}")