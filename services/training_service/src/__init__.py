"""
Training Service - Isolation Forest Anomaly Detection

This package provides the training pipeline for the anomaly detection model.
"""

__version__ = "1.0.0"
__author__ = "Your Team"

from .train import TrainingPipeline
from .model import IsolationForestModel
from .dataloader import DataLoader
from .metrics import MetricsCalculator
from .config import config

__all__ = [
    'TrainingPipeline',
    'IsolationForestModel',
    'DataLoader',
    'MetricsCalculator',
    'config'
]