# app/core/config.py
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models" / "saved_models"
DATA_DIR = BASE_DIR / "data"

# Model paths
LOGISTIC_PATH = MODELS_DIR / "logistic.pkl"
RF_PATH = MODELS_DIR / "random_forest.pkl"
XGB_PATH = MODELS_DIR / "xgboost.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
FEATURES_PATH = MODELS_DIR / "feature_names.pkl"
BACKGROUND_PATH = MODELS_DIR / "background_data.pkl"
METRICS_PATH = MODELS_DIR / "model_metrics.pkl"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
BEST_MODEL_NAME_PATH = MODELS_DIR / "best_model_name.txt"

PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
PREDICTIONS_FILE = PREDICTIONS_DIR / "predictions_history.csv"


# API settings
API_V1_STR = "/api/v1"
PROJECT_NAME = "Churn Prediction API"
VERSION = "1.0.0"

# Churn classes
CHURN_CLASSES = ["GOOD", "SHOWING EARLY SIGNS", "HIGHLY LIKELY", "CHURNED"]
RANDOM_SEED = 42
