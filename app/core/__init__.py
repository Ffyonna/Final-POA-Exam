# app/core/__init__.py
"""Core configuration and utilities"""
from app.core.config import (
    API_V1_STR, PROJECT_NAME, VERSION, CHURN_CLASSES,
    MODELS_DIR, LOGISTIC_PATH, RF_PATH, XGB_PATH
)

__all__ = [
    'API_V1_STR', 'PROJECT_NAME', 'VERSION', 'CHURN_CLASSES',
    'MODELS_DIR', 'LOGISTIC_PATH', 'RF_PATH', 'XGB_PATH'
]