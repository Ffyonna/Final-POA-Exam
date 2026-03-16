# app/services/prediction.py
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import os
import sys
import csv
from datetime import datetime
import json

from app.core.config import (
    LOGISTIC_PATH, RF_PATH, BEST_MODEL_PATH, FEATURES_PATH, CHURN_CLASSES, XGB_PATH, PREDICTIONS_DIR, PREDICTIONS_FILE
)
#from app.services.helpers import model, load_everything
from app.services import helpers
from models.feature_eng import prepare_features

# # DEBUG: Check if model files exist
# print("\n" + "="*60)
# print("DEBUG: Checking model files")
# print("="*60)
# print(f"BEST_MODEL_PATH: {BEST_MODEL_PATH}")
# print(f"File exists: {os.path.exists(BEST_MODEL_PATH)}")
# print(f"LOGISTIC_PATH: {LOGISTIC_PATH}")
# print(f"File exists: {os.path.exists(LOGISTIC_PATH)}")
# print(f"RF_PATH: {RF_PATH}")
# print(f"File exists: {os.path.exists(RF_PATH)}")
# print(f"XGB_PATH: {XGB_PATH}")
# print(f"File exists: {os.path.exists(XGB_PATH)}")
# print(f"FEATURES_PATH: {FEATURES_PATH}")
# print(f"File exists: {os.path.exists(FEATURES_PATH)}")
# print("="*60 + "\n")



# Load other required objects
logistic_pipeline = joblib.load(LOGISTIC_PATH)
rf_pipeline = joblib.load(RF_PATH)
xgb_pipeline = joblib.load(XGB_PATH)
# label_encoder = joblib.load(ENCODER_PATH)
feature_names = joblib.load(FEATURES_PATH)

# Load models and explainers
print("Loading models and explainers...")
model_name = helpers.load_everything(feature_names)

# Set current pipeline and model name from helpers
current_pipeline = helpers.model
current_model_name = model_name

print(f"current_pipeline: {current_pipeline}")
print(f"current_model_name: {current_model_name}")

# Use model from helpers as default
#current_pipeline = helpers.model
#current_model_name = model_name

def save_prediction_to_file(prediction_result: Dict[str, Any], input_data: Dict[str, Any]):
    """Save prediction to CSV file"""
    # Create directory if it doesn't exist
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    
    # Prepare row - include input data as JSON string
    row = {
        'timestamp': datetime.now().isoformat(),
        'customer_number': prediction_result['customer_number'],
        'predicted_class': prediction_result['predicted_class'],
        'confidence': prediction_result['confidence'],
        'model_used': prediction_result['model_used'],
        'input_data': json.dumps(input_data),  # Store input as JSON
        'probabilities': json.dumps(prediction_result['probabilities'])  # Store probabilities
    }
    
    # Write to CSV
    file_exists = os.path.isfile(PREDICTIONS_FILE)
    
    with open(PREDICTIONS_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def save_batch_summary(batch_result: Dict[str, Any], filename: str):
    """Save batch prediction summary"""
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    
    summary_file = PREDICTIONS_DIR / "batch_summaries.csv"
    
    row = {
        'timestamp': datetime.now().isoformat(),
        'filename': filename,
        'total_customers': batch_result['summary']['total'],
        'successful': batch_result['summary']['successful'],
        'failed': batch_result['summary']['failed'],
        'avg_confidence': batch_result['summary']['avg_confidence'],
        'model_used': batch_result['summary']['model_used']
    }
    
    file_exists = os.path.isfile(summary_file)
    
    with open(summary_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# def predict_single(customer_data: Dict[str, Any], 
#                    pipeline=None, 
#                    model_name: str = None) -> Dict[str, Any]:
#     """
#     Predict churn for a single customer
#     """
#     if pipeline is None:
#         pipeline = current_pipeline
#         model_name = current_model_name
    
#     # Prepare features
#     df = prepare_features(customer_data)
    
#     # Ensure all required features are present
#     for col in feature_names:
#         if col not in df.columns:
#             df[col] = 0
    
#     X = df[feature_names]
    
#     # Make prediction (pipeline handles preprocessing automatically)
#     pred_idx = pipeline.predict(X)[0]
#     # Map numeric prediction to class name using CHURN_CLASSES from config
#     pred_class = CHURN_CLASSES[int(pred_idx)]
    
#     # Get probabilities
#     probs = pipeline.predict_proba(X)[0]
#     prob_dict = {
#         CHURN_CLASSES[i]: float(prob) 
#         for i, prob in enumerate(probs)
#     }
    
#     return {
#         'customer_number': customer_data.get('customer_number', 'unknown'),
#         'predicted_class': pred_class,
#         'probabilities': prob_dict,
#         'confidence': float(max(probs)),
#         'model_used': model_name
#     }


def predict_single(customer_data: Dict[str, Any], 
                   pipeline=None, 
                   model_name: str = None) -> Dict[str, Any]:
    """
    Predict churn for a single customer
    """
    
    # print("\n" + "="*60)
    # print("DEBUG: predict_single called")
    # print("="*60)
    # sys.stdout.flush()
    
    if pipeline is None:
        pipeline = current_pipeline
        model_name = current_model_name
    
    # # Convert dict to DataFrame FIRST
    print("Converting dict to DataFrame...")
    sys.stdout.flush()
    df_input = pd.DataFrame([customer_data])  # CRITICAL: Convert dict to DataFrame

    # # Convert all column names to UPPERCASE to match your feature_eng.py expectations
    df_input.columns = [col.upper() for col in df_input.columns]
    # print(f"Columns after uppercase conversion: {df_input.columns.tolist()}")
    
    # # Now pass DataFrame to prepare_features
    print("Calling prepare_features with DataFrame...")
    sys.stdout.flush()
    df = prepare_features(df_input)  # Pass DataFrame, not dict
    
    # print(f"df columns after prepare_features: {df.columns.tolist()}")
    # print(f"feature_names expected: {feature_names}")
    # sys.stdout.flush()
    
    # # Check for DAYS_SINCE_LAST_TRANSACTION specifically
    # if 'DAYS_SINCE_LAST_TRANSACTION' in df.columns:
    #     print("✓ DAYS_SINCE_LAST_TRANSACTION is in df")
    # else:
    #     print("✗ DAYS_SINCE_LAST_TRANSACTION is NOT in df")
    
    # Ensure all required features are present
    for col in feature_names:
        if col not in df.columns:
            print(f"ERROR: {col} not found in df!")
            print(f"Available columns: {df.columns.tolist()}")
            df[col] = 0
    
    X = df[feature_names]
    print(f"X shape: {X.shape}")
    print(f"X columns: {X.columns.tolist()}")
    sys.stdout.flush()
    
    # Make prediction
    pred_idx = pipeline.predict(X)[0]
    pred_class = CHURN_CLASSES[int(pred_idx)]
    
    # Get probabilities
    probs = pipeline.predict_proba(X)[0]
    prob_dict = {
        CHURN_CLASSES[i]: float(prob) 
        for i, prob in enumerate(probs)
    }
    
    print("Prediction successful!")
    print("="*60 + "\n")
    sys.stdout.flush()

    customer_number = (
        customer_data.get('customer_number') or 
        customer_data.get('CUSTOMER_NUMBER') or 
        'unknown'
    )
    
    result = {
        'customer_number': customer_number,
        'predicted_class': pred_class,
        'probabilities': prob_dict,
        'confidence': float(max(probs)),
        'model_used': model_name
    }

    # Save to file (pass a copy of input data without modifying original)
    save_prediction_to_file(result, customer_data.copy())

    return result


# def predict_batch(customers_list: List[Dict[str, Any]]) -> Dict[str, Any]:
#     """Predict for multiple customers"""
#     results = []
#     errors = []
    
#     for i, customer in enumerate(customers_list):
#         try:
#             pred = predict_single(customer)
#             results.append(pred)
#         except Exception as e:
#             errors.append({"row": i, "error": str(e)})
    
#     # Generate summary
#     class_counts = {}
#     for r in results:
#         cls = r['predicted_class']
#         class_counts[cls] = class_counts.get(cls, 0) + 1
    
#     return {
#         'predictions': results,
#         'summary': {
#             'total': len(results),
#             'successful': len(results),
#             'failed': len(errors),
#             'class_distribution': class_counts,
#             'avg_confidence': float(np.mean([r['confidence'] for r in results])) if results else 0,
#             'model_used': current_model_name
#         },
#         'errors': errors if errors else None
#     }


def predict_batch(customers_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Predict for multiple customers"""
    results = []
    errors = []
    
    for i, customer in enumerate(customers_list):
        try:
            pred = predict_single(customer)
            results.append(pred)
        except Exception as e:
            error_msg = str(e)
            print(f"Error on row {i}: {error_msg}")
            errors.append({"row": i, "error": error_msg, "data": customer})
    
    # Generate summary
    class_counts = {}
    for r in results:
        cls = r['predicted_class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    result =  {
        'predictions': results,
        'summary': {
            'total': len(customers_list),
            'successful': len(results),
            'failed': len(errors),
            'class_distribution': class_counts,
            'avg_confidence': float(np.mean([r['confidence'] for r in results])) if results else 0,
            'model_used': current_model_name
        },
        'errors': errors
    }

    # Save batch summary if filename provided
    if 'filename' in customers_list[0] if customers_list else False:
        save_batch_summary(result, customers_list[0].get('filename', 'unknown'))

        return result

    
def compare_models(customer_data: Dict[str, Any]) -> Dict[str, Any]:
    """Compare predictions from all models"""
    pipelines = [
        (logistic_pipeline, "Logistic"),
        (rf_pipeline, "Random Forest"),
        (xgb_pipeline, "XGBoost")
    ]
    
    results = {}
    for pipeline, name in pipelines:
        pred = predict_single(customer_data, pipeline, name)
        results[name] = {
            'predicted_class': pred['predicted_class'],
            'confidence': pred['confidence'],
            'probabilities': pred['probabilities']
        }
    
    return {
        'customer_number': customer_data.get('customer_number', 'unknown'),
        'model_comparison': results,
        'consensus': len(set([r['predicted_class'] for r in results.values()])) == 1
    }