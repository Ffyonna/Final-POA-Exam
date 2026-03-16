# app/api/endpoints.py
from fastapi import APIRouter, HTTPException, File, UploadFile, Query
from typing import List, Dict, Any
import pandas as pd
import io
import os
import csv
from datetime import datetime

from app.api.pydantic_models import CustomerFeatures, PredictionResponse, ExplanationResponse
from app.services import prediction as pred_svc
from app.services import explainability as exp_svc
from app.core.config import API_V1_STR
import joblib
from app.core.config import METRICS_PATH, PREDICTIONS_FILE
import traceback

router = APIRouter(prefix=API_V1_STR)

# In-memory storage for demo
predictions_history = []

@router.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerFeatures):
    """Predict churn for a single customer"""
    try:
        customer_dict = customer.dict()
        result = pred_svc.predict_single(customer_dict)
        
        predictions_history.append({
            'customer_number': result['customer_number'],
            'prediction': result['predicted_class'],
            'timestamp': result.get('timestamp')
        })
        
        return PredictionResponse(**result) # This validates against the updated model
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @router.post("/predict/batch")
# async def predict_batch(customers: List[CustomerFeatures]):
#     """Predict churn for multiple customers"""
#     try:
#         customers_list = [c.dict() for c in customers]
#         results = pred_svc.predict_batch(customers_list)
#         return results
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @router.post("/predict/batch")
# async def predict_batch(file: UploadFile = File(...)):
#     """Upload CSV file and get batch predictions"""
    
#     try:
#         # Read uploaded file
#         contents = await file.read()
        
#         # Determine file type and read accordingly
#         if file.filename.endswith('.csv'):
#             df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
#         elif file.filename.endswith(('.xlsx', '.xls')):
#             df = pd.read_excel(io.BytesIO(contents))
#         else:
#             raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel file.")
        
#         # Convert DataFrame to list of dicts (each row is a customer)
#         customers_list = df.to_dict('records')
        
#         # Get predictions
#         results = pred_svc.predict_batch(customers_list)
        
#         # Add file info to response
#         return {
#             "filename": file.filename,
#             "total_customers": len(customers_list),
#             "predictions": results['predictions'],
#             "summary": results['summary']
#         }
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

## DEBUG - Above works
@router.post("/predict/batch")
async def predict_batch(file: UploadFile = File(...)):
    """Upload CSV file and get batch predictions"""
    import pandas as pd
    import io
    
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Determine file type and read accordingly
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        elif file.filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel file.")
        
        # Debug: Print column names
        print(f"CSV Columns: {df.columns.tolist()}")
        print(f"First row: {df.iloc[0].to_dict()}")
        
        # Convert DataFrame to list of dicts
        customers_list = df.to_dict('records')

        # Add filename to each record for batch tracking
        for customer in customers_list:
            customer['filename'] = file.filename
        
        # Get predictions
        results = pred_svc.predict_batch(customers_list)
        
        # Add file info to response
        return {
            "filename": file.filename,
            "total_customers": len(customers_list),
            "predictions": results['predictions'],
            "summary": results['summary'],
            "errors": results.get('errors', [])  # Show errors
        }
        
    except Exception as e:
        print(f"Error in batch prediction: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))  


# @router.post("/explain/{customer_number}", response_model=ExplanationResponse)
# async def explain_prediction(customer_number: str, customer: CustomerFeatures):
#     """Get SHAP explanation for a prediction"""
#     try:
#         customer_dict = customer.dict()
#         customer_dict['customer_number'] = customer_number
        
#         # Convert to DataFrame before passing to explain function
#         import pandas as pd
#         df_input = pd.DataFrame([customer_dict])
        
#         explanation = exp_svc.get_shap_values(df_input)  # Pass DataFrame, not dict
#         return ExplanationResponse(**explanation)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@router.post("/explain/{customer_number}", response_model=ExplanationResponse)
async def explain_prediction(customer_number: str, customer: CustomerFeatures):
    """Get SHAP explanation for a prediction"""
    try:
        customer_dict = customer.dict()
        customer_dict['customer_number'] = customer_number
        
        # DEBUG: Print before calling
        print("\n" + "="*60)
        print("DEBUG: explain_prediction endpoint called")
        print(f"customer_number: {customer_number}")
        print(f"customer_dict keys: {customer_dict.keys()}")
        print("="*60)
        
        explanation = exp_svc.get_shap_values(customer_dict)
        return ExplanationResponse(**explanation)
    except Exception as e:
        print(f"ERROR in explain SHAP endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/explain/lime/{customer_number}")
async def explain_lime(customer_number: str, customer: CustomerFeatures):
    """Get LIME explanation for a prediction"""
    try:
        customer_dict = customer.dict()
        customer_dict['customer_number'] = customer_number
        
        explanation = exp_svc.get_lime_explanation(customer_dict)  
        return explanation
    except Exception as e:
        print(f"Error in explain LIME endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare-models/{customer_number}")
async def compare_models(customer_number: str, customer: CustomerFeatures):
    """Compare predictions from all models"""
    try:
        customer_dict = customer.dict()
        customer_dict['customer_number'] = customer_number
        comparison = pred_svc.compare_models(customer_dict)
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_metrics():
    """Get model performance metrics"""
    try:
        metrics_dict = joblib.load(METRICS_PATH)
        
        # Find best model by F1 score (weighted)
        best_model = max(metrics_dict.items(), key=lambda x: x[1]['f1'])[0]
        
        return {
            "model_performance": metrics_dict,
            "best_model": best_model,
            "total_predictions": len(predictions_history)
        }
    except Exception as e:
        # Fallback if metrics file doesn't exist
        return {
            "model_performance": {
                "Logistic": {"accuracy": 0.0, "f1": 0.0, "f1_macro": 0.0, "precision": 0.0, "recall": 0.0},
                "Random Forest": {"accuracy": 0.0, "f1": 0.0, "f1_macro": 0.0, "precision": 0.0, "recall": 0.0},
                "XGBoost": {"accuracy": 0.0, "f1": 0.0, "f1_macro": 0.0, "precision": 0.0, "recall": 0.0}
            },
            "best_model": "Unknown",
            "total_predictions": len(predictions_history),
            "error": "No metrics file found. Please train models first."
        }

# @router.post("/upload-data")
# async def upload_data(file: UploadFile = File(...)):
#     """Upload CSV file with customer data"""
#     try:
#         contents = await file.read()
#         df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
#         return {
#             "filename": file.filename,
#             "rows": len(df),
#             "columns": df.columns.tolist(),
#             "preview": df.head(3).to_dict('records')
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions/history")
async def get_prediction_history(limit: int = 100):
    try:
        # Debug print to your terminal so you can see the absolute path it's checking
        abs_path = os.path.abspath(PREDICTIONS_FILE)
        print(f"DEBUG: Checking for file at {abs_path}")

        if not os.path.exists(PREDICTIONS_FILE):
            return {"total": 0, "predictions": [], "debug_path": abs_path}

        # Check if file is empty
        if os.stat(PREDICTIONS_FILE).st_size == 0:
            return {"total": 0, "predictions": [], "message": "File is physically empty"}

        # Read CSV - using low_memory=False to avoid type guesses
        df = pd.read_csv(PREDICTIONS_FILE, low_memory=False)

        if df.empty:
            return {"total": 0, "predictions": []}

        # Handle the timestamp specifically
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # Drop rows where timestamp couldn't be parsed if they exist
            df = df.dropna(subset=['timestamp'])
            df = df.sort_values(by='timestamp', ascending=False)
        
        # Take only the requested limit
        df = df.head(limit)
        
        # Convert all columns to string/float to ensure JSON compatibility
        predictions = df.fillna("N/A").to_dict(orient='records')

        return {
            "total": len(predictions),
            "predictions": predictions
        }

    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        return {
            "total": 0,
            "predictions": [],
            "error": f"Pandas failed: {str(e)}"
        }

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Churn Prediction API",
        "model": pred_svc.current_model_name,
        "predictions_served": len(predictions_history)
    }