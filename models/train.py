# models/train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.feature_eng import prepare_features, get_feature_columns, NUMERIC_FEATURES
from models.pipelines import build_preprocessor
from models.optimization import optimize_logistic, optimize_random_forest, optimize_xgboost
from app.core.config import MODELS_DIR, CHURN_CLASSES

def load_and_prepare_data(filepath):
    """Load your actual data"""
    print(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)

    # Store customer_numbers separately for reference (but not for training)
    customer_numbers = df['CUSTOMER_NUMBER'].copy() if 'CUSTOMER_NUMBER' in df.columns else None

    # Apply feature engineering 
    df = prepare_features(df)
    
    # Get features
    feature_cols = get_feature_columns()
    #feature_cols.remove('DAYS_SINCE_LAST_TRANSACTION')
    
    
    X = df[feature_cols]
    y = df['CHURN_TARGET'].values
    
    # print(f"Features shape: {X.shape}")
    # print(f"Class distribution:\n{y.value_counts()}")
    
    return X, y, feature_cols, customer_numbers

def train_models(X, y):
    """Train and compare models with optimization"""
    #X = X.drop('DAYS_SINCE_LAST_TRANSACTION', axis = 'columns')
     #print(f"Does 'DAYS_SINCE_LAST_TRANSACTION' in X? {'YES' if 'DAYS_SINCE_LAST_TRANSACTION' in X.columns else 'NO'}")
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Build preprocessor
    preprocessor = build_preprocessor()

    # DEBUG: Check preprocessor
    print("\nDEBUG: Preprocessor")
    print("="*60)
    print(f"Preprocessor transformers: {preprocessor.transformers}")
    
    results = {}
    metrics_dict = {}

    class_names = CHURN_CLASSES
    
    # 1. Logistic Regression pipeline with optimization
    print("\n" + "="*60)
    print("OPTIMIZING LOGISTIC REGRESSION")
    print("="*60)
    base_log_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000)) # , multi_class='multinomial'
    ])
    
    log_pipeline = optimize_logistic(base_log_pipeline, X_train, y_train)
    y_pred = log_pipeline.predict(X_test)
    
    metrics_dict['Logistic'] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted')
    }
    results['Logistic'] = metrics_dict['Logistic']['accuracy']
    print(f"Test Accuracy: {results['Logistic']:.4f}")
    joblib.dump(log_pipeline, MODELS_DIR / 'logistic.pkl')
    
    # 2. Random Forest pipeline with optimization
    print("\n" + "="*60)
    print("OPTIMIZING RANDOM FOREST")
    print("="*60)
    base_rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    rf_pipeline = optimize_random_forest(base_rf_pipeline, X_train, y_train)
    y_pred = rf_pipeline.predict(X_test)
    
    metrics_dict['Random Forest'] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted')
    }
    results['Random Forest'] = metrics_dict['Random Forest']['accuracy']
    print(f"Test Accuracy: {results['Random Forest']:.4f}")
    joblib.dump(rf_pipeline, MODELS_DIR / 'random_forest.pkl')
    
    # 3. XGBoost pipeline with optimization
    print("\n" + "="*60)
    print("OPTIMIZING XGBOOST")
    print("="*60)
    base_xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(objective='multi:softprob', random_state=42))
    ])
    
    xgb_pipeline = optimize_xgboost(base_xgb_pipeline, X_train, y_train)
    y_pred = xgb_pipeline.predict(X_test)
    
    metrics_dict['XGBoost'] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted')
    }
    results['XGBoost'] = metrics_dict['XGBoost']['accuracy']
    print(f"Test Accuracy: {results['XGBoost']:.4f}")
    joblib.dump(xgb_pipeline, MODELS_DIR / 'xgboost.pkl')
    
    # Save other objects
    #joblib.dump(le, MODELS_DIR / 'label_encoder.pkl')
    joblib.dump(X.columns.tolist(), MODELS_DIR / 'feature_names.pkl')
    
    # Save background data for SHAP/LIME (sample of training data)
    background_sample = X_train.sample(min(100, len(X_train)), random_state=42)

    # Convert to numpy array with simple dtype to avoid pickling issues
    background_array = background_sample.values.astype(np.float64)

    joblib.dump(background_array, MODELS_DIR / 'background_data.pkl')
    print(f"Saved background data with shape {background_array.shape}")

    joblib.dump(metrics_dict, MODELS_DIR / 'model_metrics.pkl')
    print(f"Saved metrics: {metrics_dict}")
    
    # Print comparison
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    for model_name, metrics in metrics_dict.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  F1 (weighted): {metrics['f1']:.4f}")
        print(f"  F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
    
    # Find best model by accuracy
    best_model_name = max(metrics_dict.items(), key=lambda x: x[1]['f1'])[0]
    print(f"\n✅ Best model by F1-Score: {best_model_name}")

    # Save the best model and its name
    if best_model_name == "Logistic":
        best_model = log_pipeline
    elif best_model_name == "Random Forest":
        best_model = rf_pipeline
    else:  # XGBoost
        best_model = xgb_pipeline
    
    joblib.dump(best_model, MODELS_DIR / 'best_model.pkl')
    with open(MODELS_DIR / 'best_model_name.txt', 'w') as f:
        f.write(best_model_name)
    
    print(f"✅ Saved best model ({best_model_name}) to best_model.pkl")
    
    return best_model_name

if __name__ == "__main__":
    # Path to data file
    data_path = "data/cleaned_dataset.csv"
    X, y, feature_names, customer_numbers = load_and_prepare_data(data_path)
    best_model = train_models(X, y)