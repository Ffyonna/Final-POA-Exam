# app/services/explainability.py
import pandas as pd
import numpy as np
from typing import Dict, Any

from app.services.helpers import model, explainer_shap, explainer_lime
from app.services.prediction import predict_single
from app.core.config import CHURN_CLASSES
from models.feature_eng import prepare_features

# def get_shap_values(customer_data: pd.DataFrame) -> Dict[str, Any]:  
#     """
#     Get SHAP explanations for a prediction
#     """
#     # Get prediction first
#     prediction = predict_single(customer_data.iloc[0].to_dict()) 
    
#     # Prepare features
#     df = prepare_features(customer_data)
    
#     # Get feature names from preprocessor
#     feature_names = model.named_steps["preprocessor"].get_feature_names_out().tolist()
    
#     # Transform data using preprocessor
#     X_transformed = model.named_steps["preprocessor"].transform(df)
    
#     # Get SHAP values
#     shap_values = explainer_shap.shap_values(X_transformed)
    
#     # Get predicted class index using CHURN_CLASSES
#     class_to_idx = {cls: i for i, cls in enumerate(CHURN_CLASSES)}
#     pred_idx = class_to_idx[prediction['predicted_class']]
    
#     # Extract values for predicted class
#     if isinstance(shap_values, list):
#         shap_vals = shap_values[pred_idx][0]
#     else:
#         shap_vals = shap_values[0]
    
#     # Convert to dictionary
#     shap_dict = {}
#     for i, feat in enumerate(feature_names[:len(shap_vals)]):
#         shap_dict[feat] = float(shap_vals[i])
    
#     # Get top 5 features by absolute value
#     top_features = sorted(
#         [{'feature': k, 'value': v} for k, v in shap_dict.items()],
#         key=lambda x: abs(x['value']),
#         reverse=True
#     )[:5]
    
#     return {
#         'customer_number': customer_data.get('customer_number', 'unknown'),
#         'predicted_class': prediction['predicted_class'],
#         'shap_values': shap_dict,
#         'top_features': top_features
#     }


def get_shap_values(customer_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get SHAP explanations for a prediction
    """
    
    print("\n" + "="*60)
    print("DEBUG: get_shap_values called")
    print("="*60)
    
    # Get prediction first
    prediction = predict_single(customer_data)
    print(f"Prediction: {prediction['predicted_class']}")
    
    # Convert dict to DataFrame for feature preparation
    df_input = pd.DataFrame([customer_data])
    df_input.columns = [col.upper() for col in df_input.columns]
    
    # Prepare features
    from models.feature_eng import prepare_features
    df = prepare_features(df_input)
    
    # Get the classifier and preprocessor from the pipeline
    classifier = model.named_steps["classifier"]
    preprocessor = model.named_steps["preprocessor"]
    
    # Transform data
    X_transformed = preprocessor.transform(df)
    
    # Get feature names and clean them (remove prefixes like "num__")
    raw_feature_names = preprocessor.get_feature_names_out().tolist()
    clean_feature_names = [name.replace('num__', '').replace('cat__', '') for name in raw_feature_names]
    print(f"Clean feature names: {clean_feature_names[:5]}...")
    
    # Create explainer if not exists
    global explainer_shap
    if explainer_shap is None:
        import shap
        explainer_shap = shap.TreeExplainer(classifier)
    
    # Get SHAP values
    shap_values = explainer_shap.shap_values(X_transformed)
    
    # Get predicted class index
    pred_idx = CHURN_CLASSES.index(prediction['predicted_class'])
    
    # Handle SHAP output based on shape
    if isinstance(shap_values, list):
        # Multi-class case - shap_values[pred_idx] gives array of shape (1, n_features)
        shap_vals = shap_values[pred_idx][0]
    else:
        # Binary case or single array
        if len(shap_values.shape) == 3:
            # Shape (1, n_features, n_classes)
            shap_vals = shap_values[0, :, pred_idx]
        elif len(shap_values.shape) == 2:
            # Shape (1, n_features)
            shap_vals = shap_values[0]
        else:
            shap_vals = shap_values
    
    print(f"shap_vals final shape: {shap_vals.shape}")
    
    # Convert to dictionary with clean feature names
    shap_dict = {}
    for i, feat in enumerate(clean_feature_names[:len(shap_vals)]):
        val = shap_vals[i]
        # Convert numpy types to Python float
        if hasattr(val, 'item'):
            val = val.item()
        shap_dict[feat] = float(val)
    
    # Get top 5 features by absolute value
    top_features = [
    {'feature': k, 'value': v}  # This will now match FeatureImportance model
    for k, v in sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
]
    
    # Ensure top_features has the correct structure
    formatted_top_features = []
    for item in top_features:
        formatted_top_features.append({
            'feature': item['feature'],
            'value': item['value']
        })
    
    print("="*60 + "\n")
    
    return {
        'customer_number': customer_data.get('customer_number', 'unknown'),
        'predicted_class': prediction['predicted_class'],
        'shap_values': shap_dict,
        'top_features': formatted_top_features
    }


# def get_lime_explanation(customer_data: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Get LIME explanations for a prediction
#     """
#     # Get prediction first
#     prediction = predict_single(customer_data)
    
#     # Prepare features
#     df = prepare_features(customer_data)
    
#     # Transform data using preprocessor
#     X_transformed = model.named_steps["preprocessor"].transform(df)
    
#     # Get LIME explanation
#     exp = explainer_lime.explain_instance(
#         X_transformed[0],
#         model.named_steps["classifier"].predict_proba,
#         num_features=10
#     )
    
#     # Convert to dictionary
#     lime_weights = dict(exp.as_list())
    
#     return {
#         'customer_number': customer_data.get('customer_number', 'unknown'),
#         'predicted_class': prediction['predicted_class'],
#         'lime_weights': lime_weights
#     }


# def get_lime_explanation(customer_data: Dict[str, Any]) -> Dict[str, Any]:  # Expect dict
#     """
#     Get LIME explanations for a prediction
#     """
#     import pandas as pd
#     import numpy as np
    
#     print("\n" + "="*60)
#     print("DEBUG: get_lime_explanation called")
#     print("="*60)
    
#     # customer_data should be a dict here
#     print(f"customer_data type: {type(customer_data)}")
    
#     # Get prediction first - pass the dict
#     prediction = predict_single(customer_data)
#     print(f"Prediction: {prediction['predicted_class']}")
    
#     # Convert dict to DataFrame for feature preparation
#     df_input = pd.DataFrame([customer_data])
#     df_input.columns = [col.upper() for col in df_input.columns]
    
#     # Prepare features
#     df = prepare_features(df_input)
    
#     # Rest of the function remains the same...
#     # Get the classifier and preprocessor from the pipeline
#     classifier = model.named_steps["classifier"]
#     preprocessor = model.named_steps["preprocessor"]
    
#     # Transform data
#     X_transformed = preprocessor.transform(df)
#     print(f"X_transformed shape: {X_transformed.shape}")
    
#     # Ensure X_transformed is 2D
#     if len(X_transformed.shape) == 3:
#         X_transformed = X_transformed[0]
#     if len(X_transformed.shape) == 1:
#         X_transformed = X_transformed.reshape(1, -1)
    
#     # Get feature names and clean them
#     raw_feature_names = preprocessor.get_feature_names_out().tolist()
#     clean_feature_names = [name.replace('num__', '').replace('cat__', '') for name in raw_feature_names]
    
#     # Get LIME explanation
#     exp = explainer_lime.explain_instance(
#         X_transformed[0],
#         classifier.predict_proba,
#         num_features=10
#     )
    
#     # Convert to dictionary
#     lime_weights = dict(exp.as_list())
    
#     print("="*60 + "\n")
    
#     return {
#         'customer_number': customer_data.get('customer_number', 'unknown'),
#         'predicted_class': prediction['predicted_class'],
#         'lime_weights': lime_weights
#     }

# LATEST WORKING VERSION WITH NO DEBUG
# def get_lime_explanation(customer_data: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Get LIME explanations for a prediction
#     """
    
#     print("\n" + "="*60)
#     print("DEBUG: get_lime_explanation called")
#     print("="*60)
    
#     # Get prediction first
#     prediction = predict_single(customer_data)
#     print(f"Prediction: {prediction['predicted_class']}")
    
#     # Convert dict to DataFrame for feature preparation
#     df_input = pd.DataFrame([customer_data])
#     df_input.columns = [col.upper() for col in df_input.columns]
    
#     # Prepare features
#     df = prepare_features(df_input)
    
#     # Get the preprocessor
#     preprocessor = model.named_steps["preprocessor"]
#     classifier = model.named_steps["classifier"]
    
#     # Transform data
#     X_transformed = preprocessor.transform(df)
#     print(f"X_transformed shape: {X_transformed.shape}")
    
#     # Ensure 2D
#     if len(X_transformed.shape) == 3:
#         X_transformed = X_transformed[0]
#     if len(X_transformed.shape) == 1:
#         X_transformed = X_transformed.reshape(1, -1)
    
#     # Get clean feature names
#     raw_feature_names = preprocessor.get_feature_names_out().tolist()
#     clean_feature_names = [name.replace('num__', '').replace('cat__', '') for name in raw_feature_names]
    
#     # Get LIME explanation
#     exp = explainer_lime.explain_instance(
#         X_transformed[0],
#         classifier.predict_proba,
#         num_features=10,
#         num_samples=1000  # Add more samples for better explanation
#     )
    
#     # Convert to dictionary
#     lime_weights = dict(exp.as_list())
#     print(f"LIME weights: {lime_weights}")
    
#     return {
#         'customer_number': customer_data.get('customer_number', 'unknown'),
#         'predicted_class': prediction['predicted_class'],
#         'lime_weights': lime_weights
#     }



def get_lime_explanation(customer_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get LIME explanations for a prediction
    """
    import pandas as pd
    import numpy as np
    
    print("\n" + "="*60)
    print("DEBUG: get_lime_explanation called")
    print("="*60)
    
    # Get prediction first
    prediction = predict_single(customer_data)
    print(f"Prediction: {prediction['predicted_class']}")
    print(f"Prediction probabilities: {prediction['probabilities']}")
    
    # Convert dict to DataFrame for feature preparation
    df_input = pd.DataFrame([customer_data])
    df_input.columns = [col.upper() for col in df_input.columns]
    
    # Prepare features
    from models.feature_eng import prepare_features
    df = prepare_features(df_input)
    print(f"df columns: {df.columns.tolist()}")
    print(f"df values: {df.values}")
    
    # Get the preprocessor
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    
    # Transform data
    X_transformed = preprocessor.transform(df)
    print(f"X_transformed type: {type(X_transformed)}")
    print(f"X_transformed shape: {X_transformed.shape}")
    print(f"X_transformed values: {X_transformed}")
    
    # Ensure 2D
    if len(X_transformed.shape) == 3:
        X_transformed = X_transformed[0]
    if len(X_transformed.shape) == 1:
        X_transformed = X_transformed.reshape(1, -1)
    
    print(f"Final X_transformed shape: {X_transformed.shape}")
    print(f"Final X_transformed[0]: {X_transformed[0]}")
    
    # Get clean feature names
    raw_feature_names = preprocessor.get_feature_names_out().tolist()
    clean_feature_names = [name.replace('num__', '').replace('cat__', '') for name in raw_feature_names]
    print(f"Feature names: {clean_feature_names}")
    
    # Check LIME explainer
    print(f"LIME explainer exists: {explainer_lime is not None}")
    if explainer_lime is not None:
        print(f"LIME training data shape: {explainer_lime.training_data.shape if hasattr(explainer_lime, 'training_data') else 'N/A'}")
    
    # Get LIME explanation with more verbosity
    try:
        exp = explainer_lime.explain_instance(
            X_transformed[0],
            classifier.predict_proba,
            num_features=len(clean_feature_names),  # Show all features
            num_samples=5000,  # More samples
            distance_metric='euclidean',
            model_regressor=None
        )
        
        # Get the explanation as list
        exp_list = exp.as_list()
        print(f"LIME explanation list: {exp_list}")
        
        # Convert to dictionary
        lime_weights = dict(exp_list)
        print(f"LIME weights: {lime_weights}")
        
    except Exception as e:
        print(f"LIME error: {e}")
        import traceback
        traceback.print_exc()
        lime_weights = {}
    
    print("="*60 + "\n")
    
    return {
        'customer_number': customer_data.get('customer_number', 'unknown'),
        'predicted_class': prediction['predicted_class'],
        'lime_weights': lime_weights
    }