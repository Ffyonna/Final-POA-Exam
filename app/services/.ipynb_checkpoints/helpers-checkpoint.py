# app/services/helpers.py
import joblib
from app.core.config import BEST_MODEL_PATH, BEST_MODEL_NAME_PATH, BACKGROUND_PATH, CHURN_CLASSES
#from app.services.prediction import feature_names

import shap
import pandas as pd
import numpy as np
import lime.lime_tabular

model = None
explainer_shap = None
explainer_lime = None
background = None


def load_everything(feature_names = None):
    global model, explainer_shap, explainer_lime, background
    
    print(f"Loading model from: {BEST_MODEL_PATH}")
    model = joblib.load(BEST_MODEL_PATH)
    print(f"Model loaded: {type(model)}")
    
    print(f"Loading background from: {BACKGROUND_PATH}")
    background_array = joblib.load(BACKGROUND_PATH)
    print(f"Background loaded: {background_array.shape}")
    #print(f"Background type: {type(background)}")

    # Use provided feature_names or create default ones
    if feature_names is None:
        # Fallback - get from preprocessor
        feature_names = model.named_steps["preprocessor"].get_feature_names_out().tolist()
        feature_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]

        
    # Convert back to DataFrame for LIME with proper column names
    background = pd.DataFrame(background_array, columns=feature_names)
    print(f"Background converted to DataFrame: {background.shape}")
    
    # Get the preprocessor and transform background data
    preprocessor = model.named_steps["preprocessor"]
    background_transformed = preprocessor.transform(background)
    
    # Ensure background_transformed is 2D
    if len(background_transformed.shape) == 3:
        background_transformed = background_transformed.reshape(background_transformed.shape[0], -1)
    
    print(f"Background transformed shape: {background_transformed.shape}")
    print(f"Background transformed type: {type(background_transformed)}")
    
    # Get clean feature names
    raw_feature_names = preprocessor.get_feature_names_out().tolist()
    clean_feature_names = [name.replace('num__', '').replace('cat__', '') for name in raw_feature_names]
    
    # SHAP
    print("Creating SHAP explainer...")
    explainer_shap = shap.TreeExplainer(model.named_steps["classifier"])
    
    # LIME - Convert to numpy array
    if hasattr(background_transformed, 'toarray'):
        background_array = background_transformed.toarray()
    else:
        background_array = np.array(background_transformed)
    
    print(f"Background array shape: {background_array.shape}")
    print(f"Background array type: {type(background_array)}")
    print(f"Background array min: {background_array.min()}, max: {background_array.max()}")
    
    print("Creating LIME explainer...")
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        background_array,
        feature_names=clean_feature_names,
        class_names=CHURN_CLASSES,
        mode="classification",
        discretize_continuous=True,
        random_state=42,
        verbose=True
    )
    
    # Verify LIME explainer works by checking its attributes
    print(f"LIME feature names: {explainer_lime.feature_names[:5]}...")
    print(f"LIME class names: {explainer_lime.class_names}")
    print(f"LIME mode: {explainer_lime.mode}")
    
    # Read best model name
    from app.core.config import BEST_MODEL_NAME_PATH
    with open(BEST_MODEL_NAME_PATH, 'r') as f:
        model_name = f.read().strip()
    
    print(f"✅ Model loaded successfully: {model_name}")
    return model_name

    
# def load_everything():
#     global model, explainer_shap, explainer_lime, background
    
#     print(f"Loading model from: {BEST_MODEL_PATH}")
#     model = joblib.load(BEST_MODEL_PATH)
#     print(f"Model loaded: {type(model)}")
    
        
#     print(f"Loading background from: {BACKGROUND_PATH}")
#     background = joblib.load(BACKGROUND_PATH)
#     print(f"Background loaded: {background.shape if hasattr(background, 'shape') else 'unknown'}")
    
    
#     # SHAP
#     print("Creating SHAP explainer...")
#     explainer_shap = shap.TreeExplainer(model.named_steps["classifier"])


#     # Get the preprocessor and transform background data
#     preprocessor = model.named_steps["preprocessor"]
#     background_transformed = preprocessor.transform(background)
    
#     # Get clean feature names (remove prefixes)
#     raw_feature_names = preprocessor.get_feature_names_out().tolist()
#     clean_feature_names = [name.replace('num__', '').replace('cat__', '') for name in raw_feature_names]
    
#     print(f"Clean feature names (first 5): {clean_feature_names[:5]}")
#     print(f"Background transformed shape: {background_transformed.shape}")
    
# #     # LIME
# #     print("Creating LIME explainer...")
# #     explainer_lime = lime.lime_tabular.LimeTabularExplainer(
# #         background_transformed.values,  # This should be 2D (samples, features)
# #         feature_names=model.named_steps["preprocessor"].get_feature_names_out().tolist(),
# #         class_names=CHURN_CLASSES,
# #         mode="classification",
# #         discretize_continuous=False  # Add this to avoid issues
# # )
#     print("Creating LIME explainer...")
#     explainer_lime = lime.lime_tabular.LimeTabularExplainer(
#         background_transformed,  # Use transformed data, not raw
#         feature_names=clean_feature_names,  # Use clean names
#         class_names=CHURN_CLASSES,
#         mode="classification",
#         discretize_continuous=False,
#         random_state=42
#     )

#     # In helpers.py, add this debug
#     print(f"Background transformed shape: {background_transformed.shape}")
#     print(f"Background transformed min: {background_transformed.min()}, max: {background_transformed.max()}")
#     print(f"Background transformed has NaN: {np.isnan(background_transformed).any()}")
#     print(f"Background transformed has Inf: {np.isinf(background_transformed).any()}")

#     with open(BEST_MODEL_NAME_PATH, 'r') as f:
#         model_name = f.read().strip()
#     print(f"✅ Model loaded successfully: {model_name}")
    

#     return model_name