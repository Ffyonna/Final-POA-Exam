# models/pipelines.py
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from models.feature_eng import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def build_preprocessor():
    # Filter out CUSTOMER_NUMBER from NUMERIC_FEATURES if it exists
    exclude_cols = ['CUSTOMER_NUMBER', 'CHURN_TARGET', 'DAYS_SINCE_LAST_TRANSACTION']
    numeric_features = [f for f in NUMERIC_FEATURES if f not in exclude_cols]


    print("\n" + "="*60)
    print("DEBUG: build_preprocessor")
    print("="*60)
    print(f"Original NUMERIC_FEATURES: {NUMERIC_FEATURES}")
    print(f"Exclude cols: {exclude_cols}")
    print(f"Final numeric_features: {numeric_features}")
    print(f"CATEGORICAL_FEATURES: {CATEGORICAL_FEATURES}")


    return ColumnTransformer([
        ("num", StandardScaler(), numeric_features)
       ,("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
    ])