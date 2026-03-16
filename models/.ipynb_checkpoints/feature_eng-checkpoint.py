# models/feature_eng.py
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor




NUMERIC_FEATURES = [
     'NO_DEBIT_CARDS_LAST_6M', 'NO_CREDIT_CARDS_LAST_6M', 'CURRENT_ACC', 'SAVINGS_ACC', 'TERMDEP_ACC', 'ASSETFIN_ACC', 'MOBILELOAN_ACC', 'IPF_ACC', 'MORTGAGE_ACC', 'TERMLOAN_ACC','OTHER_ACC', 'MOBILE_BANKING',
     'REVENUE_LAST_6M', 'TENOR', 'DR_TXNS_LAST_6M', 'CR_TXNS_LAST_6M', 'CR_AMNT_LAST_6M', 
     'NO_DEBIT_CARDS_CHANGE', 'NO_CREDIT_CARDS_CHANGE', 'DR_TXNS_CHANGE', 'DR_AMNT_CHANGE', 'CR_TXNS_CHANGE', 'CR_AMNT_CHANGE', 'REVENUE_CHANGE'
] # DAYS_SINCE_LAST_TRANSACTION

CATEGORICAL_FEATURES = []

# ======================== TARGET VARIABLE DEFINITION =============================================
def target_variable_definition(df: pd.DataFrame) -> pd.DataFrame:
    print("DEBUG: Entering target_variable_definition")
    print(f"DEBUG: df columns: {df.columns.tolist()}")
    print(f"DEBUG: Does 'DAYS_SINCE_LAST_TRANSACTION' exist? {'YES' if 'DAYS_SINCE_LAST_TRANSACTION' in df.columns else 'NO'}")
    
    # Check for nulls
    null_count = df['DAYS_SINCE_LAST_TRANSACTION'].isnull().sum() if 'DAYS_SINCE_LAST_TRANSACTION' in df.columns else 'N/A'
    print(f"DEBUG: Nulls in DAYS_SINCE_LAST_TRANSACTION: {null_count}")

    
    # Vectorized for speed
    conditions = [
        df['DAYS_SINCE_LAST_TRANSACTION'].between(0, 35, inclusive='both'),
        df['DAYS_SINCE_LAST_TRANSACTION'].between(36, 95, inclusive='both'),
        df['DAYS_SINCE_LAST_TRANSACTION'].between(96, 180, inclusive='both'),
        (df['DAYS_SINCE_LAST_TRANSACTION'].isnull()) | (df['DAYS_SINCE_LAST_TRANSACTION'] > 180)
    ]
    values_target = [0, 1, 2, 3]
    values_label = ['GOOD', 'EARLY SIGNS', 'HIGHLY LIKELY', 'CHURNED']

    df['CHURN_TARGET'] = np.select(conditions, values_target, default=3)
    df['CHURN_LABEL'] = np.select(conditions, values_label, default='CHURNED')
    return df


# ======================== OUTLIER TREATMENT =============================================
def winsorization_for_outliers(df: pd.DataFrame) -> pd.DataFrame:
    
    cols_to_cap = ['DAYS_SINCE_LAST_TRANSACTION', 'DR_TXNS_LAST_6M',
       'DR_TXNS_PREVIOUS_6_MONTHS', 'DR_AMNT_LAST_6M',
       'DR_AMNT_PREVIOUS_6_MONTHS', 'CR_TXNS_LAST_6M',
       'CR_TXNS_PREVIOUS_6_MONTHS', 'CR_AMNT_LAST_6M',
       'CR_AMNT_PREVIOUS_6_MONTHS', 'REVENUE_LAST_6M',
       'REVENUE_PREVIOUS_6_MONTHS'
              ]

    # Cap the outliers at the 99th percentile
    for col in cols_to_cap:
        upper_limit = df[col].quantile(0.99)
        df[col] = df[col].clip(upper = upper_limit)
        df[col] = df[col].clip(upper = upper_limit) 

    ### For age there are those below 18, and some extreme outliers, both are taken care of here
    age_99th = df['CUSTOMER_AGE'].quantile(0.99)
    df['CUSTOMER_AGE'] = df['CUSTOMER_AGE'].clip(lower = 18, upper = age_99th)

    return df

def add_velocity_features(df, previous_6m_cols):
    df = df.copy()
    
    for col_prev in previous_6m_cols:
        col_recent = col_prev.replace('_PREVIOUS_6_MONTHS', '_LAST_6M')
        change_col = col_prev.replace('_PREVIOUS_6_MONTHS', '_CHANGE')

        inactive_mask = (df[col_recent] == 0) & (df[col_prev] == 0)
        
        # Identify if the column is an outflow/debit
        is_debit = any(x in col_recent.upper() for x in ['DR_AMNT'])

        with np.errstate(divide='ignore', invalid='ignore'):
            if is_debit:
                df[change_col] = (df[col_recent] - df[col_prev]) / df[col_prev].abs()
            else:
                df[change_col] = (df[col_recent] - df[col_prev]) / df[col_prev].abs()

        # Cleanup
        df[change_col] = df[change_col].replace([np.inf], 100).replace([-np.inf], -100)

 
#       # We explicitly set this to differentiate from 2/2 stability which is 0
        df.loc[inactive_mask, change_col] = -1
        
        df[change_col] = df[change_col].fillna(0)
        df[change_col] = df[change_col].replace([None], 0)

    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    previous_6m_cols  = [
    # 'NO_T24_ACCOUNTS_PREVIOUS_6_MONTHS','NO_T24_PRODUCTS_PREVIOUS_6_MONTHS',
    'NO_DEBIT_CARDS_PREVIOUS_6_MONTHS','NO_CREDIT_CARDS_PREVIOUS_6_MONTHS',
    'DR_TXNS_PREVIOUS_6_MONTHS','DR_AMNT_PREVIOUS_6_MONTHS',
    'CR_TXNS_PREVIOUS_6_MONTHS','CR_AMNT_PREVIOUS_6_MONTHS','REVENUE_PREVIOUS_6_MONTHS'
                    ]
    
    df = add_velocity_features(df, previous_6m_cols)

    return df


def drop_correlated_vars(df: pd.DataFrame) -> pd.DataFrame:

    cols_to_exclude = ['CUSTOMER_NUMBER','CUSTOMER_BIRTHDATE', 'BUSINESS_SEGMENT_DESC', 'BUSINESS_SEGMENT',
       'SUB_SEGEMENT_DESC', 'CUS_SHORT_NAME', 'SECTOR_DESCRIPTION',
       'INDUSTRY_DESCRIPTION', 'CUSTOMER_STATUS', 'STATUS_DESC',
       'SUB_SEGEMENT', 'CUSTOMER_BRANCH_NAME', 'CUS_ACC_OFFICER',
       'ACCOUNT_OFFICER_NAME', 'AREA_SECTOR','CHURN_LABEL'] 

    
    cols_to_drop = [
    # Redundant aggregates
    # 'NO_T24_ACCOUNTS_PREVIOUS_6_MONTHS','NO_T24_PRODUCTS_PREVIOUS_6_MONTHS',
        'NO_DEBIT_CARDS_PREVIOUS_6_MONTHS','NO_CREDIT_CARDS_PREVIOUS_6_MONTHS',
    'DR_TXNS_PREVIOUS_6_MONTHS','CR_TXNS_PREVIOUS_6_MONTHS','DR_AMNT_PREVIOUS_6_MONTHS','CR_AMNT_PREVIOUS_6_MONTHS','REVENUE_PREVIOUS_6_MONTHS',
        
    # Static duplicates
    'DEBIT_CARDS', 'CREDIT_CARDS',

    # HIGH VIF
    # 'NO_T24_ACCOUNTS_CHANGE','NO_T24_PRODUCTS_CHANGE',
        'DR_AMNT_LAST_6M','NO_T24_PRODUCTS_LAST_6M','CUSTOMER_AGE','NO_T24_ACCOUNTS_LAST_6M'
    
    ]

    # Drop the columns from your dataframe
    df_cleaned = df.drop(columns = cols_to_drop, errors = 'ignore')
    
    
    df_copy_1 = df_cleaned.select_dtypes(include='number').copy()
    
    include_cols = [col for col in df_copy_1.columns if col not in cols_to_exclude]
    df_copy_1 = df_copy_1[include_cols]
    df_copy_1 = df_copy_1.fillna(df_copy_1.mean())
    
    # Check for infinite values and replace them with a large finite value
    df_copy_1.replace([np.inf, -np.inf], np.finfo(np.float64).max, inplace=True)
    
    # Compute VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df_copy_1.columns
    vif_data["VIF"] = [variance_inflation_factor(df_copy_1.values, i) for i in range(len(df_copy_1.columns))]

    uncorr_cols = vif_data['Feature'].to_list()

    cols_to_remain = [x for x in df.columns if x in uncorr_cols]
    cols_to_remain = ['CUSTOMER_NUMBER'] +  cols_to_remain
    
    df = df[cols_to_remain]

    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Target Drfinition....")
    df = target_variable_definition(df)

    # 2. Winsorization
    print("Winsorization in progress......")
    df = winsorization_for_outliers(df)

    # 3. Feature Engineering
    print("Feature Engineering in progress......")
    df = feature_engineering(df)

    # 4. Checking Correlation
    print("Correlation Handling in progress......")
    df = drop_correlated_vars(df)

    return df

def get_feature_columns():
    """Return list of feature columns used by model"""
    return [
        'NO_DEBIT_CARDS_LAST_6M', 'NO_CREDIT_CARDS_LAST_6M', 'CURRENT_ACC', 'SAVINGS_ACC', 'TERMDEP_ACC', 'ASSETFIN_ACC', 'MOBILELOAN_ACC', 'IPF_ACC', 'MORTGAGE_ACC', 'TERMLOAN_ACC', 'OTHER_ACC', 'MOBILE_BANKING', 'REVENUE_LAST_6M', 'TENOR', 'DR_TXNS_LAST_6M', 'CR_TXNS_LAST_6M', 'CR_AMNT_LAST_6M', 'NO_DEBIT_CARDS_CHANGE', 'NO_CREDIT_CARDS_CHANGE', 'DR_TXNS_CHANGE', 'DR_AMNT_CHANGE', 'CR_TXNS_CHANGE', 'CR_AMNT_CHANGE', 'REVENUE_CHANGE'
    ] # DAYS_SINCE_LAST_TRANSACTION