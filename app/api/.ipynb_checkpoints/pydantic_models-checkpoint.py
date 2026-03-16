# app/api/pydantic_models.py

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal
from datetime import datetime

class CustomerFeatures(BaseModel):
    """Input features for a single customer prediction"""
    customer_number: str = Field(alias="CUSTOMER_NUMBER")
    no_debit_cards_last_6m: int = Field(alias="NO_DEBIT_CARDS_LAST_6M", ge=0)
    no_debit_cards_previous_6_months: int = Field(alias="NO_DEBIT_CARDS_PREVIOUS_6_MONTHS", ge=0)
    no_credit_cards_last_6m: int = Field(alias="NO_CREDIT_CARDS_LAST_6M", ge=0)
    no_credit_cards_previous_6_months: int = Field(alias="NO_CREDIT_CARDS_PREVIOUS_6_MONTHS", ge=0)
    current_acc: float = Field(alias="CURRENT_ACC", ge=0)
    savings_acc: float = Field(alias="SAVINGS_ACC", ge=0)
    termdep_acc: float = Field(alias="TERMDEP_ACC", ge=0)
    assetfin_acc: float = Field(alias="ASSETFIN_ACC", ge=0)
    mobileloan_acc: float = Field(alias="MOBILELOAN_ACC", ge=0)
    ipf_acc: float = Field(alias="IPF_ACC", ge=0)
    mortgage_acc: float = Field(alias="MORTGAGE_ACC", ge=0)
    termloan_acc: float = Field(alias="TERMLOAN_ACC", ge=0)
    other_acc: float = Field(alias="OTHER_ACC", ge=0)
    mobile_banking: float = Field(alias="MOBILE_BANKING", ge=0)
    days_since_last_transaction: int = Field(alias="DAYS_SINCE_LAST_TRANSACTION", ge=0)
    revenue_last_6m: float = Field(alias="REVENUE_LAST_6M")
    tenor: int = Field(alias="TENOR", ge=0)
    dr_txns_last_6m: int = Field(alias="DR_TXNS_LAST_6M", ge=0)
    cr_txns_last_6m: int = Field(alias="CR_TXNS_LAST_6M", ge=0)
    cr_amnt_last_6m: float = Field(alias="CR_AMNT_LAST_6M", ge=0)
    dr_amnt_last_6m: float = Field(alias="DR_AMNT_LAST_6M")
    # no_t24_accounts_previous_6_months: int = Field(alias="NO_T24_ACCOUNTS_PREVIOUS_6_MONTHS", ge=0)
    # no_t24_products_previous_6_months: int = Field(alias="NO_T24_PRODUCTS_PREVIOUS_6_MONTHS", ge=0)
    revenue_previous_6_months: float = Field(alias="REVENUE_PREVIOUS_6_MONTHS", ge=0)
    dr_txns_previous_6_months: int = Field(alias="DR_TXNS_PREVIOUS_6_MONTHS", ge=0)
    dr_amnt_previous_6_months: float = Field(alias="DR_AMNT_PREVIOUS_6_MONTHS")
    cr_txns_previous_6_months: int = Field(alias="CR_TXNS_PREVIOUS_6_MONTHS", ge=0)
    cr_amnt_previous_6_months: float = Field(alias="CR_AMNT_PREVIOUS_6_MONTHS", ge=0)
    customer_age: int = Field(alias="CUSTOMER_AGE", ge=0)
    

class Config:
        populate_by_name = True
    
    # no_debit_cards_change: int = Field(alias="NO_DEBIT_CARDS_CHANGE")
    # no_credit_cards_change: int = Field(alias="NO_CREDIT_CARDS_CHANGE")
    # dr_txns_change: int = Field(alias="DR_TXNS_CHANGE")
    # dr_amnt_change: float = Field(alias="DR_AMNT_CHANGE")
    # cr_txns_change: int = Field(alias="CR_TXNS_CHANGE")
    # cr_amnt_change: float = Field(alias="CR_AMNT_CHANGE")
    # revenue_change: float = Field(alias="REVENUE_CHANGE")

class PredictionResponse(BaseModel):
    customer_number: str
    predicted_class: str
    probabilities: Dict[str, float]
    confidence: float
    model_used: str
    timestamp: datetime = datetime.now()

# class ExplanationResponse(BaseModel):
#     customer_number: str
#     predicted_class: str
#     shap_values: Optional[Dict[str, float]] = None
#     #explanation: Dict[str, float]
#     top_features: List[Dict[str, float]]


class FeatureImportance(BaseModel):
    feature: str
    value: float

class ExplanationResponse(BaseModel):
    customer_number: str
    predicted_class: str
    shap_values: Optional[Dict[str, float]] = None
    top_features: List[FeatureImportance]  # Use nested model