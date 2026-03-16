# app/main.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import router
from app.core.config import API_V1_STR, PROJECT_NAME, VERSION
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title=PROJECT_NAME,
    version=VERSION,
    description="Customer Churn Prediction API with Explainable AI",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to Churn Prediction API",
        "version": VERSION,
        "endpoints": {
            "predict": f"{API_V1_STR}/predict",
            "batch_predict": f"{API_V1_STR}/predict/batch",
            "explain": f"{API_V1_STR}/explain/{{customer_number}}",
            "explain_lime": f"{API_V1_STR}/explain/lime/{{customer_number}}",
            "compare": f"{API_V1_STR}/compare-models/{{customer_number}}",
            "metrics": f"{API_V1_STR}/metrics",
            "health": f"{API_V1_STR}/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)