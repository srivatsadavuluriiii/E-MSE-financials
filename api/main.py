from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(
    title="SME Revenue Forecasting API",
    description="Predict future revenue patterns for Small and Medium Enterprises using Machine Learning",
    version="1.0.0"
)

# Load model - use absolute paths relative to this file
MODEL_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "datasets"

try:
    model = pickle.load(open(MODEL_DIR / "cashflow_model.pkl", "rb"))
    feature_map = pickle.load(open(MODEL_DIR / "feature_map.pkl", "rb"))
    train = pd.read_csv(DATA_DIR / "train.csv", low_memory=False)
    store = pd.read_csv(DATA_DIR / "store.csv")
    feature_cols = [feature_map[i] for i in range(len(feature_map))]
    print(" Model and data loaded successfully")
except Exception as e:
    print(f" Error loading model: {e}")
    model = None
    feature_map = {}
    feature_cols = []

# Pydantic models for request/response
class StoreForecastRequest(BaseModel):
    store_id: int
    forecast_days: int = 30

class CashFlowInput(BaseModel):
    recent_sales: List[float]  # Last 7-30 days of sales
    day_of_week: int
    has_promo: bool = False
    is_holiday: bool = False

class ForecastResponse(BaseModel):
    store_id: int
    store_type: str
    predicted_daily_revenue: float
    forecast_30d_revenue: float
    forecast_60d_revenue: float
    forecast_90d_revenue: float
    confidence_lower: float
    confidence_upper: float
    recent_average_revenue: float
    trend: str  # "increasing", "stable", "decreasing"

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "SME Cash Flow Prediction API",
        "version": "1.0.0",
        "description": "Predict future cash flows for SMEs using ML",
        "endpoints": {
            "/health": "Check API health",
            "/predict/store/{store_id}": "Predict for a specific store",
            "/stores": "Get list of available stores",
            "/top-stores": "Get top performing stores"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stores")
async def get_stores():
    """Get list of all available stores"""
    try:
        available_stores = sorted(train['Store'].unique().tolist())
        return {
            "total_stores": len(available_stores),
            "stores": available_stores[:50],  # First 50 for demo
            "message": f"Showing first 50 of {len(available_stores)} stores"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/store/{store_id}")
async def predict_store_forecast(store_id: int, forecast_days: int = 30):
    """
    Predict cash flow forecast for a specific store (SME)
    
    Args:
        store_id: Store ID (1-1115)
        forecast_days: Number of days to forecast (30, 60, or 90)
    
    Returns:
        Forecast with confidence intervals
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get store data
        df = train[train['Store'] == store_id].copy()
        if len(df) == 0:
            raise HTTPException(status_code=404, detail=f"Store {store_id} not found")
        
        df_store = store[store['Store'] == store_id]
        if len(df_store) == 0:
            raise HTTPException(status_code=404, detail=f"Store {store_id} metadata not found")
        
        store_type = df_store.iloc[0]['StoreType']
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[df['Open'] == 1].sort_values('Date')
        
        if len(df) < 30:
            raise HTTPException(status_code=400, detail="Insufficient historical data")
        
        # Get recent statistics
        recent_avg = df['Sales'].tail(30).mean()
        recent_std = df['Sales'].tail(30).std()
        
        # Simple prediction based on recent median
        prediction = df['Sales'].tail(30).median()
        
        # Calculate forecasts
        forecast_30d = prediction * 30
        forecast_60d = prediction * 60
        forecast_90d = prediction * 90
        
        # Calculate trend
        recent_trend = df['Sales'].tail(14).mean() - df['Sales'].tail(30).iloc[:14].mean()
        if recent_trend > 100:
            trend = "increasing"
        elif recent_trend < -100:
            trend = "decreasing"
        else:
            trend = "stable"
        
        # Confidence intervals
        confidence_lower = prediction - (recent_std * 1.96)
        confidence_upper = prediction + (recent_std * 1.96)
        
        return ForecastResponse(
            store_id=store_id,
            store_type=str(store_type),
            predicted_daily_revenue=round(prediction, 2),
            forecast_30d_revenue=round(forecast_30d, 2),
            forecast_60d_revenue=round(forecast_60d, 2),
            forecast_90d_revenue=round(forecast_90d, 2),
            confidence_lower=round(confidence_lower, 2),
            confidence_upper=round(confidence_upper, 2),
            recent_average_revenue=round(recent_avg, 2),
            trend=trend
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/top-stores")
async def get_top_stores(limit: int = 10, forecast_days: int = 90):
    """
    Get top performing stores by forecast
    
    Args:
        limit: Number of stores to return (default: 10)
        forecast_days: Days for forecast comparison (default: 90)
    
    Returns:
        List of top stores with forecasts
    """
    try:
        # Sample of stores for demo (first 100)
        sample_stores = train['Store'].unique()[:100]
        results = []
        
        for store_id in sample_stores:
            df = train[train['Store'] == store_id].copy()
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[df['Open'] == 1].sort_values('Date')
            
            if len(df) < 30:
                continue
            
            prediction = df['Sales'].tail(30).median()
            forecast = prediction * forecast_days
            
            df_store = store[store['Store'] == store_id]
            store_type = df_store.iloc[0]['StoreType'] if len(df_store) > 0 else 'Unknown'
            
            results.append({
                'store_id': int(store_id),
                'store_type': str(store_type),
                f'{forecast_days}d_forecast': round(forecast, 2),
                'recent_avg': round(df['Sales'].tail(30).mean(), 2)
            })
        
        # Sort by forecast
        results_sorted = sorted(results, key=lambda x: x[f'{forecast_days}d_forecast'], reverse=True)
        
        return {
            "top_stores": results_sorted[:limit],
            "forecast_period": f"{forecast_days} days",
            "total_analyzed": len(results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stores/{store_id}/history")
async def get_store_history(store_id: int, days: int = 30):
    """Get historical cash flow data for a store"""
    try:
        df = train[train['Store'] == store_id].copy()
        if len(df) == 0:
            raise HTTPException(status_code=404, detail=f"Store {store_id} not found")
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[df['Open'] == 1].sort_values('Date').tail(days)
        
        history = df[['Date', 'Sales', 'Customers', 'Promo', 'DayOfWeek']].to_dict('records')
        
        return {
            "store_id": store_id,
            "days": len(history),
            "history": history
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_statistics():
    """Get overall statistics about the dataset"""
    try:
        total_stores = train['Store'].nunique()
        total_records = len(train)
        avg_sales = train['Sales'].mean()
        median_sales = train['Sales'].median()
        
        return {
            "total_stores": int(total_stores),
            "total_records": int(total_records),
            "average_daily_sales": round(avg_sales, 2),
            "median_daily_sales": round(median_sales, 2),
            "date_range": {
                "start": str(pd.to_datetime(train['Date']).min()),
                "end": str(pd.to_datetime(train['Date']).max())
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

