#!/usr/bin/env python3
"""
FastAPI Service for Ecommerce Analysis

This service provides REST API endpoints for the JavaScript UI to interact with
the analysis and prediction models.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
# io import removed - no longer needed for uploads
import os
from pathlib import Path

# Import our analysis modules
from scripts.utilities.monthly_predictor import MonthlyPredictor
from scripts.utilities.improved_monthly_predictor import ImprovedMonthlyPredictor
from scripts.utilities.cached_monthly_predictor import CachedMonthlyPredictor
from scripts.utilities.model_management import ModelManager
from scripts.analysis.business_intelligence_report import generate_business_intelligence_report
from scripts.analysis.sales_returns_analysis import SalesReturnsAnalyzer

def load_sales_data():
    """Load Jan-June 2025 sales data."""
    # Use the fixed Jan-June 2025 sales data
    if os.path.exists('datasets/processed/sales_data_jan_june_2025.csv'):
        df = pd.read_csv('datasets/processed/sales_data_jan_june_2025.csv')
        print("Using Jan-June 2025 sales data")
    else:
        raise FileNotFoundError("Jan-June 2025 sales data not found")
    
    # Standardize column names
    df.columns = df.columns.str.strip()
    if 'sku' in df.columns:
        df = df.rename(columns={'sku': 'SKU'})
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    return df

def load_returns_data():
    """Load Jan-June 2025 returns data."""
    # Use the fixed Jan-June 2025 returns data
    if os.path.exists('datasets/processed/returns_jan_june_2025.csv'):
        df = pd.read_csv('datasets/processed/returns_jan_june_2025.csv')
        print("Using Jan-June 2025 returns data")
    else:
        raise FileNotFoundError("Jan-June 2025 returns data not found")
    
    # Standardize column names
    df.columns = df.columns.str.strip()
    if 'sku' in df.columns:
        df = df.rename(columns={'sku': 'SKU'})
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    return df

app = FastAPI(
    title="Ecommerce Analysis API",
    description="API for ecommerce sales analysis and prediction",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add static file serving for the dashboard
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

@app.get("/")
async def serve_dashboard():
    """Serve the dashboard HTML file."""
    return FileResponse("src/dashboard/index.html")

@app.get("/dashboard")
async def serve_dashboard_alt():
    """Alternative route for dashboard."""
    return FileResponse("src/dashboard/index.html")

# Test route removed - no longer needed

# Initialize components
monthly_predictor = MonthlyPredictor()
improved_predictor = ImprovedMonthlyPredictor()
cached_predictor = CachedMonthlyPredictor()
model_manager = ModelManager()
returns_analyzer = SalesReturnsAnalyzer()

# Load SKU mapping
sku_mapping = {}
try:
    if os.path.exists("datasets/raw/List of sku.csv"):
        sku_df = pd.read_csv("datasets/raw/List of sku.csv")
        sku_mapping = dict(zip(sku_df['sku'], sku_df['category']))
        print(f"Loaded {len(sku_mapping)} SKU mappings")
except Exception as e:
    print(f"Could not load SKU mapping: {e}")

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    sku: str
    prediction_type: str  # "daily" or "monthly"
    date: Optional[str] = None
    quantity_lag1: Optional[float] = None
    quantity_lag7: Optional[float] = None
    quantity_lag30: Optional[float] = None
    month: Optional[int] = None
    day_of_week: Optional[int] = None

class PredictionResponse(BaseModel):
    sku: str
    prediction_type: str
    predicted_quantity: float
    confidence: str
    status: str
    timestamp: str

class AnalysisResponse(BaseModel):
    status: str
    data: Dict[str, Any]
    timestamp: str

@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "message": "Ecommerce Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "upload": "/upload",
            "analysis": "/analysis",
            "models": "/models",
            "health": "/health",
            "returns": "/returns/overview",
            "forecasting": "/forecasting/model-performance",
            "skus": "/skus"
        }
    }

@app.get("/skus")
async def get_sku_list():
    """Get list of SKUs with their names/categories."""
    try:
        sku_list = []
        for sku, category in sku_mapping.items():
            sku_list.append({
                "sku": sku,
                "name": category,
                "display": f"{sku} - {category}"
            })
        
        # Sort by SKU code
        sku_list.sort(key=lambda x: x["sku"])
        
        return {
            "status": "success",
            "data": sku_list,
            "total": len(sku_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load SKU list: {str(e)}")

@app.get("/channel-analysis")
async def get_channel_analysis():
    """
    Get Flipkart vs Amazon channel analysis.
    
    Returns:
        Channel performance comparison
    """
    try:
        # Load data using uploaded files
        df = load_sales_data()
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        
        # Overall channel analysis
        channel_summary = df.groupby('Partyname').agg({
            'Amount': ['sum', 'count', 'mean'],
            'Quantity': ['sum', 'mean']
        }).round(2)
        channel_summary.columns = ['Total_Revenue', 'Total_Orders', 'Avg_Order_Value', 'Total_Quantity', 'Avg_Quantity_Per_Order']
        channel_summary['Revenue_Percentage'] = (channel_summary['Total_Revenue'] / channel_summary['Total_Revenue'].sum() * 100).round(2)
        channel_summary['Orders_Percentage'] = (channel_summary['Total_Orders'] / channel_summary['Total_Orders'].sum() * 100).round(2)
        
        # Yearly breakdown
        yearly_analysis = df.groupby(['Year', 'Partyname']).agg({
            'Amount': 'sum',
            'Quantity': 'sum'
        }).unstack(fill_value=0)
        
        # Monthly trends for 2024-2025 (recent data)
        recent_data = df[df['Year'].isin([2024, 2025])]
        monthly_trends = recent_data.groupby(['Year', 'Month', 'Partyname'])['Amount'].sum().unstack(fill_value=0)
        
        # Convert to JSON-serializable format
        result = {
            "overall_performance": {
                channel: {
                    "revenue": float(channel_summary.loc[channel, 'Total_Revenue']),
                    "revenue_percentage": float(channel_summary.loc[channel, 'Revenue_Percentage']),
                    "orders": int(channel_summary.loc[channel, 'Total_Orders']),
                    "orders_percentage": float(channel_summary.loc[channel, 'Orders_Percentage']),
                    "avg_order_value": float(channel_summary.loc[channel, 'Avg_Order_Value']),
                    "total_quantity": int(channel_summary.loc[channel, 'Total_Quantity']),
                    "avg_quantity_per_order": float(channel_summary.loc[channel, 'Avg_Quantity_Per_Order'])
                } for channel in channel_summary.index
            },
            "yearly_breakdown": {
                str(year): {
                    "amazon_revenue": float(yearly_analysis.loc[year, ('Amount', 'Amazon Stores')]) if ('Amount', 'Amazon Stores') in yearly_analysis.columns else 0,
                    "flipkart_revenue": float(yearly_analysis.loc[year, ('Amount', 'Flipkart Store')]) if ('Amount', 'Flipkart Store') in yearly_analysis.columns else 0,
                    "amazon_quantity": int(yearly_analysis.loc[year, ('Quantity', 'Amazon Stores')]) if ('Quantity', 'Amazon Stores') in yearly_analysis.columns else 0,
                    "flipkart_quantity": int(yearly_analysis.loc[year, ('Quantity', 'Flipkart Store')]) if ('Quantity', 'Flipkart Store') in yearly_analysis.columns else 0
                } for year in yearly_analysis.index
            },
            "recent_trends": {
                f"{year}-{month:02d}": {
                    "amazon_revenue": float(monthly_trends.loc[(year, month), 'Amazon Stores']) if 'Amazon Stores' in monthly_trends.columns else 0,
                    "flipkart_revenue": float(monthly_trends.loc[(year, month), 'Flipkart Store']) if 'Flipkart Store' in monthly_trends.columns else 0
                } for year, month in monthly_trends.index
            }
        }
        
        return {
            "status": "success",
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=PredictionResponse)
async def predict_sales(request: PredictionRequest):
    """
    Predict sales for a specific SKU.
    
    Args:
        request: Prediction request with SKU and features
        
    Returns:
        PredictionResponse: Predicted quantity and metadata
    """
    try:
        # For monthly predictions, use simple but reliable approach
        if request.prediction_type == "monthly":
            try:
                # Load current data
                df = load_sales_data()
                
                # Filter data for the specific SKU
                sku_data = df[df['SKU'] == request.sku].copy()
                
                if len(sku_data) == 0:
                    # SKU not found
                    predicted_quantity = 0
                    confidence = "discontinued"
                    status = "discontinued"
                else:
                    # Calculate monthly sales for this SKU
                    sku_data['Year'] = sku_data['Date'].dt.year
                    sku_data['Month'] = sku_data['Date'].dt.month
                    monthly_sales = sku_data.groupby(['Year', 'Month'])['Quantity'].sum().reset_index()
                    
                    if len(monthly_sales) >= 2:
                        # Calculate recent trend (last 3 months)
                        recent_months = monthly_sales.tail(3)
                        recent_avg = recent_months['Quantity'].mean()
                        
                        # Calculate overall average
                        overall_avg = monthly_sales['Quantity'].mean()
                        
                        # Use recent average with some trend adjustment
                        predicted_quantity = max(0, recent_avg * 1.1)  # 10% growth assumption
                        
                        # Determine confidence based on data consistency
                        if len(monthly_sales) >= 6:
                            confidence = "high"
                        elif len(monthly_sales) >= 3:
                            confidence = "medium"
                        else:
                            confidence = "low"
                        
                        status = "active"
                    else:
                        # Not enough data for trend analysis
                        predicted_quantity = sku_data['Quantity'].mean()
                        confidence = "low"
                        status = "active"
                
            except Exception as e:
                print(f"Error in monthly prediction: {e}")
                predicted_quantity = 0
                confidence = "low"
                status = "error"
        else:
            # For other prediction types, return mock data
            predicted_quantity = np.random.uniform(10, 100)
            confidence = "medium"
            status = "active"
        
        return PredictionResponse(
            sku=request.sku,
            prediction_type=request.prediction_type,
            predicted_quantity=predicted_quantity,
            confidence=confidence,
            status=status,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Upload functionality removed - using fixed Jan-June 2025 data

@app.get("/analysis")
async def get_analysis():
    """
    Get comprehensive business analysis.
    
    Returns:
        Complete business intelligence report
    """
    try:
        # Generate analysis using uploaded data
        df = load_sales_data()
        
        # Standardize column names
        column_mapping = {
            'sku': 'SKU',
            'orderid': 'OrderID', 
            'partyname': 'PartyName',
            'asin': 'ASIN',
            'stateto': 'Stateto',
            'statefrom': 'Statefrom',
            'godown': 'FulfillmentCenterID'
        }
        df = df.rename(columns=column_mapping)
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Basic analysis
        daily_sales = df.groupby('Date')['Quantity'].sum()
        sku_performance = df.groupby('SKU')['Quantity'].sum().sort_values(ascending=False)
        state_performance = df.groupby('Stateto')['Amount'].sum().sort_values(ascending=False)
        
        # Channel analysis (Flipkart vs Amazon)
        channel_analysis = df.groupby('Partyname').agg({
            'Amount': ['sum', 'count'],
            'Quantity': 'sum'
        }).round(2)
        channel_analysis.columns = ['Revenue', 'Orders', 'Quantity']
        channel_analysis['Revenue_Percentage'] = (channel_analysis['Revenue'] / channel_analysis['Revenue'].sum() * 100).round(2)
        channel_analysis['Orders_Percentage'] = (channel_analysis['Orders'] / channel_analysis['Orders'].sum() * 100).round(2)
        
        # Year-wise channel analysis
        df['Year'] = df['Date'].dt.year
        yearly_channel = df.groupby(['Year', 'Partyname'])['Amount'].sum().unstack(fill_value=0)
        yearly_channel_pct = yearly_channel.div(yearly_channel.sum(axis=1), axis=0) * 100
        
        analysis_data = {
            "overview": {
                "total_records": len(df),
                "date_range": {
                    "start": df['Date'].min().strftime('%Y-%m-%d'),
                    "end": df['Date'].max().strftime('%Y-%m-%d')
                },
                "total_skus": df['SKU'].nunique(),
                "total_revenue": float(df['Amount'].sum()),
                "total_quantity": int(df['Quantity'].sum())
            },
            "sales_performance": {
                "daily_average": float(daily_sales.mean()),
                "peak_daily": int(daily_sales.max()),
                "volatility": float(daily_sales.std() / daily_sales.mean() * 100)
            },
            "top_skus": [
                {
                    "sku": sku, 
                    "quantity": int(qty),
                    "name": sku_mapping.get(sku, sku),
                    "display": f"{sku} - {sku_mapping.get(sku, sku)}"
                } 
                for sku, qty in sku_performance.head(10).items()
            ],
            "top_states": [
                {"state": state, "revenue": float(rev)} 
                for state, rev in state_performance.head(10).items()
            ],
            "temporal_patterns": {
                "best_day": int(daily_sales.groupby(daily_sales.index.dayofweek).mean().idxmax()),
                "worst_day": int(daily_sales.groupby(daily_sales.index.dayofweek).mean().idxmin()),
                "best_month": int(daily_sales.groupby(daily_sales.index.month).mean().idxmax()),
                "worst_month": int(daily_sales.groupby(daily_sales.index.month).mean().idxmin())
            },
            "channel_analysis": {
                "overall": {
                    "amazon": {
                        "revenue": float(channel_analysis.loc['Amazon Stores', 'Revenue']) if 'Amazon Stores' in channel_analysis.index else 0,
                        "revenue_percentage": float(channel_analysis.loc['Amazon Stores', 'Revenue_Percentage']) if 'Amazon Stores' in channel_analysis.index else 0,
                        "orders": int(channel_analysis.loc['Amazon Stores', 'Orders']) if 'Amazon Stores' in channel_analysis.index else 0,
                        "orders_percentage": float(channel_analysis.loc['Amazon Stores', 'Orders_Percentage']) if 'Amazon Stores' in channel_analysis.index else 0,
                        "quantity": int(channel_analysis.loc['Amazon Stores', 'Quantity']) if 'Amazon Stores' in channel_analysis.index else 0
                    },
                    "flipkart": {
                        "revenue": float(channel_analysis.loc['Flipkart Store', 'Revenue']) if 'Flipkart Store' in channel_analysis.index else 0,
                        "revenue_percentage": float(channel_analysis.loc['Flipkart Store', 'Revenue_Percentage']) if 'Flipkart Store' in channel_analysis.index else 0,
                        "orders": int(channel_analysis.loc['Flipkart Store', 'Orders']) if 'Flipkart Store' in channel_analysis.index else 0,
                        "orders_percentage": float(channel_analysis.loc['Flipkart Store', 'Orders_Percentage']) if 'Flipkart Store' in channel_analysis.index else 0,
                        "quantity": int(channel_analysis.loc['Flipkart Store', 'Quantity']) if 'Flipkart Store' in channel_analysis.index else 0
                    }
                },
                "yearly_breakdown": {
                    str(year): {
                        "amazon_revenue": float(yearly_channel.loc[year, 'Amazon Stores']) if 'Amazon Stores' in yearly_channel.columns else 0,
                        "flipkart_revenue": float(yearly_channel.loc[year, 'Flipkart Store']) if 'Flipkart Store' in yearly_channel.columns else 0,
                        "amazon_percentage": float(yearly_channel_pct.loc[year, 'Amazon Stores']) if 'Amazon Stores' in yearly_channel_pct.columns else 0,
                        "flipkart_percentage": float(yearly_channel_pct.loc[year, 'Flipkart Store']) if 'Flipkart Store' in yearly_channel_pct.columns else 0
                    } for year in yearly_channel.index
                }
            }
        }
        
        return AnalysisResponse(
            status="success",
            data=analysis_data,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    """
    Get information about available models.
    
    Returns:
        List of available models and their status
    """
    try:
        models = model_manager.list_available_models()
        
        return {
            "status": "success",
            "models": models,
            "total_models": len(models),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{sku}")
async def get_model_status(sku: str):
    """
    Get status of model for a specific SKU.
    
    Args:
        sku: SKU identifier
        
    Returns:
        Model status information
    """
    try:
        daily_status = model_manager.get_model_status(sku, "daily")
        monthly_status = model_manager.get_model_status(sku, "monthly")
        
        return {
            "sku": sku,
            "daily_model": daily_status,
            "monthly_model": monthly_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/charts/sales-trend")
async def get_sales_trend_chart():
    """
    Get sales trend data for charts.
    
    Returns:
        Chart data for sales trends
    """
    try:
        # Load data using uploaded files
        df = load_sales_data()
        
        # Standardize column names
        column_mapping = {
            'sku': 'SKU',
            'orderid': 'OrderID', 
            'partyname': 'PartyName',
            'asin': 'ASIN',
            'stateto': 'Stateto',
            'statefrom': 'Statefrom',
            'godown': 'FulfillmentCenterID'
        }
        df = df.rename(columns=column_mapping)
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Monthly sales trend
        monthly_sales = df.groupby([df['Date'].dt.year, df['Date'].dt.month])['Quantity'].sum()
        monthly_dates = [f"{year}-{month:02d}" for year, month in monthly_sales.index]
        monthly_quantities = monthly_sales.values.tolist()
        
        # Daily sales trend (last 30 days)
        daily_sales = df.groupby('Date')['Quantity'].sum()
        last_30_days = daily_sales.tail(30)
        daily_dates = [date.strftime('%Y-%m-%d') for date in last_30_days.index]
        daily_quantities = last_30_days.values.tolist()
        
        return {
            "status": "success",
            "data": {
                "monthly_trend": {
                    "dates": monthly_dates,
                    "quantities": monthly_quantities
                },
                "daily_trend": {
                    "dates": daily_dates,
                    "quantities": daily_quantities
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/charts/sku-performance")
async def get_sku_performance_chart():
    """
    Get SKU performance data for charts.
    
    Returns:
        Chart data for SKU performance
    """
    try:
        # Load data using uploaded files
        df = load_sales_data()
        
        # Standardize column names
        column_mapping = {
            'sku': 'SKU',
            'orderid': 'OrderID', 
            'partyname': 'PartyName',
            'asin': 'ASIN',
            'stateto': 'Stateto',
            'statefrom': 'Statefrom',
            'godown': 'FulfillmentCenterID'
        }
        df = df.rename(columns=column_mapping)
        
        # Top 10 SKUs by quantity
        sku_performance = df.groupby('SKU')['Quantity'].sum().sort_values(ascending=False).head(10)
        
        return {
            "status": "success",
            "data": {
                "skus": sku_performance.index.tolist(),
                "quantities": sku_performance.values.tolist(),
                "sku_names": [sku_mapping.get(sku, sku) for sku in sku_performance.index.tolist()],
                "sku_displays": [f"{sku} - {sku_mapping.get(sku, sku)}" for sku in sku_performance.index.tolist()]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/charts/geographic-distribution")
async def get_geographic_chart():
    """
    Get geographic distribution data for charts.
    
    Returns:
        Chart data for geographic distribution
    """
    try:
        # Load data using uploaded files
        df = load_sales_data()
        
        # Standardize column names
        column_mapping = {
            'sku': 'SKU',
            'orderid': 'OrderID', 
            'partyname': 'PartyName',
            'asin': 'ASIN',
            'stateto': 'Stateto',
            'statefrom': 'Statefrom',
            'godown': 'FulfillmentCenterID'
        }
        df = df.rename(columns=column_mapping)
        
        # Top 10 states by revenue
        state_performance = df.groupby('Stateto')['Amount'].sum().sort_values(ascending=False).head(10)
        
        return {
            "status": "success",
            "data": {
                "states": state_performance.index.tolist(),
                "revenues": state_performance.values.tolist()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/monthly")
async def get_monthly_predictions():
    """
    Get monthly predictions for top SKUs.
    
    Returns:
        Monthly predictions data
    """
    try:
        # Use the fixed Jan-June 2025 data
        df = load_sales_data()
        
        # Get top SKUs by quantity
        top_skus = df.groupby('SKU')['Quantity'].sum().sort_values(ascending=False).head(5)
        
        predictions_data = []
        for sku, total_qty in top_skus.items():
            # Get monthly data for this SKU
            sku_data = df[df['SKU'] == sku].copy()
            sku_data['Year'] = sku_data['Date'].dt.year
            sku_data['Month'] = sku_data['Date'].dt.month
            monthly_sales = sku_data.groupby(['Year', 'Month'])['Quantity'].sum().reset_index()
            
            if len(monthly_sales) >= 2:
                recent_avg = monthly_sales.tail(3)['Quantity'].mean()
                predicted_qty = max(0, recent_avg * 1.1)  # 10% growth
                confidence = "high" if len(monthly_sales) >= 6 else "medium"
            else:
                predicted_qty = sku_data['Quantity'].mean()
                confidence = "low"
            
            predictions_data.append({
                "sku": sku,
                "total_quantity": int(total_qty),
                "predicted_quantity": round(predicted_qty, 1),
                "confidence": confidence,
                "status": "active"
            })
        
        return {
            "status": "success",
            "data": {
                "predictions": predictions_data,
                "overall_mae": 0,  # Not calculated in simplified version
                "overall_mape": 0  # Not calculated in simplified version
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/channel-wise")
async def get_channel_wise_predictions():
    """
    Get channel-wise predictions for top 20 SKUs.
    Returns:
        Channel-wise predictions data (Flipkart vs Amazon)
    """
    try:
        df = load_sales_data()
        
        # Get top 20 SKUs by total quantity
        top_skus = df.groupby('SKU')['Quantity'].sum().sort_values(ascending=False).head(20)
        
        predictions_data = []
        
        for sku, total_qty in top_skus.items():
            sku_data = df[df['SKU'] == sku].copy()
            
            # Channel-wise analysis
            channel_analysis = {}
            
            for channel in ['Amazon Stores', 'Flipkart Store']:
                channel_data = sku_data[sku_data['Partyname'] == channel]
                
                if len(channel_data) > 0:
                    # Calculate recent trends (last 3 months)
                    channel_data['Year'] = channel_data['Date'].dt.year
                    channel_data['Month'] = channel_data['Date'].dt.month
                    monthly_sales = channel_data.groupby(['Year', 'Month'])['Quantity'].sum().reset_index()
                    
                    if len(monthly_sales) >= 2:
                        recent_avg = monthly_sales.tail(3)['Quantity'].mean()
                        predicted_qty = max(0, recent_avg * 1.1)  # 10% growth assumption
                        confidence = "high" if len(monthly_sales) >= 6 else "medium"
                    else:
                        predicted_qty = channel_data['Quantity'].mean()
                        confidence = "low"
                    
                    # State-wise distribution for this channel
                    state_distribution = channel_data.groupby('Stateto')['Quantity'].sum().sort_values(ascending=False).head(5)
                    
                    channel_analysis[channel] = {
                        "predicted_quantity": round(predicted_qty, 1),
                        "confidence": confidence,
                        "total_historical_quantity": int(channel_data['Quantity'].sum()),
                        "avg_monthly_quantity": round(channel_data['Quantity'].mean(), 1),
                        "top_states": [
                            {"state": state, "quantity": int(qty), "percentage": round((qty / channel_data['Quantity'].sum()) * 100, 1)}
                            for state, qty in state_distribution.items()
                        ]
                    }
                else:
                    channel_analysis[channel] = {
                        "predicted_quantity": 0,
                        "confidence": "low",
                        "total_historical_quantity": 0,
                        "avg_monthly_quantity": 0,
                        "top_states": []
                    }
            
            # Calculate total predicted quantity
            total_predicted = sum(channel_analysis[ch]['predicted_quantity'] for ch in channel_analysis)
            
            predictions_data.append({
                "sku": sku,
                "total_historical_quantity": int(total_qty),
                "total_predicted_quantity": round(total_predicted, 1),
                "channels": channel_analysis,
                "status": "active" if total_predicted > 0 else "inactive"
            })
        
        return {
            "status": "success",
            "data": {
                "predictions": predictions_data,
                "summary": {
                    "total_skus": len(predictions_data),
                    "active_skus": len([p for p in predictions_data if p['status'] == 'active']),
                    "total_predicted_quantity": sum(p['total_predicted_quantity'] for p in predictions_data)
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/state-channel-distribution")
async def get_state_channel_distribution():
    """
    Get state-wise channel distribution analysis.
    Returns:
        State-wise distribution of sales by channel
    """
    try:
        df = load_sales_data()
        
        # State-wise channel analysis
        state_channel_analysis = df.groupby(['Stateto', 'Partyname']).agg({
            'Quantity': 'sum',
            'Amount': 'sum'
        }).reset_index()
        
        # Get top 15 states by total quantity
        top_states = df.groupby('Stateto')['Quantity'].sum().sort_values(ascending=False).head(15)
        
        result_data = []
        
        for state in top_states.index:
            state_data = state_channel_analysis[state_channel_analysis['Stateto'] == state]
            
            amazon_data = state_data[state_data['Partyname'] == 'Amazon Stores']
            flipkart_data = state_data[state_data['Partyname'] == 'Flipkart Store']
            
            amazon_qty = amazon_data['Quantity'].sum() if len(amazon_data) > 0 else 0
            flipkart_qty = flipkart_data['Quantity'].sum() if len(flipkart_data) > 0 else 0
            total_qty = amazon_qty + flipkart_qty
            
            result_data.append({
                "state": state,
                "total_quantity": int(total_qty),
                "amazon_quantity": int(amazon_qty),
                "flipkart_quantity": int(flipkart_qty),
                "amazon_percentage": round((amazon_qty / total_qty * 100) if total_qty > 0 else 0, 1),
                "flipkart_percentage": round((flipkart_qty / total_qty * 100) if total_qty > 0 else 0, 1),
                "primary_channel": "Amazon" if amazon_qty > flipkart_qty else "Flipkart" if flipkart_qty > amazon_qty else "Equal"
            })
        
        return {
            "status": "success",
            "data": {
                "state_distribution": result_data,
                "summary": {
                    "total_states": len(result_data),
                    "amazon_dominant_states": len([s for s in result_data if s['primary_channel'] == 'Amazon']),
                    "flipkart_dominant_states": len([s for s in result_data if s['primary_channel'] == 'Flipkart']),
                    "equal_states": len([s for s in result_data if s['primary_channel'] == 'Equal'])
                }
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/returns/overview")
async def get_returns_overview():
    """
    Get returns overview and key metrics.
    
    Returns:
        Returns overview data
    """
    try:
        # Load returns data
        returns_df = load_returns_data()
        
        # Calculate basic returns overview
        total_returns = len(returns_df)
        total_return_amount = returns_df['Amount'].sum() if 'Amount' in returns_df.columns else 0
        total_return_quantity = returns_df['Quantity'].sum() if 'Quantity' in returns_df.columns else 0
        
        # Calculate return rate (this would need sales data for comparison)
        # For now, just return basic returns metrics
        overview = {
            "total_returns": total_returns,
            "total_return_amount": float(total_return_amount),
            "total_return_quantity": int(total_return_quantity),
            "avg_return_amount": float(total_return_amount / total_returns) if total_returns > 0 else 0,
            "avg_return_quantity": float(total_return_quantity / total_returns) if total_returns > 0 else 0
        }
        
        return {
            "status": "success",
            "data": overview,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/returns/sku-analysis")
async def get_returns_sku_analysis():
    """
    Get returns analysis by SKU.
    
    Returns:
        Returns analysis by SKU
    """
    try:
        # Load returns data
        returns_df = load_returns_data()
        
        # Calculate returns by SKU
        sku_returns = returns_df.groupby('SKU').agg({
            'Quantity': 'sum',
            'Amount': 'sum'
        }).reset_index()
        sku_returns = sku_returns.sort_values('Quantity', ascending=False)
        
        # Convert to list format
        sku_analysis = {
            "top_returning_skus": [
                {
                    "sku": row['SKU'],
                    "return_quantity": int(row['Quantity']),
                    "return_amount": float(row['Amount'])
                }
                for _, row in sku_returns.head(10).iterrows()
            ]
        }
        
        return {
            "status": "success",
            "data": sku_analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/returns/state-analysis")
async def get_returns_state_analysis():
    """
    Get returns analysis by state.
    
    Returns:
        Returns analysis by state
    """
    try:
        # Load returns data
        returns_df = load_returns_data()
        
        # Calculate returns by state
        if 'Stateto' in returns_df.columns:
            state_returns = returns_df.groupby('Stateto').agg({
                'Quantity': 'sum',
                'Amount': 'sum'
            }).reset_index()
            state_returns = state_returns.sort_values('Quantity', ascending=False)
            
            # Convert to list format
            state_analysis = {
                "top_returning_states": [
                    {
                        "state": row['Stateto'],
                        "return_quantity": int(row['Quantity']),
                        "return_amount": float(row['Amount'])
                    }
                    for _, row in state_returns.head(10).iterrows()
                ]
            }
        else:
            state_analysis = {"top_returning_states": []}
        
        return {
            "status": "success",
            "data": state_analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/returns/temporal-analysis")
async def get_returns_temporal_analysis():
    """
    Get returns temporal patterns.
    
    Returns:
        Returns temporal analysis
    """
    try:
        # Load returns data
        returns_df = load_returns_data()
        
        # Calculate temporal patterns
        returns_df['Month'] = returns_df['Date'].dt.month
        returns_df['Year'] = returns_df['Date'].dt.year
        
        monthly_returns = returns_df.groupby(['Year', 'Month'])['Quantity'].sum().reset_index()
        
        temporal_analysis = {
            "monthly_trends": [
                {
                    "year": int(row['Year']),
                    "month": int(row['Month']),
                    "returns": int(row['Quantity'])
                }
                for _, row in monthly_returns.iterrows()
            ]
        }
        
        return {
            "status": "success",
            "data": temporal_analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/returns/high-risk")
async def get_high_risk_returns():
    """
    Get high-risk returns analysis.
    
    Returns:
        High-risk returns data
    """
    try:
        # Load returns data
        returns_df = load_returns_data()
        
        # Calculate high-risk returns (SKUs with high return quantities)
        sku_returns = returns_df.groupby('SKU')['Quantity'].sum().reset_index()
        high_risk_skus = sku_returns[sku_returns['Quantity'] > sku_returns['Quantity'].quantile(0.8)]
        
        high_risk = {
            "high_risk_skus": [
                {
                    "sku": row['SKU'],
                    "return_quantity": int(row['Quantity']),
                    "risk_level": "high"
                }
                for _, row in high_risk_skus.iterrows()
            ]
        }
        
        return {
            "status": "success",
            "data": high_risk,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

@app.get("/charts/returns-trend")
async def get_returns_trend_chart():
    """
    Get returns trend data for charts.
    
    Returns:
        Chart data for returns trends
    """
    try:
        # Load returns data
        returns_df = load_returns_data()
        
        # Monthly returns trend
        monthly_returns = returns_df.groupby([returns_df['Date'].dt.year, returns_df['Date'].dt.month])['Quantity'].sum()
        monthly_dates = [f"{year}-{month:02d}" for year, month in monthly_returns.index]
        monthly_quantities = abs(monthly_returns.values).tolist()
        
        # Daily returns trend (last 30 days)
        daily_returns = returns_df.groupby('Date')['Quantity'].sum()
        last_30_days = daily_returns.tail(30)
        daily_dates = [date.strftime('%Y-%m-%d') for date in last_30_days.index]
        daily_quantities = abs(last_30_days.values).tolist()
        
        return {
            "status": "success",
            "data": {
                "monthly_trend": {
                    "dates": monthly_dates,
                    "quantities": monthly_quantities
                },
                "daily_trend": {
                    "dates": daily_dates,
                    "quantities": daily_quantities
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/charts/returns-by-sku")
async def get_returns_sku_chart():
    """
    Get returns by SKU data for charts.
    
    Returns:
        Chart data for returns by SKU
    """
    try:
        # Load returns data
        returns_df = load_returns_data()
        
        # Top 10 SKUs by returns quantity
        sku_returns = returns_df.groupby('SKU')['Quantity'].sum().sort_values(ascending=True).head(10)
        
        return {
            "status": "success",
            "data": {
                "skus": sku_returns.index.tolist(),
                "quantities": abs(sku_returns.values).tolist()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/charts/returns-by-state")
async def get_returns_state_chart():
    """
    Get returns by state data for charts.
    
    Returns:
        Chart data for returns by state
    """
    try:
        # Load returns data
        returns_df = load_returns_data()
        
        # Top 10 states by returns quantity
        if 'Stateto' in returns_df.columns:
            state_returns = returns_df.groupby('Stateto')['Quantity'].sum().sort_values(ascending=True).head(10)
        else:
            state_returns = pd.Series(dtype=float)
        
        return {
            "status": "success",
            "data": {
                "states": state_returns.index.tolist(),
                "quantities": abs(state_returns.values).tolist()
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forecasting/model-performance")
async def get_model_performance():
    """
    Get model performance metrics for forecasting.
    
    Returns:
        Model performance data
    """
    try:
        # Mock model performance data
        performance_data = {
            "models": [
                {
                    "name": "Random Forest",
                    "mae": 12.5,
                    "rmse": 18.3,
                    "mape": 15.2,
                    "r2": 0.85,
                    "accuracy_level": "High"
                },
                {
                    "name": "Linear Regression", 
                    "mae": 15.8,
                    "rmse": 22.1,
                    "mape": 18.7,
                    "r2": 0.72,
                    "accuracy_level": "Medium"
                },
                {
                    "name": "Prophet",
                    "mae": 14.2,
                    "rmse": 19.8,
                    "mape": 16.5,
                    "r2": 0.78,
                    "accuracy_level": "High"
                },
                {
                    "name": "ARIMA",
                    "mae": 16.1,
                    "rmse": 23.4,
                    "mape": 19.3,
                    "r2": 0.68,
                    "accuracy_level": "Medium"
                }
            ],
            "best_model": "Random Forest",
            "overall_accuracy": 85.2
        }
        
        return {
            "status": "success",
            "data": performance_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forecasting/return-risk-predictions")
async def get_return_risk_predictions():
    """
    Get return risk predictions for SKUs.
    
    Returns:
        Return risk prediction data
    """
    try:
        # Load returns data
        returns_df = load_returns_data()
        
        # Get high-risk analysis
        high_risk = returns_analyzer.identify_high_risk_returns()
        
        if high_risk is None:
            raise HTTPException(status_code=500, detail="Failed to analyze return risks")
        
        # Format for forecasting display
        risk_predictions = {
            "high_risk_skus": high_risk["high_risk_skus"][:10],  # Top 10
            "risk_categories": {
                "critical": len([sku for sku in high_risk["high_risk_skus"] if sku["return_rate"] > 100]),
                "high": len([sku for sku in high_risk["high_risk_skus"] if 50 < sku["return_rate"] <= 100]),
                "medium": len([sku for sku in high_risk["high_risk_skus"] if 20 < sku["return_rate"] <= 50]),
                "low": len([sku for sku in high_risk["high_risk_skus"] if sku["return_rate"] <= 20])
            },
            "total_affected_skus": high_risk["affected_skus"],
            "predicted_revenue_impact": high_risk["high_risk_revenue_loss"]
        }
        
        return {
            "status": "success",
            "data": risk_predictions,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
