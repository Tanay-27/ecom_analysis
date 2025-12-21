#!/usr/bin/env python3
"""
FastAPI Backend for E-commerce Analytics Dashboard
Modern FastAPI service serving the analytics dashboard and API endpoints
"""

import io
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

warnings.filterwarnings('ignore')

app = FastAPI(
    title="E-commerce Analytics API",
    description="Modern analytics dashboard for e-commerce sales analysis and predictions",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
CORE_DIR = Path("core")
RESULTS_DIR = CORE_DIR / "analysis_results"

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class DataProvider:
    def __init__(self):
        self.sales_data = None
        self.historical_data = None
        self.predictions = None
        self.ordering_schedule = None
        self.reorder_analysis = None
        self.sku_mapping = None
        self.load_all_data()
    
    def load_all_data(self):
        """Load all analysis results"""
        try:
            # Load recent sales data
            sales_file = PROCESSED_DIR / "sales_data_jan_june_2025.csv"
            if sales_file.exists():
                self.sales_data = pd.read_csv(sales_file)
                self.sales_data['Date'] = pd.to_datetime(self.sales_data['Date'])
            
            # Load historical data
            historical_file = PROCESSED_DIR / "historical_data_2018_nov2024.csv"
            if historical_file.exists():
                self.historical_data = pd.read_csv(historical_file)
                self.historical_data['Date'] = pd.to_datetime(self.historical_data['Date'])
            
            # Load predictions
            pred_file = RESULTS_DIR / "next_month_predictions.csv"
            if pred_file.exists():
                self.predictions = pd.read_csv(pred_file)
            
            # Load ordering schedule
            order_file = RESULTS_DIR / "ordering_schedule.csv"
            if order_file.exists():
                self.ordering_schedule = pd.read_csv(order_file)
            
            # Load reorder analysis
            reorder_file = RESULTS_DIR / "reorder_analysis.csv"
            if reorder_file.exists():
                self.reorder_analysis = pd.read_csv(reorder_file)
            
            # Load SKU mapping for product names
            try:
                sku_file = DATA_DIR / "raw" / "sku_list.csv"
                sku_df = pd.read_csv(sku_file)
                self.sku_mapping = dict(zip(sku_df['sku'], sku_df['category']))
                print(f"SKU mapping loaded: {len(self.sku_mapping)} products")
            except Exception as e:
                print(f"Could not load SKU mapping: {e}")
                self.sku_mapping = {}
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

# Initialize data provider
data_provider = DataProvider()

@app.get("/")
async def serve_dashboard():
    """Serve the main dashboard page"""
    return FileResponse("templates/dashboard.html")

@app.get("/api/data-freshness")
async def get_data_freshness():
    """Get data freshness information"""
    try:
        freshness_info = {
            'last_updated': None,
            'data_sources': {},
            'update_frequency': 'Manual - Run analysis script for updates',
            'recommended_update_frequency': 'Weekly for optimal accuracy'
        }
        
        # Check analysis results files
        if RESULTS_DIR.exists():
            files_info = {
                'predictions': RESULTS_DIR / "next_month_predictions.csv",
                'ordering_schedule': RESULTS_DIR / "ordering_schedule.csv", 
                'reorder_analysis': RESULTS_DIR / "reorder_analysis.csv"
            }
            
            latest_timestamp = None
            for name, file_path in files_info.items():
                if file_path.exists():
                    mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    freshness_info['data_sources'][name] = {
                        'last_modified': mod_time.isoformat(),
                        'file_exists': True
                    }
                    if latest_timestamp is None or mod_time > latest_timestamp:
                        latest_timestamp = mod_time
                else:
                    freshness_info['data_sources'][name] = {
                        'last_modified': None,
                        'file_exists': False
                    }
            
            # Check sales data
            sales_file = PROCESSED_DIR / "sales_data_jan_june_2025.csv"
            if sales_file.exists():
                sales_mod_time = datetime.fromtimestamp(sales_file.stat().st_mtime)
                freshness_info['data_sources']['sales_data'] = {
                    'last_modified': sales_mod_time.isoformat(),
                    'file_exists': True,
                    'period_covered': 'January - June 2025'
                }
            
            freshness_info['last_updated'] = latest_timestamp.isoformat() if latest_timestamp else None
            
        return freshness_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/business-overview")
async def business_overview():
    """API endpoint for business overview metrics"""
    if data_provider.sales_data is None:
        raise HTTPException(status_code=404, detail="No sales data available")

    sales_data = data_provider.sales_data

    # Calculate key metrics
    total_revenue = float(sales_data['Amount'].sum())
    total_orders = int(len(sales_data))
    unique_skus = int(sales_data['SKU'].nunique())
    avg_order_value = float(total_revenue / total_orders)

    # Monthly performance for growth calculation
    monthly_data = sales_data.groupby(sales_data['Date'].dt.to_period('M')).agg({
        'Amount': 'sum',
        'Quantity': 'sum'
    })

    growth_rate = 0
    if len(monthly_data) > 1:
        current_month = float(monthly_data['Amount'].iloc[-1])
        previous_month = float(monthly_data['Amount'].iloc[-2])
        growth_rate = ((current_month - previous_month) / previous_month * 100)

    # Top performing SKUs
    top_skus = sales_data.groupby('SKU')['Amount'].sum().sort_values(ascending=False).head(5)
    top_skus_data = []
    for sku, revenue in top_skus.items():
        top_skus_data.append({
            'sku': sku,
            'product_name': data_provider.sku_mapping.get(sku, 'Unknown Product'),
            'revenue': float(revenue)
        })

    # Monthly trend data for chart
    monthly_trend = []
    for period, amount in monthly_data['Amount'].items():
        monthly_trend.append({
            'month': str(period),
            'revenue': float(amount),
            'quantity': int(monthly_data.loc[period, 'Quantity'])
        })

    return {
        'total_revenue': total_revenue,
        'total_orders': total_orders,
        'unique_skus': unique_skus,
        'avg_order_value': avg_order_value,
        'growth_rate': growth_rate,
        'top_skus': top_skus_data,
        'monthly_trend': monthly_trend,
        'date_range': {
            'start': sales_data['Date'].min().strftime('%Y-%m-%d'),
            'end': sales_data['Date'].max().strftime('%Y-%m-%d')
        }
    }

@app.get("/api/ordering-recommendations")
async def ordering_recommendations():
    """API endpoint for ordering recommendations"""
    if data_provider.ordering_schedule is None:
        raise HTTPException(status_code=404, detail="No ordering schedule available")
    
    ordering = data_provider.ordering_schedule.copy()
    
    # Group by urgency
    critical = ordering[ordering['urgency'] == 'CRITICAL']
    high = ordering[ordering['urgency'] == 'HIGH']
    medium = ordering[ordering['urgency'] == 'MEDIUM']
    
    return {
        'critical_orders': critical.to_dict('records'),
        'high_priority_orders': high.to_dict('records'),
        'medium_priority_orders': medium.to_dict('records'),
        'total_critical': len(critical),
        'total_high': len(high),
        'total_medium': len(medium),
        'total_estimated_cost': float(ordering['estimated_cost'].sum())
    }

@app.get("/api/sales-predictions")
async def sales_predictions():
    """API endpoint for sales predictions"""
    if data_provider.predictions is None:
        raise HTTPException(status_code=404, detail="No predictions available")
    
    predictions = data_provider.predictions.copy()
    
    # Handle both 'SKU' and 'sku' column names
    sku_col = 'SKU' if 'SKU' in predictions.columns else 'sku'
    
    # Add product names if not already present
    if 'product_name' not in predictions.columns:
        predictions['product_name'] = predictions[sku_col].map(
            lambda x: data_provider.sku_mapping.get(x, 'Unknown Product')
        )
    
    predictions_data = []
    for _, row in predictions.iterrows():
        predictions_data.append({
            'sku': row[sku_col],
            'product_name': row.get('product_name', 'Unknown Product'),
            'monthly_quantity': float(row['predicted_monthly_quantity']),
            'predicted_monthly_quantity': float(row['predicted_monthly_quantity']),
            'daily_average': float(row.get('predicted_daily_average', row['predicted_monthly_quantity'] / 30)),
            'growth_rate': float(row.get('growth_rate', 0)),
            'confidence': row['confidence']
        })
    
    # Calculate confidence distribution
    confidence_counts = predictions['confidence'].value_counts().to_dict()
    
    return {
        'predictions': predictions_data,
        'total_predicted': int(predictions['predicted_monthly_quantity'].sum()),
        'confidence_distribution': confidence_counts
    }

@app.get("/api/comprehensive-ordering")
async def comprehensive_ordering():
    """API endpoint for comprehensive ordering analysis"""
    if (data_provider.predictions is None or 
        data_provider.sales_data is None or 
        data_provider.ordering_schedule is None):
        raise HTTPException(status_code=404, detail="Missing required data")
    
    sales_data = data_provider.sales_data.copy()
    predictions = data_provider.predictions.copy()
    ordering = data_provider.ordering_schedule.copy()
    
    # Calculate current quantities by SKU
    current_quantities = sales_data.groupby('SKU').agg({
        'Quantity': 'sum',
        'Amount': 'sum'
    }).reset_index()
    
    # Merge all data
    comprehensive_data = []
    
    # Handle both 'SKU' and 'sku' column names in predictions
    pred_sku_col = 'SKU' if 'SKU' in predictions.columns else 'sku'
    
    for _, pred_row in predictions.iterrows():
        sku = pred_row[pred_sku_col]
        
        # Get current data
        current_row = current_quantities[current_quantities['SKU'] == sku]
        current_qty = int(current_row['Quantity'].iloc[0]) if len(current_row) > 0 else 0
        
        # Get ordering data
        order_row = ordering[ordering['sku'] == sku]
        
        if len(order_row) > 0:
            order_data = order_row.iloc[0]
            urgency = order_data['urgency']
            recommended_qty = int(order_data['recommended_qty'])
            days_remaining = float(order_data['days_remaining'])
            estimated_cost = float(order_data['estimated_cost'])
            lead_time = int(order_data['lead_time'])
            current_stock = int(order_data['current_stock'])
        else:
            urgency = 'LOW'
            recommended_qty = 0
            days_remaining = 999
            estimated_cost = 0
            lead_time = 0
            current_stock = current_qty
        
        # Calculate growth rate
        predicted_qty = float(pred_row['predicted_monthly_quantity'])
        growth_rate = ((predicted_qty - current_qty) / current_qty * 100) if current_qty > 0 else 0
        
        # Get godown distribution
        sku_sales = sales_data[sales_data['SKU'] == sku]
        godown_dist = sku_sales.groupby('Godown')['Quantity'].sum().sort_values(ascending=False)
        godown_distribution = [
            {'godown': godown, 'quantity': int(qty)} 
            for godown, qty in godown_dist.head(3).items()
        ]
        
        comprehensive_data.append({
            'sku': sku,
            'product_name': data_provider.sku_mapping.get(sku, 'Unknown Product'),
            'current_monthly_quantity': current_qty,
            'predicted_monthly_quantity': predicted_qty,
            'growth_rate_percent': round(growth_rate, 1),
            'lead_time_days': lead_time,
            'recommended_order_quantity': recommended_qty,
            'current_stock': current_stock,
            'days_remaining': round(days_remaining, 1),
            'urgency': urgency,
            'estimated_cost_inr': estimated_cost,
            'confidence': pred_row['confidence'],
            'godown_distribution': godown_distribution,
            'seasonal_multiplier': 1.0  # Placeholder
        })
    
    # Calculate summary statistics
    critical_count = len([item for item in comprehensive_data if item['urgency'] == 'CRITICAL'])
    high_count = len([item for item in comprehensive_data if item['urgency'] == 'HIGH'])
    total_estimated_cost = sum(item['estimated_cost_inr'] for item in comprehensive_data)
    avg_growth_rate = sum(item['growth_rate_percent'] for item in comprehensive_data) / len(comprehensive_data)
    
    # Godown statistics
    godown_stats = sales_data.groupby('Godown').agg({
        'Quantity': 'sum',
        'Amount': 'sum'
    }).sort_values('Quantity', ascending=False).head(5).to_dict('index')
    
    return {
        'comprehensive_data': comprehensive_data,
        'summary': {
            'total_skus': len(comprehensive_data),
            'critical_orders': critical_count,
            'high_priority_orders': high_count,
            'total_estimated_cost': total_estimated_cost,
            'average_growth_rate': round(avg_growth_rate, 1),
            'unique_godowns': sales_data['Godown'].nunique(),
            'states_served_from': sales_data['Statefrom'].nunique(),
            'destination_states': sales_data['Stateto'].nunique()
        },
        'godown_stats': godown_stats
    }

@app.get("/export/ordering-schedule-csv")
async def export_ordering_schedule_csv():
    """Export ordering schedule data to CSV"""
    if data_provider.ordering_schedule is None:
        raise HTTPException(status_code=404, detail="No ordering schedule available")

    # Create CSV content
    output = io.StringIO()
    data_provider.ordering_schedule.to_csv(output, index=False)
    output.seek(0)

    # Create response
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=ordering_schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
    )

@app.get("/export/ordering-schedule-excel")
async def export_ordering_schedule_excel():
    """Export ordering schedule data to Excel"""
    if data_provider.ordering_schedule is None:
        raise HTTPException(status_code=404, detail="No ordering schedule available")

    # Create Excel file in memory
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Main ordering schedule
        schedule_df = data_provider.ordering_schedule.copy()
        schedule_df.columns = [col.replace('_', ' ').title() for col in schedule_df.columns]
        schedule_df.to_excel(writer, sheet_name='Ordering Schedule', index=False)

        # Summary by urgency
        urgency_summary = data_provider.ordering_schedule.groupby('urgency').agg({
            'recommended_qty': ['sum', 'count'],
            'estimated_cost': 'sum'
        }).round(2)
        urgency_summary.to_excel(writer, sheet_name='Summary by Urgency')

        # Critical items only
        critical_items = data_provider.ordering_schedule[data_provider.ordering_schedule['urgency'] == 'CRITICAL']
        if not critical_items.empty:
            critical_items.to_excel(writer, sheet_name='Critical Orders Only', index=False)

    output.seek(0)

    # Create response
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=ordering_schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"}
    )

@app.get("/export/comprehensive-ordering-csv")
async def export_comprehensive_csv():
    """Export comprehensive ordering data to CSV"""
    # Get the comprehensive data
    data = await comprehensive_ordering()
    
    # Create CSV content
    output = io.StringIO()
    
    # Write header
    headers = [
        'Product_Name', 'SKU', 'Current_Monthly_Quantity', 'Predicted_Monthly_Quantity',
        'Growth_Rate_Percent', 'Lead_Time_Days', 'Recommended_Order_Quantity',
        'Current_Stock', 'Days_Remaining', 'Urgency', 'Estimated_Cost_INR',
        'Confidence', 'Godown_Distribution', 'Seasonal_Multiplier'
    ]
    output.write(','.join(headers) + '\n')
    
    # Write data
    for item in data['comprehensive_data']:
        godown_dist = '; '.join([f"{g['godown']}({g['quantity']})" for g in item['godown_distribution']])
        row = [
            item['product_name'],
            item['sku'],
            str(item['current_monthly_quantity']),
            str(item['predicted_monthly_quantity']),
            str(item['growth_rate_percent']),
            str(item['lead_time_days']),
            str(item['recommended_order_quantity']),
            str(item['current_stock']),
            str(item['days_remaining']),
            item['urgency'],
            str(item['estimated_cost_inr']),
            item['confidence'],
            godown_dist,
            str(item['seasonal_multiplier'])
        ]
        output.write(','.join([f'"{field}"' for field in row]) + '\n')
    
    # Create response
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=comprehensive_ordering_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
    )

@app.get("/api/inventory-status")
async def inventory_status():
    """API endpoint for inventory status"""
    try:
        if data_provider.ordering_schedule is None:
            return {"critical_items": [], "low_stock_items": [], "message": "No inventory data available"}
        
        ordering = data_provider.ordering_schedule.copy()
        
        # Critical items (immediate attention needed)
        critical_items = []
        critical_orders = ordering[ordering['urgency'] == 'CRITICAL']
        for _, row in critical_orders.iterrows():
            critical_items.append({
                'sku': row['sku'],
                'product_name': data_provider.sku_mapping.get(row['sku'], 'Unknown Product'),
                'current_stock': int(row['current_stock']),
                'days_remaining': float(row['days_remaining']),
                'recommended_qty': int(row['recommended_qty']),
                'estimated_cost': float(row['estimated_cost']),
                'urgency': row['urgency'],
                'daily_demand': float(row['daily_demand'])
            })
        
        # Low stock items (high priority)
        low_stock_items = []
        high_orders = ordering[ordering['urgency'] == 'HIGH']
        for _, row in high_orders.iterrows():
            low_stock_items.append({
                'sku': row['sku'],
                'product_name': data_provider.sku_mapping.get(row['sku'], 'Unknown Product'),
                'current_stock': int(row['current_stock']),
                'days_remaining': float(row['days_remaining']),
                'recommended_qty': int(row['recommended_qty']),
                'estimated_cost': float(row['estimated_cost']),
                'urgency': row['urgency'],
                'daily_demand': float(row['daily_demand'])
            })
        
        # Inventory health metrics
        total_items = len(ordering)
        critical_count = len(critical_items)
        high_count = len(low_stock_items)
        medium_count = len(ordering[ordering['urgency'] == 'MEDIUM'])
        low_count = len(ordering[ordering['urgency'] == 'LOW'])
        
        # Calculate total investment needed
        total_critical_cost = sum(item['estimated_cost'] for item in critical_items)
        total_high_cost = sum(item['estimated_cost'] for item in low_stock_items)
        
        # Urgency distribution
        urgency_distribution = {
            'critical': critical_count,
            'high': high_count,
            'medium': medium_count,
            'low': low_count
        }
        
        return {
            "critical_items": critical_items[:10],  # Top 10 most critical
            "low_stock_items": low_stock_items[:10],  # Top 10 low stock
            "summary": {
                "total_items": total_items,
                "critical_items": critical_count,
                "high_priority_items": high_count,
                "medium_priority_items": medium_count,
                "low_priority_items": low_count,
                "total_critical_investment": total_critical_cost,
                "total_high_investment": total_high_cost,
                "health_score": round((low_count + medium_count) / total_items * 100, 1) if total_items > 0 else 0
            },
            "urgency_distribution": urgency_distribution
        }
        
    except Exception as e:
        return {
            "critical_items": [],
            "low_stock_items": [],
            "error": f"Inventory status failed: {str(e)}"
        }

@app.get("/api/key-insights")
async def key_insights():
    """API endpoint for key insights"""
    try:
        insights = []
        
        # Critical stock alerts
        if data_provider.ordering_schedule is not None:
            critical_count = len(data_provider.ordering_schedule[data_provider.ordering_schedule['urgency'] == 'CRITICAL'])
            if critical_count > 0:
                total_cost = data_provider.ordering_schedule[data_provider.ordering_schedule['urgency'] == 'CRITICAL']['estimated_cost'].sum()
                insights.append({
                    "type": "critical",
                    "title": f"{critical_count} Critical Stock Items",
                    "message": f"Immediate ordering required - ₹{total_cost:,.0f} investment needed",
                    "action": "Place orders today to avoid stockouts"
                })
        
        # Growth insights
        if data_provider.sales_data is not None:
            monthly_data = data_provider.sales_data.groupby(data_provider.sales_data['Date'].dt.to_period('M'))['Amount'].sum()
            if len(monthly_data) > 1:
                growth_rate = ((monthly_data.iloc[-1] - monthly_data.iloc[-2]) / monthly_data.iloc[-2] * 100)
                insights.append({
                    "type": "growth" if growth_rate > 0 else "warning",
                    "title": f"Monthly Growth: {growth_rate:+.1f}%",
                    "message": f"Revenue trend is {'positive' if growth_rate > 0 else 'declining'}",
                    "action": "Monitor key performance indicators"
                })
        
        # Prediction confidence insights
        if data_provider.predictions is not None:
            high_confidence = len(data_provider.predictions[data_provider.predictions['confidence'] == 'High'])
            total_predictions = len(data_provider.predictions)
            confidence_pct = (high_confidence / total_predictions * 100) if total_predictions > 0 else 0
            
            insights.append({
                "type": "info",
                "title": f"Prediction Confidence: {confidence_pct:.0f}%",
                "message": f"{high_confidence} of {total_predictions} predictions are high confidence",
                "action": "Focus on high-confidence SKUs for planning"
            })
        
        # Top performer insight
        if data_provider.sales_data is not None:
            top_sku = data_provider.sales_data.groupby('SKU')['Amount'].sum().idxmax()
            top_revenue = data_provider.sales_data.groupby('SKU')['Amount'].sum().max()
            product_name = data_provider.sku_mapping.get(top_sku, 'Unknown Product')
            
            insights.append({
                "type": "success",
                "title": f"Top Performer: {top_sku}",
                "message": f"{product_name} generated ₹{top_revenue:,.0f} revenue",
                "action": "Ensure adequate stock levels for top performers"
            })
        
        return {
            "insights": insights,
            "total_alerts": len([i for i in insights if i["type"] in ["critical", "warning"]]),
            "last_updated": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "insights": [{
                "type": "error",
                "title": "Data Processing Error",
                "message": f"Unable to generate insights: {str(e)}",
                "action": "Check data availability"
            }],
            "total_alerts": 1,
            "last_updated": datetime.now().isoformat()
        }

@app.get("/api/historical-analysis")
async def historical_analysis():
    """API endpoint for historical analysis"""
    try:
        if data_provider.historical_data is None and data_provider.sales_data is None:
            return {"trends": [], "seasonal_patterns": [], "message": "No historical data available"}
        
        # Use historical data if available, otherwise use current sales data
        data_to_analyze = data_provider.historical_data if data_provider.historical_data is not None else data_provider.sales_data
        
        # Monthly trends
        monthly_trends = data_to_analyze.groupby(data_to_analyze['Date'].dt.to_period('M')).agg({
            'Amount': 'sum',
            'Quantity': 'sum'
        }).reset_index()
        
        trends_data = []
        for _, row in monthly_trends.iterrows():
            trends_data.append({
                'period': str(row['Date']),
                'revenue': float(row['Amount']),
                'quantity': int(row['Quantity'])
            })
        
        # Seasonal patterns (by month)
        seasonal_data = data_to_analyze.groupby(data_to_analyze['Date'].dt.month).agg({
            'Amount': 'mean',
            'Quantity': 'mean'
        }).reset_index()
        
        seasonal_patterns = []
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for _, row in seasonal_data.iterrows():
            seasonal_patterns.append({
                'month': month_names[int(row['Date']) - 1],
                'avg_revenue': float(row['Amount']),
                'avg_quantity': float(row['Quantity'])
            })
        
        # Year-over-year growth (if we have multiple years)
        yearly_data = data_to_analyze.groupby(data_to_analyze['Date'].dt.year)['Amount'].sum()
        yoy_growth = []
        if len(yearly_data) > 1:
            for i in range(1, len(yearly_data)):
                prev_year = yearly_data.iloc[i-1]
                curr_year = yearly_data.iloc[i]
                growth_rate = ((curr_year - prev_year) / prev_year * 100) if prev_year > 0 else 0
                yoy_growth.append({
                    'year': int(yearly_data.index[i]),
                    'revenue': float(curr_year),
                    'growth_rate': float(growth_rate)
                })
        
        return {
            "trends": trends_data[-24:],  # Last 24 months
            "yearly_trend": yoy_growth,  # For the historical revenue chart
            "seasonal_patterns": seasonal_patterns,
            "yoy_growth": yoy_growth,
            "total_periods": len(trends_data),
            "date_range": {
                "start": data_to_analyze['Date'].min().strftime('%Y-%m-%d'),
                "end": data_to_analyze['Date'].max().strftime('%Y-%m-%d')
            }
        }
        
    except Exception as e:
        return {
            "trends": [],
            "seasonal_patterns": [],
            "error": f"Historical analysis failed: {str(e)}"
        }

@app.get("/api/prediction-comparison")
async def prediction_comparison():
    """API endpoint for prediction comparison"""
    try:
        if data_provider.predictions is None or data_provider.sales_data is None:
            return {"comparisons": [], "message": "Insufficient data for comparison"}
        
        # Handle both 'SKU' and 'sku' column names in predictions
        pred_sku_col = 'SKU' if 'SKU' in data_provider.predictions.columns else 'sku'
        
        # Get current month sales data for comparison
        current_sales = data_provider.sales_data.groupby('SKU').agg({
            'Quantity': 'sum',
            'Amount': 'sum'
        }).reset_index()
        
        comparisons = []
        for _, pred_row in data_provider.predictions.iterrows():
            sku = pred_row[pred_sku_col]
            predicted_qty = float(pred_row['predicted_monthly_quantity'])
            
            # Find actual sales for this SKU
            actual_row = current_sales[current_sales['SKU'] == sku]
            actual_qty = int(actual_row['Quantity'].iloc[0]) if len(actual_row) > 0 else 0
            
            # Calculate accuracy metrics
            if actual_qty > 0:
                accuracy = (1 - abs(predicted_qty - actual_qty) / actual_qty) * 100
                accuracy = max(0, min(100, accuracy))  # Clamp between 0-100%
            else:
                accuracy = 0 if predicted_qty > 0 else 100
            
            variance = predicted_qty - actual_qty
            variance_pct = (variance / actual_qty * 100) if actual_qty > 0 else 0
            
            comparisons.append({
                'sku': sku,
                'product_name': data_provider.sku_mapping.get(sku, 'Unknown Product'),
                'predicted_quantity': predicted_qty,
                'actual_quantity': actual_qty,
                'accuracy_percent': round(accuracy, 1),
                'variance': round(variance, 1),
                'variance_percent': round(variance_pct, 1),
                'confidence': pred_row['confidence'],
                'status': 'over_predicted' if variance > 0 else 'under_predicted' if variance < 0 else 'accurate'
            })
        
        # Sort by accuracy (best first)
        comparisons.sort(key=lambda x: x['accuracy_percent'], reverse=True)
        
        # Calculate summary statistics
        accuracies = [c['accuracy_percent'] for c in comparisons]
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        
        over_predicted = len([c for c in comparisons if c['status'] == 'over_predicted'])
        under_predicted = len([c for c in comparisons if c['status'] == 'under_predicted'])
        accurate = len([c for c in comparisons if c['status'] == 'accurate'])
        
        return {
            "comparisons": comparisons[:20],  # Top 20 for display
            "summary": {
                "average_accuracy": round(avg_accuracy, 1),
                "total_items": len(comparisons),
                "over_predicted": over_predicted,
                "under_predicted": under_predicted,
                "accurate": accurate,
                "high_accuracy_items": len([c for c in comparisons if c['accuracy_percent'] >= 80])
            }
        }
        
    except Exception as e:
        return {
            "comparisons": [],
            "error": f"Prediction comparison failed: {str(e)}"
        }

@app.post("/upload/sales-data")
async def upload_sales_data(file: UploadFile = File(...)):
    """Upload new sales data and trigger re-analysis"""
    try:
        # Validate file type
        if not file.filename.endswith(('.csv', '.xlsx')):
            raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")
        
        # Save uploaded file
        file_path = PROCESSED_DIR / f"sales_data_uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Read and validate the uploaded file
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            # Save CSV directly
            with open(file_path, 'wb') as f:
                f.write(contents)
        else:
            # Convert Excel to CSV
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
                temp_file.write(contents)
                temp_file.flush()
                
                # Read Excel and convert to CSV
                df = pd.read_excel(temp_file.name)
                df.to_csv(file_path, index=False)
        
        # Validate data format
        df = pd.read_csv(file_path)
        required_columns = ['Date', 'SKU', 'Quantity', 'Amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            file_path.unlink()  # Delete invalid file
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}. Required: {required_columns}"
            )
        
        # Reload data provider
        data_provider.load_all_data()
        
        return {
            "status": "success",
            "message": f"File uploaded successfully as {file_path.name}",
            "records_processed": len(df),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/upload/sku-mapping")
async def upload_sku_mapping(file: UploadFile = File(...)):
    """Upload new SKU mapping data"""
    try:
        if not file.filename.endswith(('.csv', '.xlsx')):
            raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")
        
        # Save uploaded file
        file_path = DATA_DIR / "raw" / "sku_list_uploaded.csv"
        
        contents = await file.read()
        
        if file.filename.endswith('.csv'):
            with open(file_path, 'wb') as f:
                f.write(contents)
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
                temp_file.write(contents)
                temp_file.flush()
                df = pd.read_excel(temp_file.name)
                df.to_csv(file_path, index=False)
        
        # Validate format
        df = pd.read_csv(file_path)
        if 'sku' not in df.columns or 'category' not in df.columns:
            file_path.unlink()
            raise HTTPException(status_code=400, detail="SKU mapping must have 'sku' and 'category' columns")
        
        # Reload data provider
        data_provider.load_all_data()
        
        return {
            "status": "success",
            "message": "SKU mapping uploaded successfully",
            "skus_processed": len(df),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8080, reload=True)
