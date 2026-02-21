#!/usr/bin/env python3
"""
Python Bridge Script for C# Integration
Provides a command-line interface for the hybrid prediction system
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add the parent directory to Python path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from core import HybridPredictionEngine, PredictionAPI
except ImportError:
    # Fallback for direct execution
    sys.path.append(str(Path(__file__).parent.parent / "core"))
    from hybrid_prediction_engine import HybridPredictionEngine
    from prediction_api import PredictionAPI

def convert_csharp_to_python_data(csharp_data):
    """Convert C# input format to Python pandas DataFrame"""
    historical_data = csharp_data['historical_data']
    
    # Convert to pandas DataFrame
    df_data = []
    for record in historical_data:
        df_data.append({
            'Date': pd.to_datetime(record['date']),
            'Quantity': float(record['quantity'])
        })
    
    df = pd.DataFrame(df_data)
    df = df.sort_values('Date')
    
    return df, csharp_data.get('horizon_days', 30), csharp_data.get('use_hybrid', True)

def extract_sku_from_data(df):
    """Extract or generate SKU identifier from data"""
    # For now, use a generic SKU name
    # In production, this would come from the C# application
    return "CSHARP_SKU"

def generate_daily_predictions(monthly_prediction, horizon_days):
    """Convert monthly prediction to daily predictions"""
    # Simple distribution: divide monthly by days in month
    daily_avg = monthly_prediction / 30.0
    
    # Create daily predictions with some variation
    daily_predictions = []
    for i in range(horizon_days):
        # Add slight variation (±10%) to make it more realistic
        variation = 1.0 + (i % 7 - 3) * 0.03  # Weekly pattern
        daily_pred = daily_avg * variation
        daily_predictions.append(max(0, daily_pred))  # Ensure non-negative
    
    return daily_predictions

def calculate_confidence_bounds(predictions, confidence_level=0.90):
    """Calculate confidence bounds for predictions"""
    # Use a simple approach: ±25% for high confidence, ±50% for low confidence
    margin_factor = 0.25 if confidence_level > 0.85 else 0.50
    
    lower_bounds = [max(0, pred * (1 - margin_factor)) for pred in predictions]
    upper_bounds = [pred * (1 + margin_factor) for pred in predictions]
    
    return lower_bounds, upper_bounds

def main():
    if len(sys.argv) != 4:
        print("Usage: python_prediction_bridge.py <input_file> <output_file> <data_directory>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    data_directory = sys.argv[3]
    
    try:
        # Read input from C#
        with open(input_file, 'r') as f:
            csharp_input = json.load(f)
        
        # Convert input format
        df, horizon_days, use_hybrid = convert_csharp_to_python_data(csharp_input)
        
        if len(df) == 0:
            raise ValueError("No historical data provided")
        
        # Extract SKU (in production, this would come from C# context)
        sku = extract_sku_from_data(df)
        
        # Initialize prediction system
        data_dir = Path(data_directory)
        
        if use_hybrid:
            # Try to use hybrid prediction system
            try:
                api = PredictionAPI(data_dir, use_hybrid=True)
                
                # Convert horizon from days to months (approximate)
                months_ahead = max(1, horizon_days // 30)
                
                # Get prediction
                result = api.get_prediction(sku, months_ahead=months_ahead)
                
                if 'error' not in result:
                    # Extract prediction information
                    monthly_prediction = result['predictions'][0]
                    confidence_info = result.get('confidence', {})
                    hybrid_info = result.get('hybrid_info', {})
                    historical_accuracy = result.get('historical_accuracy', {})
                    
                    # Convert to daily predictions
                    daily_predictions = generate_daily_predictions(monthly_prediction, horizon_days)
                    
                    # Calculate confidence bounds
                    confidence_level = 0.90 if confidence_info.get('confidence_level') == 'High' else 0.70
                    lower_bounds, upper_bounds = calculate_confidence_bounds(daily_predictions, confidence_level)
                    
                    # Prepare output
                    output = {
                        'predictions': daily_predictions,
                        'confidence_lower': lower_bounds,
                        'confidence_upper': upper_bounds,
                        'confidence_level': confidence_level,
                        'approach_used': hybrid_info.get('approach_used', 'hybrid'),
                        'decision_reason': hybrid_info.get('decision_reason', 'Hybrid prediction system'),
                        'historical_mape': historical_accuracy.get('mape'),
                        'model_version': 'Hybrid Python Model v1.0.0',
                        'success': True
                    }
                else:
                    raise Exception(f"Prediction failed: {result['error']}")
                    
            except Exception as e:
                # Fallback to simple statistical approach
                output = create_fallback_prediction(df, horizon_days, str(e))
        else:
            # Use simple statistical approach
            output = create_fallback_prediction(df, horizon_days, "Non-hybrid mode requested")
        
        # Write output for C#
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print("Prediction completed successfully")
        
    except Exception as e:
        # Create error output
        error_output = {
            'predictions': [0.0] * horizon_days,
            'confidence_lower': [0.0] * horizon_days,
            'confidence_upper': [0.0] * horizon_days,
            'confidence_level': 0.50,
            'approach_used': 'error_fallback',
            'decision_reason': f'Error occurred: {str(e)}',
            'historical_mape': None,
            'model_version': 'Hybrid Python Model v1.0.0',
            'success': False,
            'error': str(e)
        }
        
        with open(output_file, 'w') as f:
            json.dump(error_output, f, indent=2)
        
        print(f"Error: {str(e)}")
        sys.exit(1)

def create_fallback_prediction(df, horizon_days, reason):
    """Create a simple statistical fallback prediction"""
    # Calculate basic statistics from historical data
    recent_data = df.tail(30)  # Last 30 days
    mean_quantity = recent_data['Quantity'].mean()
    std_quantity = recent_data['Quantity'].std()
    
    # Generate predictions based on historical mean
    predictions = [max(0, mean_quantity) for _ in range(horizon_days)]
    
    # Calculate confidence bounds using standard deviation
    margin = std_quantity * 1.96  # 95% confidence interval
    lower_bounds = [max(0, pred - margin) for pred in predictions]
    upper_bounds = [pred + margin for pred in predictions]
    
    return {
        'predictions': predictions,
        'confidence_lower': lower_bounds,
        'confidence_upper': upper_bounds,
        'confidence_level': 0.70,
        'approach_used': 'statistical_fallback',
        'decision_reason': f'Fallback to statistical approach: {reason}',
        'historical_mape': None,
        'model_version': 'Statistical Fallback v1.0.0',
        'success': True
    }

if __name__ == "__main__":
    main()
