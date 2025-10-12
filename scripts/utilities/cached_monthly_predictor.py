#!/usr/bin/env python3
"""
Cached Monthly Predictor
Uses pre-calculated historical patterns + new uploaded data for fast predictions.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import os
import json
from .historical_pattern_cache import HistoricalPatternCache

class CachedMonthlyPredictor:
    def __init__(self):
        self.pattern_cache = HistoricalPatternCache()
        self.historical_patterns = {}
        self.historical_yoy_patterns = {}
        
    def load_historical_patterns(self):
        """Load pre-calculated historical patterns from cache."""
        print("📂 Loading historical patterns from cache...")
        
        if not self.pattern_cache.is_cache_valid():
            print("⚠️ No valid cache found. Please run historical_pattern_cache.py first.")
            return False
        
        patterns = self.pattern_cache.load_patterns_from_cache()
        if patterns:
            self.historical_patterns = patterns.get('seasonal_patterns', {})
            self.historical_yoy_patterns = patterns.get('yoy_patterns', {})
            print(f"✅ Loaded patterns for {len(self.historical_patterns)} SKUs")
            return True
        
        return False
    
    def load_current_data(self):
        """Load current data (2024+ or uploaded data) for predictions."""
        print("📊 Loading current data for predictions...")
        
        # Priority: uploaded data > training data > main data
        current_df = None
        if os.path.exists('uploaded_sales.csv'):
            current_df = pd.read_csv('uploaded_sales.csv')
            print("📊 Using uploaded sales data")
        elif os.path.exists('training_data_jan_june.csv'):
            current_df = pd.read_csv('training_data_jan_june.csv')
            print("📊 Using training data")
        elif os.path.exists('Sales-Table 1.csv'):
            current_df = pd.read_csv('Sales-Table 1.csv')
            print("📊 Using main sales data")
        else:
            raise FileNotFoundError("No current data found")
        
        # Standardize column names
        current_df.columns = current_df.columns.str.strip()
        if 'sku' in current_df.columns:
            current_df = current_df.rename(columns={'sku': 'SKU'})
        
        # Parse dates
        current_df['Date'] = pd.to_datetime(current_df['Date'], errors='coerce')
        current_df = current_df.dropna(subset=['Date'])
        
        # Filter to 2024+ data (post-restart)
        current_df = current_df[current_df['Date'].dt.year >= 2024].copy()
        
        print(f"📅 Current data: {len(current_df):,} records from {current_df['Date'].min().strftime('%Y-%m-%d')} to {current_df['Date'].max().strftime('%Y-%m-%d')}")
        
        return current_df
    
    def prepare_monthly_data(self, current_df):
        """Prepare monthly aggregated data from current data."""
        print("📊 Preparing monthly data...")
        
        # Create time features
        current_df['Year'] = current_df['Date'].dt.year
        current_df['Month'] = current_df['Date'].dt.month
        current_df['Quarter'] = current_df['Date'].dt.quarter
        
        # Monthly aggregation by SKU
        monthly_sales = current_df.groupby(['SKU', 'Year', 'Month']).agg({
            'Quantity': 'sum',
            'Amount': 'sum',
            'Rate': 'mean'
        }).reset_index()
        
        # Create date column for sorting
        monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(day=1))
        monthly_sales = monthly_sales.sort_values(['SKU', 'Date'])
        
        print(f"✅ Prepared {len(monthly_sales)} monthly records")
        return monthly_sales
    
    def create_cached_features(self, monthly_data, sku):
        """Create features using current data + cached historical patterns."""
        # Filter to current data only (2024+)
        sku_data = monthly_data[(monthly_data['SKU'] == sku) & (monthly_data['Year'] >= 2024)].copy()
        
        if len(sku_data) < 3:  # Need at least 3 months of current data
            return None
        
        # Basic features from current data only
        sku_data['Quantity_Lag1'] = sku_data['Quantity'].shift(1)
        sku_data['Quantity_Lag2'] = sku_data['Quantity'].shift(2)
        sku_data['Quantity_Lag3'] = sku_data['Quantity'].shift(3)
        
        # Moving averages from current data
        sku_data['MA_2'] = sku_data['Quantity'].rolling(2).mean()
        sku_data['MA_3'] = sku_data['Quantity'].rolling(3).mean()
        sku_data['MA_6'] = sku_data['Quantity'].rolling(6).mean() if len(sku_data) >= 6 else sku_data['Quantity'].rolling(3).mean()
        
        # Trend from current data
        sku_data['Trend'] = range(len(sku_data))
        
        # Seasonal features (month-based)
        sku_data['Month_Sin'] = np.sin(2 * np.pi * sku_data['Month'] / 12)
        sku_data['Month_Cos'] = np.cos(2 * np.pi * sku_data['Month'] / 12)
        
        # Apply cached historical seasonal adjustment
        if sku in self.historical_patterns:
            seasonal_factors = self.historical_patterns[sku]
            sku_data['Historical_Seasonal_Factor'] = sku_data['Month'].map(seasonal_factors).fillna(1.0)
        else:
            sku_data['Historical_Seasonal_Factor'] = 1.0
        
        # Apply cached historical YoY growth expectation
        if sku in self.historical_yoy_patterns:
            sku_data['Historical_YoY_Growth'] = self.historical_yoy_patterns[sku]
        else:
            sku_data['Historical_YoY_Growth'] = 0.0
        
        # Business restart indicator
        sku_data['is_post_restart'] = 1  # All current data is post-restart
        
        # Handle missing values
        sku_data = sku_data.replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values with 0 for lag features and moving averages
        lag_cols = ['Quantity_Lag1', 'Quantity_Lag2', 'Quantity_Lag3']
        ma_cols = ['MA_2', 'MA_3', 'MA_6']
        
        for col in lag_cols + ma_cols:
            if col in sku_data.columns:
                sku_data[col] = sku_data[col].fillna(0)
        
        # Only remove rows where essential features are missing
        essential_cols = ['Quantity', 'Year', 'Month']
        sku_data = sku_data.dropna(subset=essential_cols)
        
        return sku_data
    
    def train_cached_model(self, sku_data, sku):
        """Train model using cached features."""
        if sku_data is None or len(sku_data) < 3:
            return None, None, None, None
        
        # Feature columns
        feature_cols = ['Year', 'Month', 'Quantity_Lag1', 'Quantity_Lag2', 'Quantity_Lag3',
                       'MA_2', 'MA_3', 'MA_6', 'Month_Sin', 'Month_Cos', 'Trend',
                       'Historical_Seasonal_Factor', 'Historical_YoY_Growth', 'is_post_restart']
        
        # Check which features exist
        available_cols = [col for col in feature_cols if col in sku_data.columns]
        
        X = sku_data[available_cols]
        y = sku_data['Quantity']
        
        # For small datasets, use all data for training
        if len(X) < 6:
            X_train, X_test = X, X.tail(1)
            y_train, y_test = y, y.tail(1)
        else:
            # Use 80/20 split
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            max_features='sqrt'
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)  # Ensure non-negative
        
        return model, X_test, y_test, y_pred
    
    def predict_next_month_cached(self, model, sku_data, sku):
        """Predict next month using cached approach."""
        if model is None or sku_data is None:
            return None
        
        # Get the last record
        last_row = sku_data.tail(1).copy()
        
        # Calculate next month
        last_date = last_row['Date'].iloc[0]
        if last_date.month == 12:
            next_year = last_date.year + 1
            next_month = 1
        else:
            next_year = last_date.year
            next_month = last_date.month + 1
        
        # Update features for next month
        last_row['Year'] = next_year
        last_row['Month'] = next_month
        last_row['Month_Sin'] = np.sin(2 * np.pi * next_month / 12)
        last_row['Month_Cos'] = np.cos(2 * np.pi * next_month / 12)
        last_row['Trend'] = last_row['Trend'].iloc[0] + 1
        
        # Apply cached historical seasonal factor for next month
        if sku in self.historical_patterns:
            seasonal_factors = self.historical_patterns[sku]
            seasonal_factor = seasonal_factors.get(str(next_month), 1.0)
            last_row['Historical_Seasonal_Factor'] = seasonal_factor
        else:
            seasonal_factor = 1.0
            last_row['Historical_Seasonal_Factor'] = 1.0
        
        # Apply cached historical YoY growth
        if sku in self.historical_yoy_patterns:
            last_row['Historical_YoY_Growth'] = self.historical_yoy_patterns[sku]
        else:
            last_row['Historical_YoY_Growth'] = 0.0
        
        # Business restart indicator
        last_row['is_post_restart'] = 1
        
        # Feature columns
        feature_cols = ['Year', 'Month', 'Quantity_Lag1', 'Quantity_Lag2', 'Quantity_Lag3',
                       'MA_2', 'MA_3', 'MA_6', 'Month_Sin', 'Month_Cos', 'Trend',
                       'Historical_Seasonal_Factor', 'Historical_YoY_Growth', 'is_post_restart']
        
        # Check which features exist
        available_cols = [col for col in feature_cols if col in last_row.columns]
        X_pred = last_row[available_cols]
        
        # Make prediction
        prediction = model.predict(X_pred)[0]
        prediction = max(0, prediction)  # Ensure non-negative
        
        # Apply seasonal adjustment
        prediction = prediction * seasonal_factor
        
        return {
            'sku': sku,
            'next_month': f"{next_year}-{next_month:02d}",
            'predicted_quantity': prediction,
            'confidence': 'medium',
            'seasonal_adjustment': seasonal_factor,
            'has_historical_patterns': sku in self.historical_patterns
        }
    
    def evaluate_cached_accuracy(self, y_test, y_pred, sku):
        """Evaluate prediction accuracy."""
        if len(y_test) == 0:
            return None
        
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        # Cap MAPE to prevent extremely high values
        mape = min(mape, 200)
        
        return {
            'sku': sku,
            'mae': mae,
            'mape': mape,
            'test_size': len(y_test)
        }

def main():
    """Test the cached predictor."""
    print("="*60)
    print("🚀 TESTING CACHED MONTHLY PREDICTOR".center(60))
    print("="*60)
    
    try:
        # Initialize predictor
        predictor = CachedMonthlyPredictor()
        
        # Load historical patterns
        if not predictor.load_historical_patterns():
            print("❌ Failed to load historical patterns")
            return
        
        # Load current data
        current_df = predictor.load_current_data()
        
        # Prepare monthly data
        monthly_data = predictor.prepare_monthly_data(current_df)
        
        # Test with a few SKUs
        test_skus = ['ACLRM01', 'LRM02', 'CMSM01']
        
        for sku in test_skus:
            print(f"\n🔍 Testing SKU: {sku}")
            
            # Create cached features
            sku_data = predictor.create_cached_features(monthly_data, sku)
            
            if sku_data is not None:
                print(f"   ✅ Created features for {len(sku_data)} records")
                
                # Train model
                model, X_test, y_test, y_pred = predictor.train_cached_model(sku_data, sku)
                
                if model is not None:
                    print(f"   ✅ Model trained successfully")
                    
                    # Evaluate accuracy
                    accuracy = predictor.evaluate_cached_accuracy(y_test, y_pred, sku)
                    if accuracy:
                        print(f"   📊 MAE: {accuracy['mae']:.2f}, MAPE: {accuracy['mape']:.1f}%")
                    
                    # Predict next month
                    prediction = predictor.predict_next_month_cached(model, sku_data, sku)
                    if prediction:
                        print(f"   🔮 Next month prediction: {prediction['predicted_quantity']:.0f}")
                        print(f"   📅 Seasonal adjustment: {prediction['seasonal_adjustment']:.2f}")
                        print(f"   📊 Has historical patterns: {prediction['has_historical_patterns']}")
                else:
                    print(f"   ❌ Model training failed")
            else:
                print(f"   ❌ No features created")
        
        print("\n" + "="*60)
        print("✅ CACHED PREDICTOR TEST COMPLETE".center(60))
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
