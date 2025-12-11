#!/usr/bin/env python3
"""
Improved Monthly Predictor that uses historical data for seasonal patterns
but only 2024+ data for actual predictions (accounting for business restart).
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import os

class ImprovedMonthlyPredictor:
    def __init__(self):
        self.historical_seasonal_patterns = {}
        self.historical_yoy_patterns = {}
        
    def load_and_prepare_data(self):
        """Load and prepare data for improved monthly prediction."""
        print("📊 Loading data for improved monthly prediction...")
        
        # Load historical data for pattern extraction (2018-2023)
        historical_df = None
        if os.path.exists('datasets/raw/Sales-Table 1.csv'):
            historical_df = pd.read_csv('datasets/raw/Sales-Table 1.csv')
            print("📊 Loaded historical data from datasets/raw/Sales-Table 1.csv")
        elif os.path.exists('datasets/raw/SalesAthena.csv'):
            historical_df = pd.read_csv('datasets/raw/SalesAthena.csv')
            print("📊 Loaded historical data from datasets/raw/SalesAthena.csv")
        elif os.path.exists('datasets/raw/SalesDataAthenasql (1)/Sales-Table 1.csv'):
            historical_df = pd.read_csv('datasets/raw/SalesDataAthenasql (1)/Sales-Table 1.csv')
            print("📊 Loaded historical data from datasets/raw/SalesDataAthenasql (1)/Sales-Table 1.csv")
        
        # Load current data for predictions (2024+)
        current_df = None
        if os.path.exists('datasets/processed/uploaded_sales.csv'):
            current_df = pd.read_csv('datasets/processed/uploaded_sales.csv')
            print("📊 Loaded current data from datasets/processed/uploaded_sales.csv")
        elif os.path.exists('datasets/processed/training_data_jan_june.csv'):
            current_df = pd.read_csv('datasets/processed/training_data_jan_june.csv')
            print("📊 Loaded current data from datasets/processed/training_data_jan_june.csv")
        
        if historical_df is None and current_df is None:
            raise FileNotFoundError("No sales data found")
        
        # Combine data if we have both
        if historical_df is not None and current_df is not None:
            # Ensure both dataframes have the same column names
            if 'sku' in historical_df.columns and 'SKU' in current_df.columns:
                current_df = current_df.rename(columns={'SKU': 'sku'})
            elif 'SKU' in historical_df.columns and 'sku' in current_df.columns:
                historical_df = historical_df.rename(columns={'SKU': 'sku'})
            
            df = pd.concat([historical_df, current_df], ignore_index=True)
        elif historical_df is not None:
            df = historical_df
        else:
            df = current_df
            
        df.columns = df.columns.str.strip()
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Standardize column names
        if 'sku' in df.columns:
            df = df.rename(columns={'sku': 'SKU'})
        
        # Create time features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        
        # Monthly aggregation by SKU
        monthly_sales = df.groupby(['SKU', 'Year', 'Month']).agg({
            'Quantity': 'sum',
            'Amount': 'sum',
            'Rate': 'mean'
        }).reset_index()
        
        # Create date column for sorting
        monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(day=1))
        monthly_sales = monthly_sales.sort_values(['SKU', 'Date'])
        
        print(f"✅ Prepared {len(monthly_sales)} monthly records")
        return monthly_sales
    
    def extract_historical_patterns(self, monthly_data):
        """Extract seasonal and YoY patterns from historical data (2018-2023)."""
        print("🔍 Extracting historical patterns...")
        
        # Filter historical data (2018-2023)
        historical_data = monthly_data[monthly_data['Year'] < 2024].copy()
        
        if len(historical_data) == 0:
            print("⚠️ No historical data found for pattern extraction")
            return
        
        # Extract seasonal patterns by SKU
        for sku in historical_data['SKU'].unique():
            sku_data = historical_data[historical_data['SKU'] == sku].copy()
            
            if len(sku_data) < 12:  # Need at least 1 year of historical data
                continue
            
            # Calculate seasonal factors (monthly averages)
            seasonal_factors = sku_data.groupby('Month')['Quantity'].mean()
            overall_avg = sku_data['Quantity'].mean()
            
            # Normalize seasonal factors
            if overall_avg > 0:
                seasonal_factors = seasonal_factors / overall_avg
                self.historical_seasonal_patterns[sku] = seasonal_factors.to_dict()
            
            # Calculate YoY growth patterns (if we have multiple years)
            years = sorted(sku_data['Year'].unique())
            if len(years) >= 2:
                yoy_growths = []
                for i in range(1, len(years)):
                    year1_data = sku_data[sku_data['Year'] == years[i-1]]
                    year2_data = sku_data[sku_data['Year'] == years[i]]
                    
                    if len(year1_data) > 0 and len(year2_data) > 0:
                        yoy_growth = (year2_data['Quantity'].sum() - year1_data['Quantity'].sum()) / year1_data['Quantity'].sum()
                        yoy_growths.append(yoy_growth)
                
                if yoy_growths:
                    avg_yoy_growth = np.mean(yoy_growths)
                    self.historical_yoy_patterns[sku] = avg_yoy_growth
        
        print(f"✅ Extracted patterns for {len(self.historical_seasonal_patterns)} SKUs")
    
    def create_improved_features(self, monthly_data, sku):
        """Create features using only 2024+ data but informed by historical patterns."""
        # Filter to 2024+ data only (post-restart)
        sku_data = monthly_data[(monthly_data['SKU'] == sku) & (monthly_data['Year'] >= 2024)].copy()
        
        if len(sku_data) < 3:  # Need at least 3 months of 2024 data
            return None
        
        # Basic features from 2024 data only
        sku_data['Quantity_Lag1'] = sku_data['Quantity'].shift(1)
        sku_data['Quantity_Lag2'] = sku_data['Quantity'].shift(2)
        sku_data['Quantity_Lag3'] = sku_data['Quantity'].shift(3)
        
        # Moving averages from 2024 data
        sku_data['MA_2'] = sku_data['Quantity'].rolling(2).mean()
        sku_data['MA_3'] = sku_data['Quantity'].rolling(3).mean()
        sku_data['MA_6'] = sku_data['Quantity'].rolling(6).mean() if len(sku_data) >= 6 else sku_data['Quantity'].rolling(3).mean()
        
        # Trend from 2024 data
        sku_data['Trend'] = range(len(sku_data))
        
        # Seasonal features (month-based)
        sku_data['Month_Sin'] = np.sin(2 * np.pi * sku_data['Month'] / 12)
        sku_data['Month_Cos'] = np.cos(2 * np.pi * sku_data['Month'] / 12)
        
        # Apply historical seasonal adjustment if available
        if sku in self.historical_seasonal_patterns:
            seasonal_factors = self.historical_seasonal_patterns[sku]
            sku_data['Historical_Seasonal_Factor'] = sku_data['Month'].map(seasonal_factors).fillna(1.0)
        else:
            sku_data['Historical_Seasonal_Factor'] = 1.0
        
        # Apply historical YoY growth expectation if available
        if sku in self.historical_yoy_patterns:
            sku_data['Historical_YoY_Growth'] = self.historical_yoy_patterns[sku]
        else:
            sku_data['Historical_YoY_Growth'] = 0.0
        
        # Business restart indicator
        sku_data['is_post_restart'] = 1  # All 2024+ data is post-restart
        
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
    
    def train_improved_model(self, sku_data, sku):
        """Train model using improved features."""
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
    
    def predict_next_month_improved(self, model, sku_data, sku):
        """Predict next month using improved approach."""
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
        
        # Apply historical seasonal factor for next month
        if sku in self.historical_seasonal_patterns:
            seasonal_factors = self.historical_seasonal_patterns[sku]
            last_row['Historical_Seasonal_Factor'] = seasonal_factors.get(next_month, 1.0)
        else:
            last_row['Historical_Seasonal_Factor'] = 1.0
        
        # Apply historical YoY growth
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
        if sku in self.historical_seasonal_patterns:
            seasonal_factor = self.historical_seasonal_patterns[sku].get(next_month, 1.0)
            prediction = prediction * seasonal_factor
        
        return {
            'sku': sku,
            'next_month': f"{next_year}-{next_month:02d}",
            'predicted_quantity': prediction,
            'confidence': 'medium',
            'seasonal_adjustment': seasonal_factor if sku in self.historical_seasonal_patterns else 1.0
        }
    
    def evaluate_improved_accuracy(self, y_test, y_pred, sku):
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
    """Test the improved predictor."""
    print("="*60)
    print("🚀 TESTING IMPROVED MONTHLY PREDICTOR".center(60))
    print("="*60)
    
    try:
        # Initialize predictor
        predictor = ImprovedMonthlyPredictor()
        
        # Load data
        monthly_data = predictor.load_and_prepare_data()
        
        # Extract historical patterns
        predictor.extract_historical_patterns(monthly_data)
        
        # Test with a few SKUs
        test_skus = ['ACLRM01', 'LRM02', 'CMSM01']
        
        for sku in test_skus:
            print(f"\n🔍 Testing SKU: {sku}")
            
            # Create improved features
            sku_data = predictor.create_improved_features(monthly_data, sku)
            
            if sku_data is not None:
                print(f"   ✅ Created features for {len(sku_data)} records")
                
                # Train model
                model, X_test, y_test, y_pred = predictor.train_improved_model(sku_data, sku)
                
                if model is not None:
                    print(f"   ✅ Model trained successfully")
                    
                    # Evaluate accuracy
                    accuracy = predictor.evaluate_improved_accuracy(y_test, y_pred, sku)
                    if accuracy:
                        print(f"   📊 MAE: {accuracy['mae']:.2f}, MAPE: {accuracy['mape']:.1f}%")
                    
                    # Predict next month
                    prediction = predictor.predict_next_month_improved(model, sku_data, sku)
                    if prediction:
                        print(f"   🔮 Next month prediction: {prediction['predicted_quantity']:.0f}")
                        print(f"   📅 Seasonal adjustment: {prediction['seasonal_adjustment']:.2f}")
                else:
                    print(f"   ❌ Model training failed")
            else:
                print(f"   ❌ No features created")
        
        print("\n" + "="*60)
        print("✅ IMPROVED PREDICTOR TEST COMPLETE".center(60))
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
