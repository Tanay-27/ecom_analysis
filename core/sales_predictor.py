#!/usr/bin/env python3
"""
Sales Prediction Algorithm for E-commerce Forecasting
Focus: Predict next month sales for top SKUs with 8-12% acceptable error
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set up paths
DATA_DIR = Path("/Users/tanayshah/Desktop/personal/projects/ecom_analysis/data")
PROCESSED_DIR = DATA_DIR / "processed"
CORE_DIR = Path("/Users/tanayshah/Desktop/personal/projects/ecom_analysis/core")

class SalesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.top_skus = None
        self.historical_data = None
        self.recent_data = None
        
    def load_data(self):
        """Load and prepare data for prediction"""
        print("=== Loading Data for Prediction ===")
        
        # Load recent sales data (Jan-June 2025)
        recent_file = PROCESSED_DIR / "sales_data_jan_june_2025.csv"
        self.recent_data = pd.read_csv(recent_file)
        self.recent_data['Date'] = pd.to_datetime(self.recent_data['Date'])
        
        # Load historical data (2018-Nov 2024)
        historical_file = PROCESSED_DIR / "historical_data_2018_nov2024.csv"
        try:
            self.historical_data = pd.read_csv(historical_file)
            self.historical_data['Date'] = pd.to_datetime(self.historical_data['Date'])
            print(f"Historical data loaded: {len(self.historical_data)} records")
        except Exception as e:
            print(f"Could not load historical data: {e}")
            self.historical_data = None
        
        print(f"Recent data loaded: {len(self.recent_data)} records")
        print(f"Date range: {self.recent_data['Date'].min()} to {self.recent_data['Date'].max()}")
        
    def identify_top_skus(self, top_n=15):
        """Identify top performing SKUs for focused prediction"""
        print(f"\n=== Identifying Top {top_n} SKUs ===")
        
        # Analyze SKU performance
        sku_performance = self.recent_data.groupby('SKU').agg({
            'Amount': 'sum',
            'Quantity': 'sum',
            'Date': ['min', 'max', 'count']
        })
        
        # Flatten columns
        sku_performance.columns = ['Total_Revenue', 'Total_Quantity', 'First_Sale', 'Last_Sale', 'Total_Orders']
        
        # Calculate metrics
        sku_performance['Days_Active'] = (sku_performance['Last_Sale'] - sku_performance['First_Sale']).dt.days + 1
        sku_performance['Daily_Avg_Quantity'] = sku_performance['Total_Quantity'] / sku_performance['Days_Active']
        sku_performance['Revenue_Per_Unit'] = sku_performance['Total_Revenue'] / sku_performance['Total_Quantity']
        
        # Select top SKUs by revenue
        self.top_skus = sku_performance.sort_values('Total_Revenue', ascending=False).head(top_n)
        
        print("Top SKUs selected:")
        for sku in self.top_skus.index[:10]:
            revenue = self.top_skus.loc[sku, 'Total_Revenue']
            daily_qty = self.top_skus.loc[sku, 'Daily_Avg_Quantity']
            print(f"  {sku}: ₹{revenue:,.0f} ({daily_qty:.1f} units/day)")
        
        return self.top_skus.index.tolist()
    
    def create_features(self, data, sku_list):
        """Create features for machine learning model"""
        print("\n=== Creating Features ===")
        
        features_list = []
        
        for sku in sku_list:
            sku_data = data[data['SKU'] == sku].copy()
            if len(sku_data) < 30:  # Need minimum data points
                continue
                
            # Sort by date
            sku_data = sku_data.sort_values('Date')
            
            # Create daily aggregations
            daily_sales = sku_data.groupby('Date').agg({
                'Quantity': 'sum',
                'Amount': 'sum'
            }).reset_index()
            
            # Create date features
            daily_sales['Year'] = daily_sales['Date'].dt.year
            daily_sales['Month'] = daily_sales['Date'].dt.month
            daily_sales['DayOfWeek'] = daily_sales['Date'].dt.dayofweek
            daily_sales['DayOfMonth'] = daily_sales['Date'].dt.day
            daily_sales['WeekOfYear'] = daily_sales['Date'].dt.isocalendar().week
            
            # Create lag features (previous days' sales)
            for lag in [1, 3, 7, 14, 30]:
                daily_sales[f'Quantity_lag_{lag}'] = daily_sales['Quantity'].shift(lag)
                daily_sales[f'Amount_lag_{lag}'] = daily_sales['Amount'].shift(lag)
            
            # Create rolling averages
            for window in [7, 14, 30]:
                daily_sales[f'Quantity_rolling_{window}'] = daily_sales['Quantity'].rolling(window=window).mean()
                daily_sales[f'Amount_rolling_{window}'] = daily_sales['Amount'].rolling(window=window).mean()
            
            # Create trend features
            daily_sales['Quantity_trend_7d'] = daily_sales['Quantity'].rolling(7).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 7 else 0)
            daily_sales['Quantity_trend_30d'] = daily_sales['Quantity'].rolling(30).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 30 else 0)
            
            # Add SKU identifier
            daily_sales['SKU'] = sku
            
            features_list.append(daily_sales)
        
        if features_list:
            features_df = pd.concat(features_list, ignore_index=True)
            # Drop rows with NaN values (due to lag features)
            features_df = features_df.dropna()
            print(f"Features created for {len(features_df)} data points across {len(sku_list)} SKUs")
            return features_df
        else:
            print("No features could be created")
            return None
    
    def train_model(self, features_df):
        """Train the sales prediction model"""
        print("\n=== Training Prediction Model ===")
        
        # Prepare features and target
        feature_columns = [col for col in features_df.columns if col not in ['Date', 'SKU', 'Quantity', 'Amount']]
        X = features_df[feature_columns]
        y = features_df['Quantity']  # Predict quantity
        
        # Split data (use last 30 days as validation)
        split_date = features_df['Date'].max() - timedelta(days=30)
        train_mask = features_df['Date'] <= split_date
        
        X_train, X_val = X[train_mask], X[~train_mask]
        y_train, y_val = y[train_mask], y[~train_mask]
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Validate model
        y_pred = self.model.predict(X_val_scaled)
        mape = mean_absolute_percentage_error(y_val, y_pred) * 100
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        print(f"Model Validation Results:")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  Target Error Range: 8-12%")
        
        if mape <= 12:
            print("✅ Model meets accuracy requirements!")
        else:
            print("⚠️  Model accuracy needs improvement")
        
        return mape, rmse
    
    def predict_next_month(self, sku_list):
        """Predict sales for next month for each SKU"""
        print("\n=== Predicting Next Month Sales ===")
        
        if self.model is None:
            print("Error: Model not trained yet")
            return None
        
        predictions = {}
        
        for sku in sku_list:
            try:
                # Get recent data for this SKU
                sku_data = self.recent_data[self.recent_data['SKU'] == sku].copy()
                if len(sku_data) < 30:
                    continue
                
                # Get last known values for feature creation
                sku_data = sku_data.sort_values('Date')
                last_date = sku_data['Date'].max()
                
                # Create features for prediction (using last known values)
                daily_sales = sku_data.groupby('Date').agg({
                    'Quantity': 'sum',
                    'Amount': 'sum'
                }).reset_index()
                
                if len(daily_sales) < 30:
                    continue
                
                # Use last 30 days average as baseline
                last_30_days = daily_sales.tail(30)
                avg_daily_quantity = last_30_days['Quantity'].mean()
                
                # Apply growth trend if available
                if len(daily_sales) >= 60:
                    recent_30 = daily_sales.tail(30)['Quantity'].mean()
                    previous_30 = daily_sales.iloc[-60:-30]['Quantity'].mean()
                    growth_rate = (recent_30 - previous_30) / previous_30 if previous_30 > 0 else 0
                else:
                    growth_rate = 0
                
                # Predict next month (30 days)
                predicted_daily = avg_daily_quantity * (1 + growth_rate)
                predicted_monthly = predicted_daily * 30
                
                # Add seasonality adjustment (simple approach)
                next_month = (last_date + timedelta(days=30)).month
                seasonal_multiplier = self.get_seasonal_multiplier(sku, next_month)
                predicted_monthly *= seasonal_multiplier
                
                predictions[sku] = {
                    'predicted_monthly_quantity': max(0, int(predicted_monthly)),
                    'predicted_daily_average': max(0, predicted_monthly / 30),
                    'growth_rate': growth_rate,
                    'seasonal_multiplier': seasonal_multiplier,
                    'confidence': 'Medium' if abs(growth_rate) < 0.2 else 'Low'
                }
                
            except Exception as e:
                print(f"Error predicting for {sku}: {e}")
                continue
        
        print(f"Predictions generated for {len(predictions)} SKUs")
        return predictions
    
    def get_seasonal_multiplier(self, sku, month):
        """Get seasonal adjustment multiplier for a given month"""
        # Simple seasonal adjustment based on historical patterns
        sku_data = self.recent_data[self.recent_data['SKU'] == sku].copy()
        sku_data['Month'] = pd.to_datetime(sku_data['Date']).dt.month
        
        monthly_avg = sku_data.groupby('Month')['Quantity'].mean()
        overall_avg = sku_data['Quantity'].mean()
        
        if month in monthly_avg.index and overall_avg > 0:
            return monthly_avg[month] / overall_avg
        else:
            return 1.0  # No adjustment if no data
    
    def run_prediction_pipeline(self):
        """Run the complete prediction pipeline"""
        print("Starting Sales Prediction Pipeline...")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Identify top SKUs
        top_sku_list = self.identify_top_skus(15)
        
        # Create features and train model
        features_df = self.create_features(self.recent_data, top_sku_list)
        
        if features_df is not None and len(features_df) > 100:
            # Train model
            mape, rmse = self.train_model(features_df)
            
            # Generate predictions
            predictions = self.predict_next_month(top_sku_list)
            
            # Save results
            self.save_predictions(predictions)
            
            return predictions, mape
        else:
            print("Insufficient data for model training")
            return None, None
    
    def save_predictions(self, predictions):
        """Save predictions to CSV file"""
        if predictions:
            results_dir = CORE_DIR / "analysis_results"
            results_dir.mkdir(exist_ok=True)
            
            pred_df = pd.DataFrame(predictions).T
            pred_df.index.name = 'SKU'
            pred_df.to_csv(results_dir / "next_month_predictions.csv")
            
            print(f"Predictions saved to: {results_dir / 'next_month_predictions.csv'}")

def main():
    """Main function to run sales prediction"""
    predictor = SalesPredictor()
    predictions, accuracy = predictor.run_prediction_pipeline()
    
    if predictions:
        print("\n=== Top Predictions Summary ===")
        for sku, pred in list(predictions.items())[:5]:
            monthly_qty = pred['predicted_monthly_quantity']
            daily_avg = pred['predicted_daily_average']
            confidence = pred['confidence']
            print(f"{sku}: {monthly_qty} units/month ({daily_avg:.1f}/day) - {confidence} confidence")
    
    return predictions, accuracy

if __name__ == "__main__":
    predictions, accuracy = main()
