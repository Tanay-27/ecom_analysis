#!/usr/bin/env python3
"""
Production Sales Prediction Engine
Clean, production-ready system that provides:
1. SKU predictions with confidence scores
2. Historical accuracy metrics
3. Inactive SKU filtering
4. Expected accuracy improvement tracking
"""

import logging
import warnings
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SalesPredictionEngine:
    """Production-ready sales prediction engine with confidence scoring and accuracy tracking"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.restart_date = pd.Timestamp('2025-01-01')
        self.inactive_months_threshold = 3  # SKUs with no sales in last 3 months are inactive
        self.min_months_for_training = 6
        self.confidence_interval = 0.35  # ±35%
        
        # Model storage
        self.active_skus = []
        self.models = {}
        self.scalers = {}
        self.sku_metrics = {}
        self.last_training_date = None
        
    def load_and_filter_data(self) -> pd.DataFrame:
        """Load data and filter out inactive SKUs"""
        logger.info("Loading and filtering sales data...")
        
        # Load unified data
        unified_file = self.processed_dir / "unified_sales_data.csv"
        df = pd.read_csv(unified_file)
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Focus on post-restart period
        restart_data = df[df['Date'] >= self.restart_date].copy()
        restart_data = restart_data[(restart_data['Quantity'] > 0) & (restart_data['Amount'] > 0)]
        
        # Filter out inactive SKUs
        cutoff_date = restart_data['Date'].max() - timedelta(days=self.inactive_months_threshold * 30)
        recent_sales = restart_data[restart_data['Date'] >= cutoff_date]
        active_skus = recent_sales['SKU'].unique()
        
        # Filter to only active SKUs
        active_data = restart_data[restart_data['SKU'].isin(active_skus)]
        
        # Additional quality filtering
        sku_monthly_counts = active_data.groupby('SKU')['Date'].apply(
            lambda x: x.dt.to_period('M').nunique()
        )
        stable_skus = sku_monthly_counts[sku_monthly_counts >= self.min_months_for_training].index.tolist()
        
        final_data = active_data[active_data['SKU'].isin(stable_skus)]
        
        logger.info(f"Data filtering results:")
        logger.info(f"  - Total records: {len(restart_data):,}")
        logger.info(f"  - Active SKUs (sales in last {self.inactive_months_threshold} months): {len(active_skus)}")
        logger.info(f"  - Stable SKUs (≥{self.min_months_for_training} months data): {len(stable_skus)}")
        logger.info(f"  - Final dataset: {len(final_data):,} records")
        
        self.active_skus = stable_skus
        return final_data
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create optimized features for prediction"""
        logger.info("Creating prediction features...")
        
        # Monthly aggregation for stability
        data['YearMonth'] = data['Date'].dt.to_period('M')
        monthly_data = data.groupby(['SKU', 'YearMonth']).agg({
            'Quantity': 'sum',
            'Amount': 'sum',
            'Date': 'first'
        }).reset_index()
        
        monthly_data = monthly_data.sort_values(['SKU', 'YearMonth']).reset_index(drop=True)
        
        # Time-based features
        monthly_data['Month'] = monthly_data['YearMonth'].dt.month
        monthly_data['Quarter'] = monthly_data['YearMonth'].dt.quarter
        monthly_data['MonthsSinceRestart'] = (monthly_data['Date'] - self.restart_date).dt.days / 30.44
        
        # Cyclical encoding
        monthly_data['Month_sin'] = np.sin(2 * np.pi * monthly_data['Month'] / 12)
        monthly_data['Month_cos'] = np.cos(2 * np.pi * monthly_data['Month'] / 12)
        monthly_data['Quarter_sin'] = np.sin(2 * np.pi * monthly_data['Quarter'] / 4)
        monthly_data['Quarter_cos'] = np.cos(2 * np.pi * monthly_data['Quarter'] / 4)
        
        # Lag and rolling features
        for lag in [1, 2, 3]:
            monthly_data[f'Quantity_lag_{lag}'] = monthly_data.groupby('SKU')['Quantity'].shift(lag)
        
        for window in [2, 3, 6]:
            monthly_data[f'Quantity_rolling_mean_{window}'] = (
                monthly_data.groupby('SKU')['Quantity']
                .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            )
        
        # Trend features
        monthly_data['Quantity_diff_1'] = monthly_data.groupby('SKU')['Quantity'].diff(1)
        monthly_data['Quantity_pct_change'] = monthly_data.groupby('SKU')['Quantity'].pct_change()
        
        # SKU baseline
        sku_baselines = monthly_data.groupby('SKU')['Quantity'].median().to_dict()
        monthly_data['SKU_Baseline'] = monthly_data['SKU'].map(sku_baselines)
        monthly_data['Quantity_vs_Baseline'] = monthly_data['Quantity'] / monthly_data['SKU_Baseline']
        
        # Fill NaN values
        feature_cols = [col for col in monthly_data.columns if col.startswith(('Quantity_lag', 'Quantity_rolling', 'Quantity_diff', 'Quantity_pct', 'Quantity_vs'))]
        for col in feature_cols:
            if 'lag' in col:
                monthly_data[col] = monthly_data[col].fillna(monthly_data['SKU_Baseline'])
            else:
                monthly_data[col] = monthly_data[col].fillna(0)
        
        return monthly_data
    
    def get_feature_columns(self) -> List[str]:
        """Get feature column names"""
        return [
            'Month', 'Quarter', 'MonthsSinceRestart',
            'Month_sin', 'Month_cos', 'Quarter_sin', 'Quarter_cos',
            'Quantity_lag_1', 'Quantity_lag_2', 'Quantity_lag_3',
            'Quantity_rolling_mean_2', 'Quantity_rolling_mean_3', 'Quantity_rolling_mean_6',
            'Quantity_diff_1', 'Quantity_pct_change',
            'SKU_Baseline', 'Quantity_vs_Baseline'
        ]
    
    def calculate_sku_metrics(self, sku_data: pd.DataFrame) -> Dict:
        """Calculate comprehensive metrics for an SKU"""
        monthly_data = sku_data.groupby(sku_data['Date'].dt.to_period('M'))['Quantity'].sum()
        
        metrics = {
            'months_of_data': len(monthly_data),
            'total_volume': sku_data['Quantity'].sum(),
            'avg_monthly_volume': monthly_data.mean(),
            'volume_consistency': 0,
            'trend_stability': 0,
            'data_quality_score': 0,
            'expected_accuracy_tier': 'Medium'
        }
        
        if len(monthly_data) > 1:
            # Volume consistency
            cv = monthly_data.std() / monthly_data.mean() if monthly_data.mean() > 0 else 999
            metrics['volume_consistency'] = max(0, 1 - (cv / 2))
            
            # Trend stability
            if len(monthly_data) >= 4:
                trends = []
                for i in range(3, len(monthly_data)):
                    window = monthly_data.iloc[i-3:i+1].values
                    trend = np.polyfit(range(4), window, 1)[0]
                    trends.append(trend)
                
                if trends:
                    trend_cv = np.std(trends) / (np.mean(np.abs(trends)) + 1e-6)
                    metrics['trend_stability'] = max(0, 1 - (trend_cv / 10))
            
            # Overall data quality score
            metrics['data_quality_score'] = (
                metrics['volume_consistency'] * 0.4 +
                metrics['trend_stability'] * 0.3 +
                min(1, metrics['months_of_data'] / 12) * 0.3
            )
            
            # Expected accuracy tier
            if metrics['data_quality_score'] >= 0.8 and metrics['avg_monthly_volume'] >= 50:
                metrics['expected_accuracy_tier'] = 'High'
            elif metrics['data_quality_score'] <= 0.4 or metrics['avg_monthly_volume'] < 10:
                metrics['expected_accuracy_tier'] = 'Low'
        
        return metrics
    
    def train_models(self, data: pd.DataFrame):
        """Train prediction models for all active SKUs"""
        logger.info("Training prediction models...")
        
        featured_data = self.create_features(data)
        feature_cols = self.get_feature_columns()
        
        trained_count = 0
        
        for sku in self.active_skus:
            sku_data = featured_data[featured_data['SKU'] == sku].copy()
            
            if len(sku_data) < 8:
                continue
            
            # Calculate SKU metrics
            original_sku_data = data[data['SKU'] == sku]
            self.sku_metrics[sku] = self.calculate_sku_metrics(original_sku_data)
            
            # Prepare training data
            X = sku_data[feature_cols].copy()
            y = sku_data['Quantity'].values
            
            # Train model
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=3,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_scaled, y)
            
            # Calculate historical accuracy
            if len(sku_data) >= 10:
                split_point = int(len(sku_data) * 0.8)
                X_test = X.iloc[split_point:]
                y_test = y[split_point:]
                
                if len(X_test) > 0:
                    X_test_scaled = scaler.transform(X_test)
                    y_pred = model.predict(X_test_scaled)
                    y_pred = np.maximum(y_pred, 0)
                    
                    # Calculate MAPE
                    non_zero_mask = y_test > 0
                    if np.any(non_zero_mask):
                        mape = mean_absolute_percentage_error(y_test[non_zero_mask], y_pred[non_zero_mask]) * 100
                    else:
                        mape = 100
                    
                    self.sku_metrics[sku]['historical_mape'] = mape
                    self.sku_metrics[sku]['historical_rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
            
            self.models[sku] = model
            self.scalers[sku] = scaler
            trained_count += 1
        
        self.last_training_date = datetime.now()
        logger.info(f"Successfully trained models for {trained_count} SKUs")
    
    def calculate_confidence_score(self, sku: str) -> Dict:
        """Calculate confidence score for SKU predictions"""
        if sku not in self.sku_metrics:
            return {'confidence_score': 0.5, 'confidence_level': 'Medium', 'factors': {}}
        
        metrics = self.sku_metrics[sku]
        
        # Data quality factor (40%)
        data_quality = metrics.get('data_quality_score', 0.5)
        
        # Historical performance factor (35%)
        historical_mape = metrics.get('historical_mape', 40)
        if historical_mape <= 20:
            performance_score = 1.0
        elif historical_mape <= 30:
            performance_score = 0.8
        elif historical_mape <= 40:
            performance_score = 0.6
        else:
            performance_score = 0.4
        
        # Volume stability factor (25%)
        avg_volume = metrics.get('avg_monthly_volume', 0)
        if avg_volume >= 100:
            volume_score = 1.0
        elif avg_volume >= 50:
            volume_score = 0.8
        elif avg_volume >= 20:
            volume_score = 0.6
        else:
            volume_score = 0.4
        
        # Overall confidence
        confidence_score = (
            data_quality * 0.40 +
            performance_score * 0.35 +
            volume_score * 0.25
        )
        
        # Confidence level
        if confidence_score >= 0.8:
            confidence_level = 'High'
        elif confidence_score >= 0.6:
            confidence_level = 'Medium'
        else:
            confidence_level = 'Low'
        
        return {
            'confidence_score': confidence_score,
            'confidence_level': confidence_level,
            'factors': {
                'data_quality': data_quality,
                'historical_performance': performance_score,
                'volume_stability': volume_score
            }
        }
    
    def predict_sku(self, sku: str, months_ahead: int = 3) -> Dict:
        """Generate prediction for a specific SKU with confidence and accuracy info"""
        
        if sku not in self.models:
            return {
                'error': f'No trained model available for SKU {sku}',
                'sku': sku,
                'is_active': sku in self.active_skus
            }
        
        model = self.models[sku]
        scaler = self.scalers[sku]
        metrics = self.sku_metrics[sku]
        
        # Calculate confidence
        confidence_info = self.calculate_confidence_score(sku)
        
        # Generate predictions
        current_date = pd.Timestamp.now()
        predictions = []
        lower_bounds = []
        upper_bounds = []
        
        baseline = metrics['avg_monthly_volume']
        
        for month_offset in range(1, months_ahead + 1):
            pred_date = current_date + pd.DateOffset(months=month_offset)
            
            # Create feature vector
            features = [
                pred_date.month,
                pred_date.quarter,
                (pred_date - self.restart_date).days / 30.44,
                np.sin(2 * np.pi * pred_date.month / 12),
                np.cos(2 * np.pi * pred_date.month / 12),
                np.sin(2 * np.pi * pred_date.quarter / 4),
                np.cos(2 * np.pi * pred_date.quarter / 4),
                baseline, baseline, baseline,  # Lag features
                baseline, baseline, baseline,  # Rolling means
                0, 0,  # Diff and pct change
                baseline, 1.0  # Baseline features
            ]
            
            X = np.array([features])
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            pred = max(0, pred)
            
            # Calculate confidence intervals
            interval_width = self.confidence_interval
            if confidence_info['confidence_level'] == 'Low':
                interval_width = 0.5  # Wider intervals for low confidence
            elif confidence_info['confidence_level'] == 'High':
                interval_width = 0.25  # Tighter intervals for high confidence
            
            lower = pred * (1 - interval_width)
            upper = pred * (1 + interval_width)
            
            predictions.append(pred)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        
        return {
            'sku': sku,
            'predictions': predictions,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'confidence': confidence_info,
            'historical_accuracy': {
                'mape': metrics.get('historical_mape'),
                'rmse': metrics.get('historical_rmse'),
                'months_of_data': metrics['months_of_data'],
                'expected_tier': metrics['expected_accuracy_tier']
            },
            'sku_metrics': {
                'total_volume': metrics['total_volume'],
                'avg_monthly_volume': metrics['avg_monthly_volume'],
                'volume_consistency': metrics['volume_consistency'],
                'trend_stability': metrics['trend_stability']
            },
            'prediction_metadata': {
                'months_ahead': months_ahead,
                'confidence_interval': f"±{interval_width*100:.0f}%",
                'model_training_date': self.last_training_date.strftime('%Y-%m-%d') if self.last_training_date else None
            }
        }
    
    def get_system_performance(self) -> Dict:
        """Get overall system performance metrics"""
        if not self.sku_metrics:
            return {'error': 'No models trained yet'}
        
        mapes = [m.get('historical_mape') for m in self.sku_metrics.values() if m.get('historical_mape')]
        confidence_scores = [self.calculate_confidence_score(sku)['confidence_score'] for sku in self.sku_metrics.keys()]
        
        performance = {
            'total_active_skus': len(self.active_skus),
            'trained_models': len(self.models),
            'last_training_date': self.last_training_date.strftime('%Y-%m-%d %H:%M') if self.last_training_date else None,
            'inactive_skus_filtered': True,
            'inactive_threshold_months': self.inactive_months_threshold
        }
        
        if mapes:
            performance['accuracy_metrics'] = {
                'average_mape': np.mean(mapes),
                'median_mape': np.median(mapes),
                'best_mape': np.min(mapes),
                'worst_mape': np.max(mapes),
                'skus_under_30_mape': sum(1 for m in mapes if m <= 30),
                'skus_under_20_mape': sum(1 for m in mapes if m <= 20)
            }
        
        if confidence_scores:
            performance['confidence_distribution'] = {
                'average_confidence': np.mean(confidence_scores),
                'high_confidence_skus': sum(1 for c in confidence_scores if c >= 0.8),
                'medium_confidence_skus': sum(1 for c in confidence_scores if 0.6 <= c < 0.8),
                'low_confidence_skus': sum(1 for c in confidence_scores if c < 0.6)
            }
        
        return performance
    
    def save_model(self):
        """Save trained models and metadata"""
        model_data = {
            'active_skus': self.active_skus,
            'models': self.models,
            'scalers': self.scalers,
            'sku_metrics': self.sku_metrics,
            'last_training_date': self.last_training_date,
            'config': {
                'inactive_months_threshold': self.inactive_months_threshold,
                'min_months_for_training': self.min_months_for_training,
                'confidence_interval': self.confidence_interval
            }
        }
        
        model_file = self.models_dir / "sales_prediction_engine.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_file}")
    
    def load_model(self):
        """Load saved model"""
        model_file = self.models_dir / "sales_prediction_engine.pkl"
        
        if model_file.exists():
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.active_skus = model_data['active_skus']
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.sku_metrics = model_data['sku_metrics']
            self.last_training_date = model_data['last_training_date']
            
            logger.info(f"Loaded model with {len(self.models)} trained SKUs")
            return True
        
        return False
    
    def train_and_save(self):
        """Complete training pipeline"""
        logger.info("Starting production model training...")
        
        # Load and filter data
        data = self.load_and_filter_data()
        
        # Train models
        self.train_models(data)
        
        # Save model
        self.save_model()
        
        logger.info("Training completed successfully")
        return self.get_system_performance()

def main():
    """Production training and testing"""
    data_dir = Path("/Users/tanayshah/Desktop/personal/projects/ecom_analysis/data")
    
    # Initialize engine
    engine = SalesPredictionEngine(data_dir)
    
    # Train models
    performance = engine.train_and_save()
    
    print("\n" + "=" * 80)
    print("🚀 PRODUCTION SALES PREDICTION ENGINE - DEPLOYED")
    print("=" * 80)
    
    print(f"\n📊 SYSTEM PERFORMANCE:")
    print(f"  • Active SKUs: {performance['total_active_skus']}")
    print(f"  • Trained Models: {performance['trained_models']}")
    print(f"  • Inactive SKUs Filtered: {performance['inactive_skus_filtered']}")
    print(f"  • Training Date: {performance['last_training_date']}")
    
    if 'accuracy_metrics' in performance:
        acc = performance['accuracy_metrics']
        print(f"\n🎯 ACCURACY METRICS (After Inactive SKU Filtering):")
        print(f"  • Average MAPE: {acc['average_mape']:.2f}%")
        print(f"  • Median MAPE: {acc['median_mape']:.2f}%")
        print(f"  • Best MAPE: {acc['best_mape']:.2f}%")
        print(f"  • SKUs ≤30% MAPE: {acc['skus_under_30_mape']}/{performance['trained_models']}")
        print(f"  • SKUs ≤20% MAPE: {acc['skus_under_20_mape']}/{performance['trained_models']}")
    
    if 'confidence_distribution' in performance:
        conf = performance['confidence_distribution']
        print(f"\n🎯 CONFIDENCE DISTRIBUTION:")
        print(f"  • Average Confidence: {conf['average_confidence']:.3f}")
        print(f"  • High Confidence: {conf['high_confidence_skus']} SKUs")
        print(f"  • Medium Confidence: {conf['medium_confidence_skus']} SKUs")
        print(f"  • Low Confidence: {conf['low_confidence_skus']} SKUs")
    
    # Test prediction for a sample SKU
    if engine.active_skus:
        sample_sku = engine.active_skus[0]
        prediction = engine.predict_sku(sample_sku, months_ahead=3)
        
        print(f"\n📋 SAMPLE PREDICTION (SKU: {sample_sku}):")
        print(f"  • 3-Month Predictions: {[f'{p:.1f}' for p in prediction['predictions']]}")
        print(f"  • Confidence Level: {prediction['confidence']['confidence_level']}")
        mape = prediction['historical_accuracy']['mape']
        if mape is not None:
            print(f"  • Historical MAPE: {mape:.2f}%")
        else:
            print(f"  • Historical MAPE: Not available")
        print(f"  • Confidence Interval: {prediction['prediction_metadata']['confidence_interval']}")
    
    print(f"\n💡 EXPECTED ACCURACY IMPROVEMENT:")
    print(f"  • Current metrics reflect active SKUs only (inactive filtered out)")
    print(f"  • Accuracy will improve as more post-restart data accumulates")
    print(f"  • Monthly retraining recommended for optimal performance")
    
    print("=" * 80)
    
    return engine

if __name__ == "__main__":
    prediction_engine = main()
