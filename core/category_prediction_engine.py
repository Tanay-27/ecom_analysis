#!/usr/bin/env python3
"""
Category-Based Prediction Engine
Tests category-level predictions vs individual SKU predictions to determine optimal approach
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CategoryPredictionEngine:
    """Implements category-based prediction with SKU distribution"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.restart_date = pd.Timestamp('2025-01-01')
        self.min_months_data = 6
        
        # Storage
        self.category_models = {}
        self.category_scalers = {}
        self.sku_distribution_patterns = {}
        self.category_data = None
        self.sku_mapping = None
        
    def load_and_prepare_data(self):
        """Load sales data and SKU categories"""
        logger.info("Loading and preparing category-based data...")
        
        # Load unified sales data
        unified_file = self.processed_dir / "unified_sales_data.csv"
        df = pd.read_csv(unified_file, low_memory=False)
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Load SKU categories
        sku_list_file = self.data_dir / "raw" / "sku_list.csv"
        sku_categories = pd.read_csv(sku_list_file)
        
        # Merge sales data with categories
        df_with_categories = df.merge(sku_categories[['sku', 'category']], 
                                    left_on='SKU', right_on='sku', how='left')
        
        # Filter out "NOT SELLING" and missing categories
        df_with_categories = df_with_categories[
            (df_with_categories['category'].notna()) & 
            (df_with_categories['category'] != 'NOT SELLING')
        ]
        
        # Focus on post-restart data
        restart_data = df_with_categories[df_with_categories['Date'] >= self.restart_date].copy()
        
        logger.info(f"Loaded {len(restart_data):,} records with category mapping")
        logger.info(f"Categories: {restart_data['category'].nunique()}")
        logger.info(f"SKUs: {restart_data['SKU'].nunique()}")
        
        self.category_data = restart_data
        self.sku_mapping = sku_categories.set_index('sku')['category'].to_dict()
        
        return restart_data
    
    def create_category_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features at category level"""
        logger.info("Creating category-level features...")
        
        # Monthly aggregation by category
        data['YearMonth'] = data['Date'].dt.to_period('M')
        
        monthly_category = data.groupby(['category', 'YearMonth']).agg({
            'Quantity': 'sum',
            'Amount': 'sum',
            'SKU': 'nunique'  # Number of active SKUs in category
        }).reset_index()
        
        # Create feature matrix for each category
        category_features = {}
        
        for category in monthly_category['category'].unique():
            cat_data = monthly_category[monthly_category['category'] == category].copy()
            cat_data = cat_data.sort_values('YearMonth')
            
            if len(cat_data) >= self.min_months_data:
                # Create time-based features
                cat_data['month'] = cat_data['YearMonth'].dt.month
                cat_data['quarter'] = cat_data['YearMonth'].dt.quarter
                
                # Lag features
                for lag in [1, 2, 3]:
                    cat_data[f'quantity_lag_{lag}'] = cat_data['Quantity'].shift(lag)
                    cat_data[f'amount_lag_{lag}'] = cat_data['Amount'].shift(lag)
                
                # Rolling features
                for window in [3, 6]:
                    cat_data[f'quantity_rolling_mean_{window}'] = cat_data['Quantity'].rolling(window).mean()
                    cat_data[f'quantity_rolling_std_{window}'] = cat_data['Quantity'].rolling(window).std()
                
                # Trend features
                cat_data['quantity_trend'] = cat_data['Quantity'].pct_change(periods=3)
                cat_data['amount_trend'] = cat_data['Amount'].pct_change(periods=3)
                
                # Seasonal features
                cat_data['month_sin'] = np.sin(2 * np.pi * cat_data['month'] / 12)
                cat_data['month_cos'] = np.cos(2 * np.pi * cat_data['month'] / 12)
                cat_data['quarter_sin'] = np.sin(2 * np.pi * cat_data['quarter'] / 4)
                cat_data['quarter_cos'] = np.cos(2 * np.pi * cat_data['quarter'] / 4)
                
                # SKU diversity features
                cat_data['sku_diversity'] = cat_data['SKU']
                cat_data['avg_sku_volume'] = cat_data['Quantity'] / cat_data['SKU']
                
                category_features[category] = cat_data.fillna(0)
        
        logger.info(f"Created features for {len(category_features)} categories")
        return category_features
    
    def analyze_sku_distribution_patterns(self, data: pd.DataFrame):
        """Analyze how volume distributes among SKUs within each category"""
        logger.info("Analyzing SKU distribution patterns within categories...")
        
        # Monthly SKU distribution within categories
        monthly_sku_category = data.groupby(['category', 'SKU', 'YearMonth'])['Quantity'].sum().reset_index()
        
        distribution_patterns = {}
        
        for category in monthly_sku_category['category'].unique():
            cat_data = monthly_sku_category[monthly_sku_category['category'] == category]
            
            # Calculate monthly distribution patterns
            monthly_distributions = {}
            for month in cat_data['YearMonth'].unique():
                month_data = cat_data[cat_data['YearMonth'] == month]
                total_volume = month_data['Quantity'].sum()
                
                if total_volume > 0:
                    sku_shares = month_data.set_index('SKU')['Quantity'] / total_volume
                    monthly_distributions[month] = sku_shares.to_dict()
            
            if monthly_distributions:
                # Calculate average distribution pattern
                all_skus = set()
                for month_dist in monthly_distributions.values():
                    all_skus.update(month_dist.keys())
                
                avg_distribution = {}
                for sku in all_skus:
                    shares = [monthly_distributions[month].get(sku, 0) 
                             for month in monthly_distributions.keys()]
                    avg_distribution[sku] = np.mean(shares)
                
                # Normalize to ensure sum = 1
                total_share = sum(avg_distribution.values())
                if total_share > 0:
                    avg_distribution = {sku: share/total_share 
                                     for sku, share in avg_distribution.items()}
                
                distribution_patterns[category] = {
                    'avg_distribution': avg_distribution,
                    'sku_count': len(all_skus),
                    'stability': self._calculate_distribution_stability(monthly_distributions),
                    'top_sku': max(avg_distribution.items(), key=lambda x: x[1])[0] if avg_distribution else None
                }
        
        self.sku_distribution_patterns = distribution_patterns
        logger.info(f"Analyzed distribution patterns for {len(distribution_patterns)} categories")
        
        return distribution_patterns
    
    def _calculate_distribution_stability(self, monthly_distributions: Dict) -> float:
        """Calculate how stable SKU distribution is within a category"""
        if len(monthly_distributions) < 2:
            return 0.0
        
        # Calculate coefficient of variation for each SKU's share
        all_skus = set()
        for month_dist in monthly_distributions.values():
            all_skus.update(month_dist.keys())
        
        sku_cvs = []
        for sku in all_skus:
            shares = [monthly_distributions[month].get(sku, 0) 
                     for month in monthly_distributions.keys()]
            if np.mean(shares) > 0:
                cv = np.std(shares) / np.mean(shares)
                sku_cvs.append(cv)
        
        # Return average stability (lower CV = more stable)
        return 1 / (1 + np.mean(sku_cvs)) if sku_cvs else 0.0
    
    def train_category_models(self, category_features: Dict):
        """Train prediction models for each category"""
        logger.info("Training category-level models...")
        
        feature_columns = [
            'month', 'quarter', 'quantity_lag_1', 'quantity_lag_2', 'quantity_lag_3',
            'amount_lag_1', 'amount_lag_2', 'amount_lag_3',
            'quantity_rolling_mean_3', 'quantity_rolling_mean_6',
            'quantity_rolling_std_3', 'quantity_rolling_std_6',
            'quantity_trend', 'amount_trend',
            'month_sin', 'month_cos', 'quarter_sin', 'quarter_cos',
            'sku_diversity', 'avg_sku_volume'
        ]
        
        successful_models = 0
        
        for category, cat_data in category_features.items():
            try:
                # Prepare training data
                train_data = cat_data.dropna(subset=feature_columns + ['Quantity'])
                
                if len(train_data) >= 6:  # Need minimum data for training
                    X = train_data[feature_columns]
                    y = train_data['Quantity']
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    
                    # Train model
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=42
                    )
                    model.fit(X_scaled, y)
                    
                    # Store model and scaler
                    self.category_models[category] = model
                    self.category_scalers[category] = scaler
                    successful_models += 1
                    
                    logger.info(f"Trained model for {category}: {len(train_data)} data points")
                
            except Exception as e:
                logger.warning(f"Failed to train model for {category}: {e}")
        
        logger.info(f"Successfully trained {successful_models} category models")
        return successful_models
    
    def predict_category_volume(self, category: str, months_ahead: int = 3) -> List[float]:
        """Predict total volume for a category"""
        if category not in self.category_models:
            return [0.0] * months_ahead
        
        model = self.category_models[category]
        scaler = self.category_scalers[category]
        
        # Get latest data for the category
        cat_data = self.category_data[self.category_data['category'] == category]
        monthly_cat = cat_data.groupby(cat_data['Date'].dt.to_period('M')).agg({
            'Quantity': 'sum',
            'Amount': 'sum',
            'SKU': 'nunique'
        }).reset_index()
        monthly_cat = monthly_cat.sort_values('Date')
        
        predictions = []
        
        # Create features for prediction
        for month_offset in range(1, months_ahead + 1):
            try:
                # Get the most recent data point
                latest_data = monthly_cat.iloc[-1].copy()
                
                # Project forward
                future_date = latest_data['Date'] + month_offset
                
                # Create feature vector (simplified)
                features = {
                    'month': future_date.month,
                    'quarter': future_date.quarter,
                    'quantity_lag_1': latest_data['Quantity'],
                    'quantity_lag_2': monthly_cat.iloc[-2]['Quantity'] if len(monthly_cat) > 1 else latest_data['Quantity'],
                    'quantity_lag_3': monthly_cat.iloc[-3]['Quantity'] if len(monthly_cat) > 2 else latest_data['Quantity'],
                    'amount_lag_1': latest_data['Amount'],
                    'amount_lag_2': monthly_cat.iloc[-2]['Amount'] if len(monthly_cat) > 1 else latest_data['Amount'],
                    'amount_lag_3': monthly_cat.iloc[-3]['Amount'] if len(monthly_cat) > 2 else latest_data['Amount'],
                    'quantity_rolling_mean_3': monthly_cat['Quantity'].tail(3).mean(),
                    'quantity_rolling_mean_6': monthly_cat['Quantity'].tail(6).mean(),
                    'quantity_rolling_std_3': monthly_cat['Quantity'].tail(3).std(),
                    'quantity_rolling_std_6': monthly_cat['Quantity'].tail(6).std(),
                    'quantity_trend': 0.0,  # Simplified
                    'amount_trend': 0.0,    # Simplified
                    'month_sin': np.sin(2 * np.pi * future_date.month / 12),
                    'month_cos': np.cos(2 * np.pi * future_date.month / 12),
                    'quarter_sin': np.sin(2 * np.pi * future_date.quarter / 4),
                    'quarter_cos': np.cos(2 * np.pi * future_date.quarter / 4),
                    'sku_diversity': latest_data['SKU'],
                    'avg_sku_volume': latest_data['Quantity'] / latest_data['SKU'] if latest_data['SKU'] > 0 else 0
                }
                
                # Fill NaN values
                for key, value in features.items():
                    if pd.isna(value):
                        features[key] = 0.0
                
                # Convert to array and scale
                feature_vector = np.array([list(features.values())])
                feature_vector_scaled = scaler.transform(feature_vector)
                
                # Predict
                pred = model.predict(feature_vector_scaled)[0]
                predictions.append(max(0, pred))  # Ensure non-negative
                
            except Exception as e:
                logger.warning(f"Prediction failed for {category} month {month_offset}: {e}")
                predictions.append(0.0)
        
        return predictions
    
    def distribute_category_prediction_to_skus(self, category: str, total_prediction: float) -> Dict[str, float]:
        """Distribute category prediction to individual SKUs"""
        if category not in self.sku_distribution_patterns:
            return {}
        
        distribution = self.sku_distribution_patterns[category]['avg_distribution']
        
        # Apply distribution to total prediction
        sku_predictions = {}
        for sku, share in distribution.items():
            sku_predictions[sku] = total_prediction * share
        
        return sku_predictions
    
    def get_category_based_sku_prediction(self, sku: str, months_ahead: int = 3) -> Dict:
        """Get SKU prediction using category-based approach"""
        if sku not in self.sku_mapping:
            return {'error': f'SKU {sku} not found in category mapping'}
        
        category = self.sku_mapping[sku]
        
        if category == 'NOT SELLING':
            return {'error': f'SKU {sku} is marked as NOT SELLING'}
        
        # Get category prediction
        category_predictions = self.predict_category_volume(category, months_ahead)
        
        # Distribute to SKUs
        sku_predictions = []
        for total_pred in category_predictions:
            sku_distribution = self.distribute_category_prediction_to_skus(category, total_pred)
            sku_pred = sku_distribution.get(sku, 0.0)
            sku_predictions.append(sku_pred)
        
        # Get distribution info
        distribution_info = self.sku_distribution_patterns.get(category, {})
        
        return {
            'sku': sku,
            'category': category,
            'predictions': sku_predictions,
            'category_total_predictions': category_predictions,
            'sku_share_in_category': distribution_info.get('avg_distribution', {}).get(sku, 0.0),
            'category_sku_count': distribution_info.get('sku_count', 0),
            'distribution_stability': distribution_info.get('stability', 0.0),
            'approach': 'category_based',
            'months_ahead': months_ahead
        }
    
    def compare_with_individual_predictions(self, test_skus: List[str]) -> Dict:
        """Compare category-based vs individual SKU predictions"""
        logger.info("Comparing category-based vs individual SKU predictions...")
        
        # Load individual prediction engine for comparison
        from sales_prediction_engine import SalesPredictionEngine
        individual_engine = SalesPredictionEngine(self.data_dir)
        individual_engine.load_model()
        
        comparison_results = {
            'category_based': {},
            'individual_based': {},
            'comparison_metrics': {}
        }
        
        for sku in test_skus:
            try:
                # Category-based prediction
                cat_result = self.get_category_based_sku_prediction(sku, months_ahead=3)
                if 'error' not in cat_result:
                    comparison_results['category_based'][sku] = cat_result
                
                # Individual-based prediction
                ind_result = individual_engine.predict_sku(sku, months_ahead=3)
                if 'error' not in ind_result:
                    comparison_results['individual_based'][sku] = ind_result
                
            except Exception as e:
                logger.warning(f"Comparison failed for {sku}: {e}")
        
        # Calculate comparison metrics
        category_predictions = []
        individual_predictions = []
        
        for sku in test_skus:
            if (sku in comparison_results['category_based'] and 
                sku in comparison_results['individual_based']):
                
                cat_pred = comparison_results['category_based'][sku]['predictions'][0]
                ind_pred = comparison_results['individual_based'][sku]['predictions'][0]
                
                category_predictions.append(cat_pred)
                individual_predictions.append(ind_pred)
        
        if category_predictions and individual_predictions:
            # Calculate correlation and differences
            correlation = np.corrcoef(category_predictions, individual_predictions)[0, 1]
            mean_diff = np.mean(np.array(category_predictions) - np.array(individual_predictions))
            
            comparison_results['comparison_metrics'] = {
                'correlation': correlation,
                'mean_difference': mean_diff,
                'category_avg': np.mean(category_predictions),
                'individual_avg': np.mean(individual_predictions),
                'skus_compared': len(category_predictions)
            }
        
        return comparison_results
    
    def save_models(self):
        """Save trained category models"""
        model_file = self.models_dir / "category_prediction_engine.pkl"
        
        model_data = {
            'category_models': self.category_models,
            'category_scalers': self.category_scalers,
            'sku_distribution_patterns': self.sku_distribution_patterns,
            'sku_mapping': self.sku_mapping,
            'training_date': datetime.now().isoformat()
        }
        
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved category models to {model_file}")
    
    def load_models(self):
        """Load saved category models"""
        model_file = self.models_dir / "category_prediction_engine.pkl"
        
        if model_file.exists():
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.category_models = model_data['category_models']
            self.category_scalers = model_data['category_scalers']
            self.sku_distribution_patterns = model_data['sku_distribution_patterns']
            self.sku_mapping = model_data['sku_mapping']
            
            logger.info(f"Loaded category models: {len(self.category_models)} categories")
            return True
        
        return False

def main():
    """Test category-based prediction approach"""
    data_dir = Path("/Users/tanayshah/Desktop/personal/projects/ecom_analysis/data")
    
    # Initialize category engine
    cat_engine = CategoryPredictionEngine(data_dir)
    
    print("🔬 CATEGORY-BASED PREDICTION ANALYSIS")
    print("=" * 60)
    
    # Load and prepare data
    data = cat_engine.load_and_prepare_data()
    
    # Create category features
    category_features = cat_engine.create_category_features(data)
    
    # Analyze SKU distribution patterns
    distribution_patterns = cat_engine.analyze_sku_distribution_patterns(data)
    
    # Train category models
    models_trained = cat_engine.train_category_models(category_features)
    
    print(f"\n📊 Training Results:")
    print(f"  • Categories with models: {models_trained}")
    print(f"  • Distribution patterns analyzed: {len(distribution_patterns)}")
    
    # Test predictions for sample SKUs
    test_skus = ['CMSM06', 'LRM02', 'LAF01', 'ACIB6DS', 'HY128']
    
    print(f"\n🔮 Sample Category-Based Predictions:")
    for sku in test_skus:
        result = cat_engine.get_category_based_sku_prediction(sku, months_ahead=3)
        
        if 'error' not in result:
            category = result['category']
            pred = result['predictions'][0]
            share = result['sku_share_in_category']
            stability = result['distribution_stability']
            
            print(f"\n  📦 {sku} ({category}):")
            print(f"    • Prediction: {pred:.1f} units")
            print(f"    • Category share: {share:.1%}")
            print(f"    • Distribution stability: {stability:.3f}")
        else:
            print(f"\n  ❌ {sku}: {result['error']}")
    
    # Compare with individual predictions
    print(f"\n⚖️  Comparing Approaches:")
    comparison = cat_engine.compare_with_individual_predictions(test_skus)
    
    if 'comparison_metrics' in comparison and comparison['comparison_metrics']:
        metrics = comparison['comparison_metrics']
        print(f"  • Correlation: {metrics['correlation']:.3f}")
        print(f"  • Category avg: {metrics['category_avg']:.1f} units")
        print(f"  • Individual avg: {metrics['individual_avg']:.1f} units")
        print(f"  • Mean difference: {metrics['mean_difference']:.1f} units")
        print(f"  • SKUs compared: {metrics['skus_compared']}")
    
    # Save models
    cat_engine.save_models()
    
    print(f"\n💡 Analysis Complete!")
    print(f"  • Models saved for future use")
    print(f"  • Category approach shows {'high' if comparison.get('comparison_metrics', {}).get('correlation', 0) > 0.7 else 'moderate'} correlation with individual predictions")
    
    return cat_engine

if __name__ == "__main__":
    category_engine = main()
