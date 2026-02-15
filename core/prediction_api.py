#!/usr/bin/env python3
"""
Production API for Sales Prediction Engine
Simple interface for getting predictions, managing models, and system monitoring
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .sales_prediction_engine import SalesPredictionEngine
from .hybrid_prediction_engine import HybridPredictionEngine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredictionAPI:
    """Production API wrapper for the sales prediction engine with hybrid capabilities"""
    
    def __init__(self, data_dir: Path, use_hybrid: bool = True):
        self.data_dir = Path(data_dir)
        self.use_hybrid = use_hybrid
        
        if use_hybrid:
            self.engine = HybridPredictionEngine(data_dir)
            print("Loaded hybrid prediction engine")
        else:
            self.engine = SalesPredictionEngine(data_dir)
            if not self.engine.load_model():
                print("No existing model found. Training new model...")
                self.engine.train_and_save()
            else:
                print(f"Loaded existing model with {len(self.engine.models)} trained SKUs")
    
    def get_prediction(self, sku: str, months_ahead: int = 3) -> Dict:
        """Get prediction for a specific SKU with confidence and accuracy info"""
        if self.use_hybrid:
            return self.engine.predict_sku_hybrid(sku, months_ahead)
        else:
            return self.engine.predict_sku(sku, months_ahead)
    
    def get_active_skus(self) -> list:
        """Get list of all active SKUs that can be predicted"""
        if self.use_hybrid:
            # For hybrid engine, get from individual engine
            return self.engine.individual_engine.active_skus
        else:
            return self.engine.active_skus
    
    def get_system_performance(self) -> dict:
        """Get overall system performance metrics"""
        if self.use_hybrid:
            return self.engine.individual_engine.get_system_performance()
        else:
            return self.engine.get_system_performance()
    
    def retrain_models(self):
        """Retrain all models with latest data"""
        print("Retraining models with latest data...")
        if self.use_hybrid:
            return self.engine.individual_engine.train_and_save()
        else:
            return self.engine.train_and_save()

def main():
    """Demo usage of the prediction API"""
    data_dir = Path("/Users/tanayshah/Desktop/personal/projects/ecom_analysis/data")
    
    # Initialize API
    api = PredictionAPI(data_dir)
    
    # Get system performance
    performance = api.get_system_performance()
    print("\n" + "=" * 60)
    print("📊 PRODUCTION PREDICTION API - READY")
    print("=" * 60)
    
    print(f"\n🎯 System Status:")
    print(f"  • Active SKUs: {performance['total_active_skus']}")
    print(f"  • Trained Models: {performance['trained_models']}")
    print(f"  • Last Training: {performance['last_training_date']}")
    
    if 'accuracy_metrics' in performance:
        acc = performance['accuracy_metrics']
        print(f"\n📈 Accuracy After Inactive SKU Filtering:")
        print(f"  • Median MAPE: {acc['median_mape']:.2f}%")
        print(f"  • SKUs ≤30% MAPE: {acc['skus_under_30_mape']}/{performance['trained_models']}")
        print(f"  • SKUs ≤20% MAPE: {acc['skus_under_20_mape']}/{performance['trained_models']}")
    
    # Demo predictions for top 3 SKUs
    active_skus = api.get_active_skus()[:3]
    print(f"\n🔮 Sample Predictions:")
    
    for sku in active_skus:
        prediction = api.get_prediction(sku, months_ahead=3)
        
        if 'error' not in prediction:
            print(f"\n  📦 SKU: {sku}")
            print(f"    • Predictions: {[f'{p:.1f}' for p in prediction['predictions']]}")
            print(f"    • Confidence: {prediction['confidence']['confidence_level']} ({prediction['confidence']['confidence_score']:.3f})")
            print(f"    • Interval: {prediction['prediction_metadata']['confidence_interval']}")
            
            if prediction['historical_accuracy']['mape']:
                print(f"    • Historical MAPE: {prediction['historical_accuracy']['mape']:.2f}%")
            
            print(f"    • Data Quality: {prediction['sku_metrics']['volume_consistency']:.3f}")
    
    print(f"\n💡 Usage Instructions:")
    print(f"  • api.get_prediction('SKU_NAME', months_ahead=3)")
    print(f"  • api.get_active_skus() - list all predictable SKUs")
    print(f"  • api.retrain_models() - monthly retraining")
    print(f"  • Inactive SKUs (no sales in 3 months) are automatically filtered")
    
    print("=" * 60)
    
    return api

if __name__ == "__main__":
    prediction_api = main()
