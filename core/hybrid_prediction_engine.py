#!/usr/bin/env python3
"""
Hybrid Prediction Engine
Combines individual SKU predictions with category-based predictions using intelligent decision logic
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pickle

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridPredictionEngine:
    """Intelligent hybrid engine that chooses between individual and category-based predictions"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        
        # Load individual prediction engine for comparison
        from .sales_prediction_engine import SalesPredictionEngine
        from .category_prediction_engine import CategoryPredictionEngine
        
        self.individual_engine = SalesPredictionEngine(data_dir)
        self.category_engine = CategoryPredictionEngine(data_dir)
        
        # Decision criteria thresholds
        self.min_months_individual = 6
        self.dominant_sku_threshold = 0.30  # 30% category share
        self.stability_threshold = 0.50
        self.confidence_boost_factor = 0.15  # Boost confidence for hybrid decisions
        
        # Load engines
        self._load_engines()
        
    def _load_engines(self):
        """Load both individual and category engines"""
        logger.info("Loading hybrid prediction engines...")
        
        # Load individual engine
        individual_loaded = self.individual_engine.load_model()
        if not individual_loaded:
            logger.warning("Individual engine not loaded - training may be required")
        
        # Load or train category engine
        category_loaded = self.category_engine.load_models()
        if not category_loaded:
            logger.info("Category engine not found - training...")
            self.category_engine.load_and_prepare_data()
            category_features = self.category_engine.create_category_features(self.category_engine.category_data)
            self.category_engine.analyze_sku_distribution_patterns(self.category_engine.category_data)
            self.category_engine.train_category_models(category_features)
            self.category_engine.save_models()
        
        logger.info("Hybrid engines loaded successfully")
    
    def _analyze_sku_characteristics(self, sku: str) -> Dict:
        """Analyze SKU characteristics to determine best prediction approach"""
        characteristics = {
            'has_individual_model': False,
            'months_of_data': 0,
            'category': None,
            'category_share': 0.0,
            'category_stability': 0.0,
            'volume_level': 'low',
            'data_quality_score': 0.0
        }
        
        # Check individual model availability - try prediction to see if model exists
        try:
            test_result = self.individual_engine.predict_sku(sku, months_ahead=1)
            if 'error' not in test_result:
                characteristics['has_individual_model'] = True
                
                # Estimate data quality from prediction metadata
                if 'historical_accuracy' in test_result:
                    mape = test_result['historical_accuracy'].get('mape', 100)
                    if mape is not None and mape < 50:
                        characteristics['data_quality_score'] = max(0.1, 1 - (mape / 100))
                        characteristics['months_of_data'] = test_result['historical_accuracy'].get('months_of_data', 6)
                    else:
                        characteristics['data_quality_score'] = 0.3
                        characteristics['months_of_data'] = 6  # Assume minimum
        except Exception:
            characteristics['has_individual_model'] = False
        
        # Get category information
        if (hasattr(self.category_engine, 'sku_mapping') and 
            self.category_engine.sku_mapping is not None and 
            sku in self.category_engine.sku_mapping):
            
            category = self.category_engine.sku_mapping[sku]
            characteristics['category'] = category
            
            # Get category share and stability
            if (hasattr(self.category_engine, 'sku_distribution_patterns') and
                self.category_engine.sku_distribution_patterns is not None):
                
                pattern_info = self.category_engine.sku_distribution_patterns.get(category, {})
                avg_distribution = pattern_info.get('avg_distribution', {})
                characteristics['category_share'] = avg_distribution.get(sku, 0.0)
                characteristics['category_stability'] = pattern_info.get('stability', 0.0)
        
        # Determine volume level (simplified)
        if characteristics['category_share'] > 0.5:
            characteristics['volume_level'] = 'high'
        elif characteristics['category_share'] > 0.2:
            characteristics['volume_level'] = 'medium'
        
        return characteristics
    
    def _decide_prediction_approach(self, sku: str) -> Tuple[str, str, float]:
        """
        Decide which prediction approach to use
        Returns: (approach, reason, confidence_adjustment)
        """
        characteristics = self._analyze_sku_characteristics(sku)
        
        # Decision logic based on analysis findings
        
        # Rule 1: No individual model available -> use category
        if not characteristics['has_individual_model']:
            if characteristics['category'] and characteristics['category'] != 'NOT SELLING':
                return 'category', 'No individual model available', 0.0
            else:
                return 'fallback', 'No model available for SKU or category', -0.3
        
        # Rule 2: Insufficient individual data -> prefer category if stable
        if characteristics['months_of_data'] < self.min_months_individual:
            if characteristics['category_stability'] > self.stability_threshold:
                return 'category', f'Limited individual data ({characteristics["months_of_data"]} months)', -0.1
            else:
                return 'individual', 'Individual model despite limited data', -0.2
        
        # Rule 3: Dominant SKU in unstable category -> use individual
        if (characteristics['category_share'] > self.dominant_sku_threshold and 
            characteristics['category_stability'] < self.stability_threshold):
            return 'individual', f'Dominant SKU ({characteristics["category_share"]:.1%}) in unstable category', 0.1
        
        # Rule 4: Non-dominant SKU in stable category -> consider category
        if (characteristics['category_share'] <= self.dominant_sku_threshold and 
            characteristics['category_stability'] > self.stability_threshold):
            return 'hybrid', f'Non-dominant SKU in stable category', 0.05
        
        # Rule 5: High quality individual data -> use individual
        if characteristics['data_quality_score'] > 0.8:
            return 'individual', 'High quality individual data', 0.1
        
        # Default: Use individual if available
        return 'individual', 'Default individual approach', 0.0
    
    def _blend_predictions(self, individual_result: Dict, category_result: Dict, 
                          blend_weight: float = 0.7) -> Dict:
        """Blend individual and category predictions"""
        
        if 'error' in individual_result or 'error' in category_result:
            # Return the working prediction
            return individual_result if 'error' not in individual_result else category_result
        
        # Blend predictions
        ind_preds = individual_result['predictions']
        cat_preds = category_result['predictions']
        
        blended_preds = []
        for i in range(min(len(ind_preds), len(cat_preds))):
            blended = (blend_weight * ind_preds[i] + (1 - blend_weight) * cat_preds[i])
            blended_preds.append(blended)
        
        # Create blended result
        blended_result = individual_result.copy()
        blended_result['predictions'] = blended_preds
        
        # Adjust confidence intervals
        if 'lower_bounds' in individual_result and 'upper_bounds' in individual_result:
            blended_result['lower_bounds'] = [
                blend_weight * individual_result['lower_bounds'][i] + 
                (1 - blend_weight) * blended_preds[i] * 0.75  # Conservative lower bound
                for i in range(len(blended_preds))
            ]
            blended_result['upper_bounds'] = [
                blend_weight * individual_result['upper_bounds'][i] + 
                (1 - blend_weight) * blended_preds[i] * 1.25  # Conservative upper bound
                for i in range(len(blended_preds))
            ]
        
        # Update metadata
        blended_result['prediction_metadata']['approach'] = 'hybrid_blend'
        blended_result['prediction_metadata']['blend_weight'] = blend_weight
        
        return blended_result
    
    def predict_sku_hybrid(self, sku: str, months_ahead: int = 3) -> Dict:
        """Get hybrid prediction for a SKU"""
        
        # Decide approach
        approach, reason, confidence_adjustment = self._decide_prediction_approach(sku)
        
        logger.info(f"SKU {sku}: Using {approach} approach - {reason}")
        
        try:
            if approach == 'individual':
                result = self.individual_engine.predict_sku(sku, months_ahead)
                result['hybrid_info'] = {
                    'approach_used': 'individual',
                    'decision_reason': reason,
                    'confidence_adjustment': confidence_adjustment
                }
                
                # Apply confidence adjustment
                if 'confidence' in result and 'confidence_score' in result['confidence']:
                    original_score = result['confidence']['confidence_score']
                    adjusted_score = max(0.1, min(1.0, original_score + confidence_adjustment))
                    result['confidence']['confidence_score'] = adjusted_score
                    
                    # Update confidence level
                    if adjusted_score >= 0.8:
                        result['confidence']['confidence_level'] = 'High'
                    elif adjusted_score >= 0.6:
                        result['confidence']['confidence_level'] = 'Medium'
                    else:
                        result['confidence']['confidence_level'] = 'Low'
                
                return result
                
            elif approach == 'category':
                result = self.category_engine.get_category_based_sku_prediction(sku, months_ahead)
                
                if 'error' not in result:
                    # Convert to individual engine format
                    converted_result = {
                        'sku': sku,
                        'predictions': result['predictions'],
                        'confidence': {
                            'confidence_score': max(0.1, 0.6 + confidence_adjustment),
                            'confidence_level': 'Medium' if confidence_adjustment >= 0 else 'Low',
                            'factors': {
                                'category_stability': result.get('distribution_stability', 0.5),
                                'category_share': result.get('sku_share_in_category', 0.0),
                                'data_source': 'category_based'
                            }
                        },
                        'historical_accuracy': {
                            'approach': 'category_based',
                            'category': result.get('category', 'Unknown')
                        },
                        'prediction_metadata': {
                            'approach': 'category_based',
                            'months_ahead': months_ahead,
                            'category': result.get('category', 'Unknown')
                        },
                        'hybrid_info': {
                            'approach_used': 'category',
                            'decision_reason': reason,
                            'confidence_adjustment': confidence_adjustment
                        }
                    }
                    return converted_result
                else:
                    return result
                    
            elif approach == 'hybrid':
                # Get both predictions and blend
                individual_result = self.individual_engine.predict_sku(sku, months_ahead)
                category_result = self.category_engine.get_category_based_sku_prediction(sku, months_ahead)
                
                # Determine blend weight based on characteristics
                characteristics = self._analyze_sku_characteristics(sku)
                blend_weight = 0.7  # Default favor individual
                
                if characteristics['category_stability'] > 0.7:
                    blend_weight = 0.5  # Equal weight for very stable categories
                elif characteristics['data_quality_score'] < 0.5:
                    blend_weight = 0.4  # Favor category for poor individual data
                
                # Convert category result to individual format first
                if 'error' not in category_result:
                    category_converted = {
                        'predictions': category_result['predictions'],
                        'lower_bounds': [p * 0.75 for p in category_result['predictions']],
                        'upper_bounds': [p * 1.25 for p in category_result['predictions']]
                    }
                    
                    if 'error' not in individual_result:
                        blended_result = self._blend_predictions(individual_result, category_converted, blend_weight)
                        blended_result['hybrid_info'] = {
                            'approach_used': 'hybrid_blend',
                            'decision_reason': reason,
                            'blend_weight': blend_weight,
                            'confidence_adjustment': confidence_adjustment
                        }
                        return blended_result
                    else:
                        # Fall back to category if individual fails
                        return self.predict_sku_hybrid(sku, months_ahead)  # Recursive call will choose category
                else:
                    # Fall back to individual if category fails
                    return individual_result
                    
            else:  # fallback
                return {
                    'error': f'No suitable prediction approach available for SKU {sku}',
                    'hybrid_info': {
                        'approach_used': 'fallback',
                        'decision_reason': reason
                    }
                }
                
        except Exception as e:
            logger.error(f"Hybrid prediction failed for {sku}: {e}")
            return {
                'error': f'Hybrid prediction failed: {str(e)}',
                'hybrid_info': {
                    'approach_used': 'error',
                    'decision_reason': f'Exception: {str(e)}'
                }
            }
    
    def get_approach_statistics(self, test_skus: List[str]) -> Dict:
        """Analyze which approach would be used for a list of SKUs"""
        approach_stats = {
            'individual': 0,
            'category': 0,
            'hybrid': 0,
            'fallback': 0,
            'decisions': {}
        }
        
        for sku in test_skus:
            approach, reason, confidence_adj = self._decide_prediction_approach(sku)
            approach_stats[approach] += 1
            approach_stats['decisions'][sku] = {
                'approach': approach,
                'reason': reason,
                'confidence_adjustment': confidence_adj
            }
        
        return approach_stats
    
    def batch_predict_hybrid(self, skus: List[str], months_ahead: int = 3) -> Dict[str, Dict]:
        """Get hybrid predictions for multiple SKUs"""
        results = {}
        
        for sku in skus:
            results[sku] = self.predict_sku_hybrid(sku, months_ahead)
        
        return results
    
    def save_hybrid_config(self):
        """Save hybrid engine configuration"""
        config_file = self.models_dir / "hybrid_prediction_config.pkl"
        
        config = {
            'min_months_individual': self.min_months_individual,
            'dominant_sku_threshold': self.dominant_sku_threshold,
            'stability_threshold': self.stability_threshold,
            'confidence_boost_factor': self.confidence_boost_factor,
            'created_date': datetime.now().isoformat()
        }
        
        with open(config_file, 'wb') as f:
            pickle.dump(config, f)
        
        logger.info(f"Saved hybrid configuration to {config_file}")

def main():
    """Test hybrid prediction engine"""
    data_dir = Path("/Users/tanayshah/Desktop/personal/projects/ecom_analysis/data")
    
    print("🔀 HYBRID PREDICTION ENGINE - TESTING")
    print("=" * 60)
    
    # Initialize hybrid engine
    hybrid_engine = HybridPredictionEngine(data_dir)
    
    # Test SKUs
    test_skus = ['CMSM06', 'LRM02', 'LAF01', 'ACIB6DS', 'HY128', 'LSG01', 'CKS1', 'LPC01']
    
    # Analyze approach decisions
    print("\n📊 Approach Decision Analysis:")
    approach_stats = hybrid_engine.get_approach_statistics(test_skus)
    
    print(f"  • Individual: {approach_stats['individual']} SKUs")
    print(f"  • Category: {approach_stats['category']} SKUs") 
    print(f"  • Hybrid: {approach_stats['hybrid']} SKUs")
    print(f"  • Fallback: {approach_stats['fallback']} SKUs")
    
    print(f"\n🔍 Decision Details:")
    for sku, decision in approach_stats['decisions'].items():
        print(f"  {sku}: {decision['approach']} - {decision['reason']}")
    
    # Test hybrid predictions
    print(f"\n🔮 Hybrid Predictions:")
    
    sample_skus = ['CMSM06', 'LRM02', 'ACIB6DS']
    for sku in sample_skus:
        result = hybrid_engine.predict_sku_hybrid(sku, months_ahead=3)
        
        if 'error' not in result:
            approach = result['hybrid_info']['approach_used']
            reason = result['hybrid_info']['decision_reason']
            pred = result['predictions'][0]
            confidence = result.get('confidence', {}).get('confidence_level', 'Unknown')
            
            print(f"\n  📦 {sku}:")
            print(f"    • Approach: {approach}")
            print(f"    • Reason: {reason}")
            print(f"    • Prediction: {pred:.1f} units")
            print(f"    • Confidence: {confidence}")
        else:
            print(f"\n  ❌ {sku}: {result['error']}")
    
    # Save configuration
    hybrid_engine.save_hybrid_config()
    
    print(f"\n✅ Hybrid engine testing complete!")
    print(f"  • Decision logic implemented")
    print(f"  • Approach selection working")
    print(f"  • Configuration saved")
    
    return hybrid_engine

if __name__ == "__main__":
    hybrid_engine = main()
