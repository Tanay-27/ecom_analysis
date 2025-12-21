#!/usr/bin/env python3
"""
Quick script to update predictions with product names and enhanced confidence
"""

import pandas as pd
from pathlib import Path

# Set up paths
DATA_DIR = Path("/Users/tanayshah/Desktop/personal/projects/ecom_analysis/data")
RESULTS_DIR = Path("/Users/tanayshah/Desktop/personal/projects/ecom_analysis/core/analysis_results")

# Load SKU mapping
try:
    sku_file = DATA_DIR / "raw" / "sku_list.csv"
    sku_df = pd.read_csv(sku_file)
    sku_mapping = dict(zip(sku_df['sku'], sku_df['category']))
    print(f"SKU mapping loaded: {len(sku_mapping)} products")
except Exception as e:
    print(f"Could not load SKU mapping: {e}")
    sku_mapping = {}

# Load current predictions
pred_file = RESULTS_DIR / "next_month_predictions.csv"
predictions = pd.read_csv(pred_file)

# Add product names
predictions['product_name'] = predictions['SKU'].map(sku_mapping).fillna('Unknown Product')

# Enhanced confidence calculation based on growth rate and other factors
def calculate_enhanced_confidence(row):
    growth_rate = abs(row['growth_rate'])
    monthly_qty = row['predicted_monthly_quantity']
    
    # Start with base score
    score = 50
    
    # Factor 1: Growth stability (40% weight)
    if growth_rate < 0.1:  # Very stable
        score += 40
    elif growth_rate < 0.3:  # Moderate
        score += 25
    elif growth_rate < 0.5:  # High but manageable
        score += 15
    else:  # Very high growth
        score += 5
    
    # Factor 2: Volume level (30% weight)
    if monthly_qty >= 1000:  # High volume
        score += 30
    elif monthly_qty >= 300:  # Medium volume
        score += 20
    elif monthly_qty >= 100:  # Low-medium volume
        score += 15
    else:  # Very low volume
        score += 5
    
    # Factor 3: SKU maturity (30% weight) - based on SKU pattern
    sku = row['SKU']
    if any(x in sku for x in ['LRM', 'CMSM']):  # Established product lines
        score += 20
    elif len(sku) > 5:  # Complex SKUs might be newer
        score += 15
    else:
        score += 10
    
    # Convert to confidence level
    if score >= 85:
        return 'High'
    elif score >= 65:
        return 'Medium'
    else:
        return 'Low'

# Apply enhanced confidence calculation
predictions['confidence'] = predictions.apply(calculate_enhanced_confidence, axis=1)

# Save updated predictions
predictions.to_csv(pred_file, index=False)
print(f"Updated predictions saved with {len(predictions)} SKUs")

# Show confidence distribution
confidence_dist = predictions['confidence'].value_counts()
print("\nConfidence Distribution:")
for conf, count in confidence_dist.items():
    print(f"  {conf}: {count} SKUs")

# Show some examples with product names
print("\nSample predictions with product names:")
for _, row in predictions.head(5).iterrows():
    print(f"  {row['SKU']} ({row['product_name']}): {row['predicted_monthly_quantity']} units - {row['confidence']} confidence")
