#!/usr/bin/env python3
"""
Historical Pattern Cache System
Pre-calculates and caches seasonal patterns and YoY trends from 2018-2023 data.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

class HistoricalPatternCache:
    def __init__(self):
        self.cache_file = "cache/historical_patterns_cache.json"
        self.historical_data_file = "datasets/processed/historical_data_2018_nov2024.csv"
        self.patterns = {}
        
    def split_and_save_historical_data(self):
        """Split the main data file and save 2018-2023 data separately."""
        print("📊 Splitting historical data (2018-Nov 2024)...")
        
        # Load main data file
        main_data_file = None
        if os.path.exists('datasets/raw/Sales-Table 1.csv'):
            main_data_file = 'datasets/raw/Sales-Table 1.csv'
        elif os.path.exists('datasets/raw/SalesAthena.csv'):
            main_data_file = 'datasets/raw/SalesAthena.csv'
        elif os.path.exists('datasets/raw/SalesDataAthenasql (1)/Sales-Table 1.csv'):
            main_data_file = 'datasets/raw/SalesDataAthenasql (1)/Sales-Table 1.csv'
        else:
            raise FileNotFoundError("No main data file found")
        
        print(f"📁 Loading data from: {main_data_file}")
        df = pd.read_csv(main_data_file)
        df.columns = df.columns.str.strip()
        
        # Standardize column names
        if 'sku' in df.columns:
            df = df.rename(columns={'sku': 'SKU'})
        
        # Parse dates
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Create time features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Quarter'] = df['Date'].dt.quarter
        
        # Filter historical data (2018-Nov 2024)
        historical_df = df[(df['Date'].dt.year >= 2018) & 
                          ((df['Date'].dt.year < 2024) | 
                           ((df['Date'].dt.year == 2024) & (df['Date'].dt.month <= 11)))].copy()
        
        print(f"📅 Historical data: {len(historical_df):,} records from {historical_df['Date'].min().strftime('%Y-%m-%d')} to {historical_df['Date'].max().strftime('%Y-%m-%d')}")
        
        # Save historical data
        historical_df.to_csv(self.historical_data_file, index=False)
        print(f"✅ Saved historical data to: {self.historical_data_file}")
        
        return historical_df
    
    def calculate_seasonal_patterns(self, historical_df):
        """Calculate seasonal patterns from historical data."""
        print("🔍 Calculating seasonal patterns...")
        
        # Create monthly aggregation
        monthly_data = historical_df.groupby(['SKU', 'Year', 'Month']).agg({
            'Quantity': 'sum',
            'Amount': 'sum',
            'Rate': 'mean'
        }).reset_index()
        
        seasonal_patterns = {}
        yoy_patterns = {}
        
        for sku in monthly_data['SKU'].unique():
            sku_data = monthly_data[monthly_data['SKU'] == sku].copy()
            
            if len(sku_data) < 12:  # Need at least 1 year of data
                continue
            
            # Calculate seasonal factors (monthly averages)
            seasonal_factors = sku_data.groupby('Month')['Quantity'].mean()
            overall_avg = sku_data['Quantity'].mean()
            
            # Normalize seasonal factors
            if overall_avg > 0:
                seasonal_factors = seasonal_factors / overall_avg
                seasonal_patterns[sku] = seasonal_factors.to_dict()
            
            # Calculate YoY growth patterns
            years = sorted(sku_data['Year'].unique())
            if len(years) >= 2:
                yoy_growths = []
                for i in range(1, len(years)):
                    year1_data = sku_data[sku_data['Year'] == years[i-1]]
                    year2_data = sku_data[sku_data['Year'] == years[i]]
                    
                    if len(year1_data) > 0 and len(year2_data) > 0:
                        year1_total = year1_data['Quantity'].sum()
                        year2_total = year2_data['Quantity'].sum()
                        
                        if year1_total > 0:
                            yoy_growth = (year2_total - year1_total) / year1_total
                            yoy_growths.append(yoy_growth)
                
                if yoy_growths:
                    avg_yoy_growth = np.mean(yoy_growths)
                    yoy_patterns[sku] = avg_yoy_growth
        
        self.patterns = {
            'seasonal_patterns': seasonal_patterns,
            'yoy_patterns': yoy_patterns,
            'calculated_at': datetime.now().isoformat(),
            'data_period': f"{historical_df['Date'].min().strftime('%Y-%m-%d')} to {historical_df['Date'].max().strftime('%Y-%m-%d')}",
            'total_skus': len(seasonal_patterns)
        }
        
        print(f"✅ Calculated patterns for {len(seasonal_patterns)} SKUs")
        return self.patterns
    
    def save_patterns_to_cache(self):
        """Save calculated patterns to cache file."""
        print("💾 Saving patterns to cache...")
        
        with open(self.cache_file, 'w') as f:
            json.dump(self.patterns, f, indent=2)
        
        print(f"✅ Patterns cached to: {self.cache_file}")
    
    def load_patterns_from_cache(self):
        """Load patterns from cache file."""
        if not os.path.exists(self.cache_file):
            print("⚠️ No cache file found")
            return None
        
        print("📂 Loading patterns from cache...")
        
        with open(self.cache_file, 'r') as f:
            self.patterns = json.load(f)
        
        print(f"✅ Loaded patterns for {self.patterns.get('total_skus', 0)} SKUs")
        print(f"📅 Data period: {self.patterns.get('data_period', 'Unknown')}")
        print(f"🕒 Calculated at: {self.patterns.get('calculated_at', 'Unknown')}")
        
        return self.patterns
    
    def get_seasonal_factor(self, sku, month):
        """Get seasonal factor for a specific SKU and month."""
        if not self.patterns or 'seasonal_patterns' not in self.patterns:
            return 1.0
        
        seasonal_patterns = self.patterns['seasonal_patterns']
        if sku in seasonal_patterns:
            return seasonal_patterns[sku].get(str(month), 1.0)
        
        return 1.0
    
    def get_yoy_growth(self, sku):
        """Get YoY growth expectation for a specific SKU."""
        if not self.patterns or 'yoy_patterns' not in self.patterns:
            return 0.0
        
        yoy_patterns = self.patterns['yoy_patterns']
        return yoy_patterns.get(sku, 0.0)
    
    def get_available_skus(self):
        """Get list of SKUs with historical patterns."""
        if not self.patterns or 'seasonal_patterns' not in self.patterns:
            return []
        
        return list(self.patterns['seasonal_patterns'].keys())
    
    def is_cache_valid(self):
        """Check if cache exists and is valid."""
        return os.path.exists(self.cache_file) and os.path.exists(self.historical_data_file)
    
    def rebuild_cache(self):
        """Rebuild the entire cache from scratch."""
        print("🔄 Rebuilding historical pattern cache...")
        
        # Split and save historical data
        historical_df = self.split_and_save_historical_data()
        
        # Calculate patterns
        patterns = self.calculate_seasonal_patterns(historical_df)
        
        # Save to cache
        self.save_patterns_to_cache()
        
        return patterns

def main():
    """Main function to build the historical pattern cache."""
    print("="*60)
    print("🏗️ BUILDING HISTORICAL PATTERN CACHE".center(60))
    print("="*60)
    
    try:
        cache = HistoricalPatternCache()
        
        # Check if cache already exists
        if cache.is_cache_valid():
            print("✅ Cache already exists!")
            cache.load_patterns_from_cache()
            
            # Show some examples
            available_skus = cache.get_available_skus()[:5]
            print(f"\n📊 Sample SKUs with patterns: {', '.join(available_skus)}")
            
            # Show seasonal factors for first SKU
            if available_skus:
                sample_sku = available_skus[0]
                print(f"\n📅 Seasonal factors for {sample_sku}:")
                for month in range(1, 13):
                    factor = cache.get_seasonal_factor(sample_sku, month)
                    print(f"   Month {month:2d}: {factor:.2f}x")
            
            print(f"\n💡 To rebuild cache, delete {cache.cache_file} and {cache.historical_data_file}")
        else:
            # Build cache from scratch
            patterns = cache.rebuild_cache()
            
            print("\n" + "="*60)
            print("✅ CACHE BUILD COMPLETE".center(60))
            print("="*60)
            
            print(f"\n📊 Summary:")
            print(f"   • Historical data: {cache.historical_data_file}")
            print(f"   • Patterns cache: {cache.cache_file}")
            print(f"   • SKUs with patterns: {patterns['total_skus']}")
            print(f"   • Data period: {patterns['data_period']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
