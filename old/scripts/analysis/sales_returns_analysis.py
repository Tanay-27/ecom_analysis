#!/usr/bin/env python3
"""
Comprehensive Sales & Returns Analysis

This script analyzes both sales and returns data to provide complete
business intelligence including return rates, net revenue, and risk analysis.
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

class SalesReturnsAnalyzer:
    """
    Comprehensive analyzer for both sales and returns data.
    """
    
    def __init__(self):
        self.sales_df = None
        self.returns_df = None
        self.combined_df = None
        
    def load_data(self, load_returns=False, verbose=False):
        """Load sales data and optionally returns data."""
        try:
            if verbose:
                print("📊 Loading Sales Data...")
            
            # Load sales data - prioritize Sales-Table 1.csv
            if os.path.exists('datasets/raw/Sales-Table 1.csv'):
                self.sales_df = pd.read_csv('datasets/raw/Sales-Table 1.csv')
            elif os.path.exists('datasets/raw/SalesAthena.csv'):
                self.sales_df = pd.read_csv('datasets/raw/SalesAthena.csv')
            elif os.path.exists('datasets/processed/uploaded_sales.csv'):
                self.sales_df = pd.read_csv('datasets/processed/uploaded_sales.csv')
            else:
                raise FileNotFoundError("No sales data file found")
                
            self.sales_df.columns = self.sales_df.columns.str.strip()
            self.sales_df['Date'] = pd.to_datetime(self.sales_df['Date'], errors='coerce')
            self.sales_df = self.sales_df.dropna(subset=['Date'])
            self.sales_df['Type'] = 'Sales'
            
            if verbose:
                print(f"✅ Sales data: {len(self.sales_df):,} records")
            
            # Only load returns data if explicitly requested
            if load_returns:
                if verbose:
                    print("📊 Loading Returns Data...")
                
                # Only load returns if explicitly requested - no automatic loading
                if load_returns and os.path.exists('datasets/processed/uploaded_returns.csv'):
                    self.returns_df = pd.read_csv('datasets/processed/uploaded_returns.csv')
                    self.returns_df.columns = self.returns_df.columns.str.strip()
                    self.returns_df['Date'] = pd.to_datetime(self.returns_df['Date'], errors='coerce')
                    self.returns_df = self.returns_df.dropna(subset=['Date'])
                    self.returns_df['Type'] = 'Returns'
                    
                    # Standardize column names
                    self.returns_df = self.returns_df.rename(columns={'sku': 'SKU'})
                    
                    if verbose:
                        print(f"✅ Returns data: {len(self.returns_df):,} records")
                    
                    # Combine data only if returns data is loaded
                    self.combined_df = pd.concat([self.sales_df, self.returns_df], ignore_index=True)
                    self.combined_df = self.combined_df.sort_values(['Date', 'SKU'])
                else:
                    if verbose:
                        print("⚠️ No returns data file found")
                    self.returns_df = None
                    self.combined_df = self.sales_df
            else:
                self.returns_df = None
                self.combined_df = self.sales_df
            
            if verbose:
                print(f"✅ Combined data: {len(self.combined_df):,} records")
            return True
            
        except Exception as e:
            if verbose:
                print(f"❌ Error loading data: {e}")
            return False
        
    def analyze_return_rates(self, verbose=False):
        """Analyze return rates by various dimensions."""
        if self.sales_df is None or self.returns_df is None:
            return None
            
        if verbose:
            print("\n📊 RETURN RATE ANALYSIS:")
            print("-" * 50)
        
        # Calculate return rates by SKU
        sales_by_sku = self.sales_df.groupby('SKU')['Quantity'].sum()
        returns_by_sku = self.returns_df.groupby('SKU')['Quantity'].sum().abs()  # Make positive
        
        # Calculate return rate
        return_rates = (returns_by_sku / sales_by_sku * 100).fillna(0)
        return_rates = return_rates.sort_values(ascending=False)
        
        overall_return_rate = (returns_by_sku.sum() / sales_by_sku.sum() * 100) if sales_by_sku.sum() > 0 else 0
        
        if verbose:
            print(f"Overall Return Rate: {overall_return_rate:.2f}%")
            print(f"Total Sales: {sales_by_sku.sum():,} units")
            print(f"Total Returns: {returns_by_sku.sum():,} units")
            
            print(f"\nTop 10 SKUs by Return Rate:")
            for i, (sku, rate) in enumerate(return_rates.head(10).items(), 1):
                sales_qty = sales_by_sku.get(sku, 0)
                return_qty = returns_by_sku.get(sku, 0)
                print(f"{i:2d}. {sku}: {rate:.1f}% ({return_qty:,} returns / {sales_qty:,} sales)")
        
        # Return rates by state
        sales_by_state = self.sales_df.groupby('Stateto')['Quantity'].sum()
        returns_by_state = self.returns_df.groupby('Stateto')['Quantity'].sum().abs()
        state_return_rates = (returns_by_state / sales_by_state * 100).fillna(0)
        state_return_rates = state_return_rates.sort_values(ascending=False)
        
        if verbose:
            print(f"\nTop 10 States by Return Rate:")
            for i, (state, rate) in enumerate(state_return_rates.head(10).items(), 1):
                sales_qty = sales_by_state.get(state, 0)
                return_qty = returns_by_state.get(state, 0)
                print(f"{i:2d}. {state}: {rate:.1f}% ({return_qty:,} returns / {sales_qty:,} sales)")
        
        # Calculate returns revenue
        returns_revenue = abs(self.returns_df['Amount'].sum())
        
        return {
            'overall_return_rate': float(overall_return_rate),
            'returns_revenue': float(returns_revenue),
            'total_returns': int(returns_by_sku.sum()),
            'unique_skus_returned': int(len(returns_by_sku)),
            'sku_return_rates': return_rates.to_dict(),
            'state_return_rates': state_return_rates.to_dict(),
            'total_sales': int(sales_by_sku.sum())
        }
    
    def analyze_returns_by_sku(self):
        """Analyze returns by SKU for API."""
        if self.sales_df is None or self.returns_df is None:
            return None
            
        # Calculate return rates by SKU
        sales_by_sku = self.sales_df.groupby('SKU')['Quantity'].sum()
        returns_by_sku = self.returns_df.groupby('SKU')['Quantity'].sum().abs()
        
        # Calculate return rate
        return_rates = (returns_by_sku / sales_by_sku * 100).fillna(0)
        return_rates = return_rates.sort_values(ascending=False)
        
        # Get top returning SKUs
        top_returning_skus = []
        for sku, rate in return_rates.head(20).items():
            sales_qty = sales_by_sku.get(sku, 0)
            return_qty = returns_by_sku.get(sku, 0)
            top_returning_skus.append({
                'sku': sku,
                'return_rate': rate,
                'sales_quantity': sales_qty,
                'returns_quantity': return_qty
            })
        
        return {
            'top_returning_skus': top_returning_skus,
            'overall_return_rate': float(returns_by_sku.sum() / sales_by_sku.sum() * 100),
            'total_skus_with_returns': int(len(returns_by_sku))
        }
    
    def analyze_returns_by_state(self):
        """Analyze returns by state for API."""
        if self.sales_df is None or self.returns_df is None:
            return None
            
        # Calculate return rates by state
        sales_by_state = self.sales_df.groupby('Stateto')['Quantity'].sum()
        returns_by_state = self.returns_df.groupby('Stateto')['Quantity'].sum().abs()
        state_return_rates = (returns_by_state / sales_by_state * 100).fillna(0)
        state_return_rates = state_return_rates.sort_values(ascending=False)
        
        # Get top returning states
        top_returning_states = []
        for state, rate in state_return_rates.head(20).items():
            sales_qty = sales_by_state.get(state, 0)
            return_qty = returns_by_state.get(state, 0)
            top_returning_states.append({
                'state': state,
                'return_rate': rate,
                'sales_quantity': sales_qty,
                'returns_quantity': return_qty
            })
        
        return {
            'top_returning_states': top_returning_states,
            'overall_return_rate': float(returns_by_state.sum() / sales_by_state.sum() * 100),
            'total_states_with_returns': int(len(returns_by_state))
        }
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in returns for API."""
        if self.sales_df is None or self.returns_df is None:
            return None
            
        # Monthly patterns
        monthly_sales = self.sales_df.groupby([self.sales_df['Date'].dt.year, self.sales_df['Date'].dt.month])['Quantity'].sum()
        monthly_returns = self.returns_df.groupby([self.returns_df['Date'].dt.year, self.returns_df['Date'].dt.month])['Quantity'].sum().abs()
        monthly_return_rates = (monthly_returns / monthly_sales * 100).fillna(0)
        
        # Day of week patterns
        dow_sales = self.sales_df.groupby(self.sales_df['Date'].dt.dayofweek)['Quantity'].sum()
        dow_returns = self.returns_df.groupby(self.returns_df['Date'].dt.dayofweek)['Quantity'].sum().abs()
        dow_return_rates = (dow_returns / dow_sales * 100).fillna(0)
        
        dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_patterns = []
        for i, (dow, rate) in enumerate(dow_return_rates.items()):
            dow_patterns.append({
                'day': dow_names[i],
                'return_rate': rate,
                'sales_quantity': dow_sales.get(dow, 0),
                'returns_quantity': dow_returns.get(dow, 0)
            })
        
        return {
            'monthly_return_rates': monthly_return_rates.to_dict(),
            'dow_patterns': dow_patterns,
            'avg_monthly_return_rate': float(monthly_return_rates.mean()),
            'max_monthly_return_rate': float(monthly_return_rates.max()),
            'min_monthly_return_rate': float(monthly_return_rates.min())
        }
    
    def identify_high_risk_returns(self):
        """Identify high-risk returns for API."""
        if self.sales_df is None or self.returns_df is None:
            return None
            
        # Calculate return rates by SKU
        sales_by_sku = self.sales_df.groupby('SKU')['Quantity'].sum()
        returns_by_sku = self.returns_df.groupby('SKU')['Quantity'].sum().abs()
        return_rates = (returns_by_sku / sales_by_sku * 100).fillna(0)
        
        # High-risk SKUs (>50% return rate)
        high_risk_skus = []
        for sku, rate in return_rates[return_rates > 50].sort_values(ascending=False).items():
            high_risk_skus.append({
                'sku': sku,
                'return_rate': rate,
                'returns': returns_by_sku.get(sku, 0)
            })
        
        # Calculate return rates by state
        sales_by_state = self.sales_df.groupby('Stateto')['Quantity'].sum()
        returns_by_state = self.returns_df.groupby('Stateto')['Quantity'].sum().abs()
        state_return_rates = (returns_by_state / sales_by_state * 100).fillna(0)
        
        # High-risk states (>40% return rate)
        high_risk_states = []
        for state, rate in state_return_rates[state_return_rates > 40].sort_values(ascending=False).items():
            high_risk_states.append({
                'state': state,
                'return_rate': rate,
                'returns': returns_by_state.get(state, 0)
            })
        
        # Calculate financial impact
        high_risk_revenue_loss = 0
        for sku in [item['sku'] for item in high_risk_skus]:
            sku_returns = self.returns_df[self.returns_df['SKU'] == sku]
            high_risk_revenue_loss += abs(sku_returns['Amount'].sum())
        
        return {
            'high_risk_skus': high_risk_skus,
            'high_risk_states': high_risk_states,
            'high_risk_revenue_loss': float(high_risk_revenue_loss),
            'affected_skus': int(len(high_risk_skus)),
            'affected_states': int(len(high_risk_states))
        }

def main():
    """Main function for comprehensive sales and returns analysis."""
    print("="*80)
    print("📊 COMPREHENSIVE SALES & RETURNS ANALYSIS")
    print("="*80)
    print("Analyzing both sales and returns data for complete business intelligence")
    print("="*80)
    
    # Initialize analyzer
    analyzer = SalesReturnsAnalyzer()
    
    # Load data
    if not analyzer.load_data():
        print("❌ Failed to load data")
        return None
    
    # Analyze return rates
    return_analysis = analyzer.analyze_return_rates()
    
    print(f"\n✅ Comprehensive Sales & Returns Analysis Completed!")
    print("="*80)
    
    return return_analysis

if __name__ == "__main__":
    main()