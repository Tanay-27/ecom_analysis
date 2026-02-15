#!/usr/bin/env python3
"""
Multi-Dimensional Sales Predictor
Provides predictions by Channel (Platform) and Geography (State/City) using hybrid approach
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiDimensionalPredictor:
    """Predicts sales across channels and geographical dimensions"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.restart_date = pd.Timestamp('2025-01-01')
        
        # Dimension thresholds
        self.min_months_data = 4
        self.min_volume_threshold = 50
        self.min_godown_volume = 30  # Minimum volume for godown analysis
        
        # Storage for models and patterns
        self.channel_patterns = {}
        self.geo_patterns = {}
        self.godown_patterns = {}
        self.sku_channel_models = {}
        self.sku_geo_models = {}
        
        # Godown mapping (Channel-City combinations as fulfillment centers)
        self.godown_mapping = {}
        
    def load_and_analyze_data(self) -> pd.DataFrame:
        """Load data and analyze multi-dimensional patterns"""
        logger.info("Loading and analyzing multi-dimensional data...")
        
        # Load unified data
        unified_file = self.processed_dir / "unified_sales_data.csv"
        df = pd.read_csv(unified_file, low_memory=False)
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Focus on post-restart data
        restart_data = df[df['Date'] >= self.restart_date].copy()
        restart_data = restart_data[(restart_data['Quantity'] > 0) & (restart_data['Amount'] > 0)]
        
        logger.info(f"Post-restart data: {len(restart_data):,} records")
        logger.info(f"Channels: {restart_data['Platform'].nunique()}")
        logger.info(f"States: {restart_data['State'].nunique()}")
        logger.info(f"SKUs: {restart_data['SKU'].nunique()}")
        
        return restart_data
    
    def analyze_channel_patterns(self, data: pd.DataFrame) -> Dict:
        """Analyze channel distribution patterns and growth trends"""
        logger.info("Analyzing channel patterns...")
        
        # Monthly channel analysis
        data['YearMonth'] = data['Date'].dt.to_period('M')
        
        # Channel volume patterns
        channel_monthly = data.groupby(['YearMonth', 'Platform'])['Quantity'].sum().unstack(fill_value=0)
        
        # Channel growth trends
        channel_trends = {}
        for channel in channel_monthly.columns:
            monthly_data = channel_monthly[channel]
            if len(monthly_data) >= 6:  # Need sufficient data for trend
                # Calculate growth rate
                non_zero_data = monthly_data[monthly_data > 0]
                if len(non_zero_data) >= 3:
                    growth_rate = (non_zero_data.iloc[-1] / non_zero_data.iloc[0]) ** (1/len(non_zero_data)) - 1
                    channel_trends[channel] = {
                        'monthly_avg': monthly_data.mean(),
                        'growth_rate': growth_rate,
                        'volatility': monthly_data.std() / monthly_data.mean() if monthly_data.mean() > 0 else 999,
                        'active_months': len(non_zero_data),
                        'total_volume': monthly_data.sum()
                    }
        
        # SKU-Channel distribution patterns
        sku_channel_dist = data.groupby(['SKU', 'Platform'])['Quantity'].sum().unstack(fill_value=0)
        
        # Calculate channel share for each SKU
        sku_channel_shares = sku_channel_dist.div(sku_channel_dist.sum(axis=1), axis=0).fillna(0)
        
        self.channel_patterns = {
            'trends': channel_trends,
            'monthly_data': channel_monthly,
            'sku_shares': sku_channel_shares,
            'total_distribution': data.groupby('Platform')['Quantity'].sum()
        }
        
        return self.channel_patterns
    
    def analyze_geographical_patterns(self, data: pd.DataFrame) -> Dict:
        """Analyze geographical distribution and growth patterns"""
        logger.info("Analyzing geographical patterns...")
        
        # State-wise monthly analysis
        data['YearMonth'] = data['Date'].dt.to_period('M')
        
        # Top states by volume (focus on significant markets)
        state_volumes = data.groupby('State')['Quantity'].sum().sort_values(ascending=False)
        top_states = state_volumes[state_volumes >= self.min_volume_threshold].index.tolist()
        
        logger.info(f"Analyzing {len(top_states)} significant states")
        
        # State growth trends
        state_monthly = data.groupby(['YearMonth', 'State'])['Quantity'].sum().unstack(fill_value=0)
        
        geo_trends = {}
        for state in top_states:
            if state in state_monthly.columns:
                monthly_data = state_monthly[state]
                non_zero_data = monthly_data[monthly_data > 0]
                
                if len(non_zero_data) >= 3:
                    growth_rate = (non_zero_data.iloc[-1] / non_zero_data.iloc[0]) ** (1/len(non_zero_data)) - 1
                    geo_trends[state] = {
                        'monthly_avg': monthly_data.mean(),
                        'growth_rate': growth_rate,
                        'volatility': monthly_data.std() / monthly_data.mean() if monthly_data.mean() > 0 else 999,
                        'active_months': len(non_zero_data),
                        'total_volume': monthly_data.sum()
                    }
        
        # SKU-State distribution patterns
        sku_state_data = data[data['State'].isin(top_states)]
        sku_state_dist = sku_state_data.groupby(['SKU', 'State'])['Quantity'].sum().unstack(fill_value=0)
        
        # Calculate geographical share for each SKU
        sku_geo_shares = sku_state_dist.div(sku_state_dist.sum(axis=1), axis=0).fillna(0)
        
        self.geo_patterns = {
            'trends': geo_trends,
            'monthly_data': state_monthly,
            'sku_shares': sku_geo_shares,
            'top_states': top_states,
            'total_distribution': data.groupby('State')['Quantity'].sum()
        }
        
        return self.geo_patterns
    
    def analyze_godown_patterns(self, data: pd.DataFrame) -> Dict:
        """Analyze godown/fulfillment center patterns using Channel-City combinations"""
        logger.info("Analyzing godown/fulfillment center patterns...")
        
        # Create godown identifier as Channel-City combination
        data['Godown_ID'] = data['Platform'] + '_' + data['City']
        
        # Identify significant godowns (fulfillment centers)
        godown_volumes = data.groupby('Godown_ID')['Quantity'].sum().sort_values(ascending=False)
        significant_godowns = godown_volumes[godown_volumes >= self.min_godown_volume].index.tolist()
        
        logger.info(f"Analyzing {len(significant_godowns)} significant godowns/fulfillment centers")
        
        # Monthly godown analysis
        data['YearMonth'] = data['Date'].dt.to_period('M')
        godown_monthly = data.groupby(['YearMonth', 'Godown_ID'])['Quantity'].sum().unstack(fill_value=0)
        
        # Godown trends and capacity patterns
        godown_trends = {}
        for godown in significant_godowns:
            if godown in godown_monthly.columns:
                monthly_data = godown_monthly[godown]
                non_zero_data = monthly_data[monthly_data > 0]
                
                if len(non_zero_data) >= 3:
                    # Extract channel and city
                    channel, city = godown.split('_', 1)
                    
                    growth_rate = (non_zero_data.iloc[-1] / non_zero_data.iloc[0]) ** (1/len(non_zero_data)) - 1
                    godown_trends[godown] = {
                        'channel': channel,
                        'city': city,
                        'monthly_avg': monthly_data.mean(),
                        'growth_rate': growth_rate,
                        'volatility': monthly_data.std() / monthly_data.mean() if monthly_data.mean() > 0 else 999,
                        'active_months': len(non_zero_data),
                        'total_volume': monthly_data.sum(),
                        'capacity_utilization': monthly_data.max() / monthly_data.mean() if monthly_data.mean() > 0 else 1
                    }
        
        # SKU-Godown distribution patterns
        sku_godown_data = data[data['Godown_ID'].isin(significant_godowns)]
        sku_godown_dist = sku_godown_data.groupby(['SKU', 'Godown_ID'])['Quantity'].sum().unstack(fill_value=0)
        
        # Calculate godown share for each SKU
        sku_godown_shares = sku_godown_dist.div(sku_godown_dist.sum(axis=1), axis=0).fillna(0)
        
        # Channel-State-Godown mapping for inventory allocation
        channel_state_godown = data.groupby(['Platform', 'State', 'Godown_ID'])['Quantity'].sum().reset_index()
        
        self.godown_patterns = {
            'trends': godown_trends,
            'monthly_data': godown_monthly,
            'sku_shares': sku_godown_shares,
            'significant_godowns': significant_godowns,
            'total_distribution': data.groupby('Godown_ID')['Quantity'].sum(),
            'channel_state_mapping': channel_state_godown
        }
        
        # Create godown mapping for easy lookup
        self.godown_mapping = {}
        for godown in significant_godowns:
            channel, city = godown.split('_', 1)
            if channel not in self.godown_mapping:
                self.godown_mapping[channel] = {}
            
            # Get state for this city-channel combination
            state_data = data[(data['Platform'] == channel) & (data['City'] == city)]['State'].mode()
            state = state_data.iloc[0] if len(state_data) > 0 else 'Unknown'
            
            self.godown_mapping[channel][city] = {
                'godown_id': godown,
                'state': state,
                'volume': godown_volumes[godown],
                'serves_states': data[data['Godown_ID'] == godown]['State'].unique().tolist()
            }
        
        return self.godown_patterns
    
    def determine_prediction_approach(self, sku: str, dimension: str) -> str:
        """Determine best approach: model-based vs pattern-based"""
        
        if dimension == 'channel':
            patterns = self.channel_patterns
            shares = patterns['sku_shares']
        else:  # geography
            patterns = self.geo_patterns
            shares = patterns['sku_shares']
        
        if sku not in shares.index:
            return 'fallback'
        
        sku_data = shares.loc[sku]
        active_dimensions = (sku_data > 0).sum()
        max_share = sku_data.max()
        
        # Decision logic
        if active_dimensions >= 3 and max_share < 0.8:
            return 'model_based'  # Multi-dimensional, use ML model
        elif active_dimensions >= 2:
            return 'pattern_based'  # Use historical patterns + growth
        else:
            return 'single_dimension'  # Dominant single channel/state
    
    def predict_channel_breakdown(self, sku: str, total_prediction: float, months_ahead: int = 3) -> Dict:
        """Predict channel-wise breakdown of total SKU prediction"""
        
        approach = self.determine_prediction_approach(sku, 'channel')
        
        if approach == 'fallback' or sku not in self.channel_patterns['sku_shares'].index:
            # Use overall channel distribution
            total_dist = self.channel_patterns['total_distribution']
            channel_shares = total_dist / total_dist.sum()
            
            return {
                'approach': 'fallback_distribution',
                'predictions': {channel: total_prediction * share for channel, share in channel_shares.items()},
                'confidence': 'Low',
                'method': 'Overall channel distribution applied'
            }
        
        sku_shares = self.channel_patterns['sku_shares'].loc[sku]
        active_channels = sku_shares[sku_shares > 0.01].index.tolist()  # >1% share
        
        if approach == 'single_dimension':
            # Dominant channel approach
            dominant_channel = sku_shares.idxmax()
            predictions = {channel: 0 for channel in sku_shares.index}
            predictions[dominant_channel] = total_prediction
            
            return {
                'approach': 'single_dominant',
                'predictions': predictions,
                'confidence': 'High',
                'method': f'Dominant channel: {dominant_channel}'
            }
        
        elif approach == 'pattern_based':
            # Historical pattern + growth adjustment
            base_shares = sku_shares.copy()
            
            # Apply growth trends
            adjusted_shares = base_shares.copy()
            for channel in active_channels:
                if channel in self.channel_patterns['trends']:
                    growth_rate = self.channel_patterns['trends'][channel]['growth_rate']
                    # Apply growth for future months
                    growth_factor = (1 + growth_rate) ** (months_ahead / 12)
                    adjusted_shares[channel] *= growth_factor
            
            # Normalize to sum to 1
            adjusted_shares = adjusted_shares / adjusted_shares.sum()
            
            predictions = {channel: total_prediction * share for channel, share in adjusted_shares.items()}
            
            return {
                'approach': 'pattern_with_growth',
                'predictions': predictions,
                'confidence': 'Medium',
                'method': 'Historical patterns + growth trends'
            }
        
        else:  # model_based
            # For complex multi-channel SKUs, use simplified model
            # (In practice, you might train a separate model here)
            
            # Use pattern-based as fallback for now
            base_shares = sku_shares / sku_shares.sum()
            predictions = {channel: total_prediction * share for channel, share in base_shares.items()}
            
            return {
                'approach': 'model_based_simplified',
                'predictions': predictions,
                'confidence': 'Medium',
                'method': 'Multi-channel model (simplified)'
            }
    
    def predict_geographical_breakdown(self, sku: str, total_prediction: float, months_ahead: int = 3) -> Dict:
        """Predict state-wise breakdown of total SKU prediction"""
        
        approach = self.determine_prediction_approach(sku, 'geography')
        
        if approach == 'fallback' or sku not in self.geo_patterns['sku_shares'].index:
            # Use top 10 states distribution
            total_dist = self.geo_patterns['total_distribution']
            top_10_states = total_dist.nlargest(10)
            geo_shares = top_10_states / top_10_states.sum()
            
            predictions = {state: total_prediction * share for state, share in geo_shares.items()}
            predictions['Others'] = 0  # Remaining states
            
            return {
                'approach': 'fallback_top_states',
                'predictions': predictions,
                'confidence': 'Low',
                'method': 'Top 10 states distribution applied'
            }
        
        sku_shares = self.geo_patterns['sku_shares'].loc[sku]
        active_states = sku_shares[sku_shares > 0.01].index.tolist()  # >1% share
        
        if approach == 'single_dimension':
            # Dominant state approach
            dominant_state = sku_shares.idxmax()
            predictions = {state: 0 for state in self.geo_patterns['top_states']}
            predictions[dominant_state] = total_prediction * 0.9  # 90% to dominant
            predictions['Others'] = total_prediction * 0.1  # 10% to others
            
            return {
                'approach': 'single_dominant_geo',
                'predictions': predictions,
                'confidence': 'High',
                'method': f'Dominant state: {dominant_state}'
            }
        
        elif approach == 'pattern_based':
            # Historical pattern + regional growth
            base_shares = sku_shares.copy()
            
            # Apply geographical growth trends
            adjusted_shares = base_shares.copy()
            for state in active_states:
                if state in self.geo_patterns['trends']:
                    growth_rate = self.geo_patterns['trends'][state]['growth_rate']
                    growth_factor = (1 + growth_rate) ** (months_ahead / 12)
                    adjusted_shares[state] *= growth_factor
            
            # Normalize
            total_share = adjusted_shares.sum()
            if total_share > 0:
                adjusted_shares = adjusted_shares / total_share
            
            # Focus on significant states
            significant_states = adjusted_shares[adjusted_shares > 0.01].head(15)
            others_share = 1 - significant_states.sum()
            
            predictions = {state: total_prediction * share for state, share in significant_states.items()}
            if others_share > 0:
                predictions['Others'] = total_prediction * others_share
            
            return {
                'approach': 'geo_pattern_with_growth',
                'predictions': predictions,
                'confidence': 'Medium',
                'method': 'Regional patterns + growth trends'
            }
        
        else:  # model_based
            # Simplified geographical model
            base_shares = sku_shares / sku_shares.sum() if sku_shares.sum() > 0 else sku_shares
            
            # Focus on top states
            top_shares = base_shares.nlargest(10)
            others_share = 1 - top_shares.sum()
            
            predictions = {state: total_prediction * share for state, share in top_shares.items()}
            if others_share > 0:
                predictions['Others'] = total_prediction * others_share
            
            return {
                'approach': 'geo_model_simplified',
                'predictions': predictions,
                'confidence': 'Medium',
                'method': 'Multi-state model (simplified)'
            }
    
    def predict_godown_breakdown(self, sku: str, total_prediction: float, months_ahead: int = 3) -> Dict:
        """Predict godown/fulfillment center breakdown for inventory allocation"""
        
        if not self.godown_patterns or sku not in self.godown_patterns['sku_shares'].index:
            # Use top godowns by volume as fallback
            total_dist = self.godown_patterns.get('total_distribution', {})
            if total_dist.empty:
                return {
                    'approach': 'no_godown_data',
                    'predictions': {},
                    'confidence': 'Low',
                    'method': 'No godown data available'
                }
            
            top_godowns = total_dist.nlargest(10)
            godown_shares = top_godowns / top_godowns.sum()
            
            predictions = {}
            for godown_id, share in godown_shares.items():
                channel, city = godown_id.split('_', 1)
                predictions[f"{channel}_{city}"] = total_prediction * share
            
            return {
                'approach': 'fallback_top_godowns',
                'predictions': predictions,
                'confidence': 'Low',
                'method': 'Top godowns distribution applied'
            }
        
        sku_shares = self.godown_patterns['sku_shares'].loc[sku]
        active_godowns = sku_shares[sku_shares > 0.01].index.tolist()  # >1% share
        
        # Historical pattern + godown capacity/growth
        base_shares = sku_shares.copy()
        
        # Apply godown-specific growth and capacity factors
        adjusted_shares = base_shares.copy()
        for godown in active_godowns:
            if godown in self.godown_patterns['trends']:
                trend = self.godown_patterns['trends'][godown]
                growth_rate = trend['growth_rate']
                capacity_factor = 1 / trend['capacity_utilization']  # Lower utilization = more capacity
                
                # Apply growth and capacity adjustment
                growth_factor = (1 + growth_rate) ** (months_ahead / 12)
                adjusted_shares[godown] *= growth_factor * capacity_factor
        
        # Normalize to sum to 1
        total_share = adjusted_shares.sum()
        if total_share > 0:
            adjusted_shares = adjusted_shares / total_share
        
        # Focus on significant godowns
        significant_godowns = adjusted_shares[adjusted_shares > 0.01].head(10)
        
        predictions = {}
        for godown_id, share in significant_godowns.items():
            channel, city = godown_id.split('_', 1)
            predictions[f"{channel}_{city}"] = total_prediction * share
        
        return {
            'approach': 'godown_pattern_with_capacity',
            'predictions': predictions,
            'confidence': 'Medium',
            'method': 'Godown patterns + growth + capacity utilization'
        }
    
    def get_inventory_allocation_matrix(self, sku: str, total_prediction: float, months_ahead: int = 3) -> Dict:
        """Get complete inventory allocation matrix: Channel x State x Godown"""
        
        # Get all dimensional breakdowns
        channel_breakdown = self.predict_channel_breakdown(sku, total_prediction, months_ahead)
        geo_breakdown = self.predict_geographical_breakdown(sku, total_prediction, months_ahead)
        godown_breakdown = self.predict_godown_breakdown(sku, total_prediction, months_ahead)
        
        # Create inventory allocation recommendations
        allocation_matrix = {}
        
        # Get top combinations for practical allocation
        top_channels = sorted(channel_breakdown['predictions'].items(), key=lambda x: x[1], reverse=True)[:3]
        top_states = sorted(geo_breakdown['predictions'].items(), key=lambda x: x[1], reverse=True)[:5]
        top_godowns = sorted(godown_breakdown['predictions'].items(), key=lambda x: x[1], reverse=True)[:8]
        
        # Create Channel-State-Godown allocation matrix
        for channel, channel_vol in top_channels:
            for state, state_vol in top_states:
                if state != 'Others':
                    # Find relevant godowns for this channel-state combination
                    relevant_godowns = []
                    for godown_key, godown_vol in top_godowns:
                        godown_channel, godown_city = godown_key.split('_', 1)
                        
                        # Check if godown serves this channel and state
                        if (godown_channel == channel and 
                            godown_key in self.godown_patterns.get('trends', {}) and
                            state in self.godown_patterns['trends'][godown_key].get('serves_states', [])):
                            
                            relevant_godowns.append((godown_key, godown_vol, godown_city))
                    
                    if relevant_godowns:
                        # Calculate allocation for this channel-state combination
                        channel_share = channel_vol / total_prediction if total_prediction > 0 else 0
                        state_share = state_vol / total_prediction if total_prediction > 0 else 0
                        combined_vol = total_prediction * channel_share * state_share
                        
                        if combined_vol > 1:  # Only include significant allocations
                            allocation_key = f"{channel}_{state}"
                            allocation_matrix[allocation_key] = {
                                'total_allocation': combined_vol,
                                'godowns': []
                            }
                            
                            # Distribute across relevant godowns
                            total_godown_capacity = sum(gv for _, gv, _ in relevant_godowns)
                            for godown_key, godown_vol, city in relevant_godowns:
                                if total_godown_capacity > 0:
                                    godown_allocation = combined_vol * (godown_vol / total_godown_capacity)
                                    allocation_matrix[allocation_key]['godowns'].append({
                                        'godown_id': godown_key,
                                        'city': city,
                                        'allocation': godown_allocation,
                                        'percentage': (godown_allocation / combined_vol * 100) if combined_vol > 0 else 0
                                    })
        
        return {
            'sku': sku,
            'total_prediction': total_prediction,
            'channel_breakdown': channel_breakdown,
            'geographical_breakdown': geo_breakdown,
            'godown_breakdown': godown_breakdown,
            'inventory_allocation_matrix': allocation_matrix,
            'godown_mapping': self.godown_mapping,
            'metadata': {
                'months_ahead': months_ahead,
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'allocation_method': 'Channel x State x Godown optimization'
            }
        }
    
    def get_multi_dimensional_prediction(self, sku: str, total_prediction: float, months_ahead: int = 3) -> Dict:
        """Get complete multi-dimensional breakdown including godown allocation"""
        return self.get_inventory_allocation_matrix(sku, total_prediction, months_ahead)
    
    def analyze_and_setup(self):
        """Analyze data and setup prediction patterns"""
        logger.info("Setting up multi-dimensional prediction system...")
        
        # Load data
        data = self.load_and_analyze_data()
        
        # Analyze patterns
        self.analyze_channel_patterns(data)
        self.analyze_geographical_patterns(data)
        self.analyze_godown_patterns(data)
        
        logger.info("Multi-dimensional analysis complete")
        
        return {
            'channels_analyzed': len(self.channel_patterns['trends']),
            'states_analyzed': len(self.geo_patterns['trends']),
            'godowns_analyzed': len(self.godown_patterns['trends']),
            'top_channels': list(self.channel_patterns['total_distribution'].nlargest(3).index),
            'top_states': list(self.geo_patterns['total_distribution'].nlargest(5).index),
            'top_godowns': list(self.godown_patterns['total_distribution'].nlargest(5).index)
        }

def main():
    """Demo multi-dimensional predictions"""
    data_dir = Path("/Users/tanayshah/Desktop/personal/projects/ecom_analysis/data")
    
    # Initialize predictor
    predictor = MultiDimensionalPredictor(data_dir)
    
    # Setup analysis
    setup_results = predictor.analyze_and_setup()
    
    print("\n" + "=" * 80)
    print("🌍 MULTI-DIMENSIONAL SALES PREDICTOR - READY")
    print("=" * 80)
    
    print(f"\n📊 Analysis Results:")
    print(f"  • Channels Analyzed: {setup_results['channels_analyzed']}")
    print(f"  • States Analyzed: {setup_results['states_analyzed']}")
    print(f"  • Godowns Analyzed: {setup_results['godowns_analyzed']}")
    print(f"  • Top Channels: {', '.join(setup_results['top_channels'])}")
    print(f"  • Top States: {', '.join(setup_results['top_states'])}")
    print(f"  • Top Godowns: {', '.join([g.replace('_', ' - ') for g in setup_results['top_godowns']])}")
    
    # Demo prediction for a sample SKU
    sample_sku = 'CMSM06'  # Use a known active SKU
    total_prediction = 150  # Sample total prediction
    
    result = predictor.get_multi_dimensional_prediction(sample_sku, total_prediction, months_ahead=3)
    
    print(f"\n🔮 Complete Inventory Allocation (SKU: {sample_sku}):")
    print(f"  Total Prediction: {total_prediction}")
    
    print(f"\n📺 Channel Breakdown ({result['channel_breakdown']['approach']}):")
    for channel, volume in result['channel_breakdown']['predictions'].items():
        if volume > 0:
            percentage = (volume / total_prediction * 100) if total_prediction > 0 else 0
            print(f"    • {channel}: {volume:.1f} units ({percentage:.1f}%)")
    
    print(f"\n🗺️  Geographical Breakdown ({result['geographical_breakdown']['approach']}):")
    for state, volume in result['geographical_breakdown']['predictions'].items():
        if volume > 0:
            percentage = (volume / total_prediction * 100) if total_prediction > 0 else 0
            print(f"    • {state}: {volume:.1f} units ({percentage:.1f}%)")
    
    print(f"\n� Godown/Fulfillment Center Breakdown ({result['godown_breakdown']['approach']}):")
    for godown, volume in result['godown_breakdown']['predictions'].items():
        if volume > 0:
            percentage = (volume / total_prediction * 100) if total_prediction > 0 else 0
            channel, city = godown.split('_', 1)
            print(f"    • {channel} in {city}: {volume:.1f} units ({percentage:.1f}%)")
    
    print(f"\n🎯 Inventory Allocation Matrix (Channel-State-Godown):")
    for allocation_key, allocation_data in result['inventory_allocation_matrix'].items():
        channel, state = allocation_key.split('_', 1)
        total_alloc = allocation_data['total_allocation']
        percentage = (total_alloc / total_prediction * 100) if total_prediction > 0 else 0
        print(f"\n  📦 {channel} → {state}: {total_alloc:.1f} units ({percentage:.1f}%)")
        
        for godown_info in allocation_data['godowns']:
            godown_alloc = godown_info['allocation']
            godown_pct = godown_info['percentage']
            city = godown_info['city']
            print(f"    └── Stock at {city}: {godown_alloc:.1f} units ({godown_pct:.1f}% of {state})")
    
    print(f"\n💡 Inventory Recommendations:")
    channel_conf = result['channel_breakdown']['confidence']
    geo_conf = result['geographical_breakdown']['confidence']
    godown_conf = result['godown_breakdown']['confidence']
    
    if all(conf in ['High', 'Medium'] for conf in [channel_conf, geo_conf, godown_conf]):
        print(f"  ✅ Good confidence across dimensions - use for detailed inventory planning")
        print(f"  📋 Focus stocking on top godown-state combinations shown above")
    else:
        print(f"  ⚠️  Mixed confidence - use allocation matrix for directional planning")
        print(f"  🔄 Monitor actual vs predicted patterns for optimization")
    
    print("=" * 80)
    
    return predictor

if __name__ == "__main__":
    multi_predictor = main()
