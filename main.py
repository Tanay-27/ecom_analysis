#!/usr/bin/env python3
"""
Sales Prediction System - Main Entry Point
Clean, modular interface for the complete prediction system
"""

import sys
import argparse
from pathlib import Path
from typing import Optional

# Add core modules to path
sys.path.append(str(Path(__file__).parent / "core"))

from core import (
    DataCoordinator,
    SalesPredictionEngine, 
    HybridPredictionEngine,
    MultiDimensionalPredictor,
    DataExportManager,
    PredictionAPI
)

def setup_system(data_dir: Path) -> dict:
    """Initialize and setup the complete prediction system"""
    print("🚀 Setting up Sales Prediction System...")
    
    # Step 1: Data coordination
    print("\n📊 Step 1: Data Coordination")
    coordinator = DataCoordinator(data_dir)
    unified_data = coordinator.unify_all_data()
    
    if unified_data.empty:
        raise ValueError("No data available. Please check data files.")
    
    coordinator.save_unified_data()
    summary = coordinator.get_data_summary()
    print(f"✅ Unified {summary['total_records']:,} records, {summary['unique_skus']} SKUs")
    
    # Step 2: Train individual models
    print("\n🤖 Step 2: Training Individual Models")
    engine = SalesPredictionEngine(data_dir)
    filtered_data = engine.load_and_filter_data()
    engine.train_models(filtered_data)
    engine.save_model()
    
    performance = engine.get_system_performance()
    print(f"✅ Trained models for {performance['active_skus']} SKUs")
    print(f"   Median MAPE: {performance['median_mape']:.2f}%")
    
    # Step 3: Setup multi-dimensional analysis
    print("\n🌍 Step 3: Multi-Dimensional Analysis Setup")
    multi_predictor = MultiDimensionalPredictor(data_dir)
    setup_results = multi_predictor.analyze_and_setup()
    
    print(f"✅ Analyzed {setup_results['channels_analyzed']} channels, "
          f"{setup_results['states_analyzed']} states, "
          f"{setup_results['godowns_analyzed']} fulfillment centers")
    
    return {
        'coordinator': coordinator,
        'engine': engine,
        'multi_predictor': multi_predictor,
        'performance': performance,
        'summary': summary
    }

def predict_single_sku(sku: str, data_dir: Path, months_ahead: int = 3, use_hybrid: bool = True):
    """Get detailed prediction for a single SKU"""
    print(f"🔍 Analyzing SKU: {sku}")
    print("=" * 50)
    
    # Initialize API
    api = PredictionAPI(data_dir, use_hybrid=use_hybrid)
    
    # Get prediction
    result = api.get_prediction(sku, months_ahead=months_ahead)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return None
    
    # Display results
    print(f"\n📈 Prediction Results:")
    print(f"  Total Prediction: {result['predictions'][0]:.1f} units")
    print(f"  Confidence: {result['confidence']['confidence_level']}")
    
    if 'historical_accuracy' in result:
        mape = result['historical_accuracy'].get('mape')
        if mape is not None:
            print(f"  Historical MAPE: {mape:.2f}%")
    
    if use_hybrid and 'hybrid_info' in result:
        approach = result['hybrid_info']['approach_used']
        reason = result['hybrid_info']['decision_reason']
        print(f"  Approach: {approach}")
        print(f"  Reason: {reason}")
    
    # Get multi-dimensional breakdown
    multi_predictor = MultiDimensionalPredictor(data_dir)
    multi_predictor.analyze_and_setup()
    
    allocation = multi_predictor.get_inventory_allocation_matrix(
        sku, result['predictions'][0], months_ahead=months_ahead
    )
    
    print(f"\n📺 Channel Breakdown:")
    for channel, volume in allocation['channel_breakdown']['predictions'].items():
        if volume > 0:
            pct = (volume / result['predictions'][0] * 100) if result['predictions'][0] > 0 else 0
            print(f"   • {channel}: {volume:.1f} units ({pct:.1f}%)")
    
    print(f"\n🏭 Top Fulfillment Centers:")
    godown_items = sorted(allocation['godown_breakdown']['predictions'].items(),
                         key=lambda x: x[1], reverse=True)[:5]
    for godown, volume in godown_items:
        if volume > 0:
            channel, city = godown.split('_', 1)
            pct = (volume / result['predictions'][0] * 100) if result['predictions'][0] > 0 else 0
            print(f"   • {channel} in {city}: {volume:.1f} units ({pct:.1f}%)")
    
    return result

def batch_predictions(data_dir: Path, use_hybrid: bool = True, export: bool = True):
    """Generate predictions for all active SKUs"""
    print("🔮 Generating Batch Predictions")
    print("=" * 40)
    
    # Initialize components
    api = PredictionAPI(data_dir, use_hybrid=use_hybrid)
    active_skus = api.get_active_skus()
    
    print(f"Processing {len(active_skus)} active SKUs...")
    
    all_predictions = {}
    successful = 0
    
    for i, sku in enumerate(active_skus, 1):
        print(f"  {i}/{len(active_skus)}: {sku}", end="")
        
        try:
            result = api.get_prediction(sku, months_ahead=3)
            if 'error' not in result:
                all_predictions[sku] = result
                successful += 1
                print(" ✅")
            else:
                print(f" ❌ {result['error']}")
        except Exception as e:
            print(f" ❌ Error: {str(e)}")
    
    print(f"\n✅ Generated predictions for {successful}/{len(active_skus)} SKUs")
    
    if export and all_predictions:
        print("\n📤 Exporting Results...")
        export_manager = DataExportManager(data_dir)
        
        # Get multi-dimensional data
        multi_predictor = MultiDimensionalPredictor(data_dir)
        multi_predictor.analyze_and_setup()
        
        enhanced_predictions = {}
        for sku, pred in all_predictions.items():
            try:
                allocation = multi_predictor.get_inventory_allocation_matrix(
                    sku, pred['predictions'][0], months_ahead=3
                )
                enhanced_predictions[sku] = allocation
                
                # Store in database
                export_manager.store_prediction_results(sku, allocation)
            except Exception as e:
                print(f"   Warning: Multi-dimensional analysis failed for {sku}: {e}")
        
        # Export to Excel
        if enhanced_predictions:
            excel_path = export_manager.export_to_excel(enhanced_predictions)
            print(f"✅ Excel export: {excel_path}")
            print(f"✅ Database: {export_manager.db_path}")
    
    return all_predictions

def query_fulfillment_center(channel: str, city: str, data_dir: Path):
    """Query predictions for a specific fulfillment center"""
    export_manager = DataExportManager(data_dir)
    result = export_manager.query_by_fulfillment_center(channel, city)
    
    if 'error' in result:
        print(f"❌ {result['error']}")
        return None
    
    print(f"🏭 Fulfillment Center: {channel} in {city}")
    print("=" * 50)
    print(f"📊 Total Predicted Volume: {result['total_predicted_volume']:.1f} units")
    print(f"📦 SKUs Handled: {result['sku_count']}")
    
    print(f"\n🔝 Top SKUs:")
    for sku_info in sorted(result['sku_predictions'], 
                          key=lambda x: x['predicted_volume'], reverse=True)[:10]:
        print(f"   • {sku_info['sku']}: {sku_info['predicted_volume']:.1f} units "
              f"({sku_info['percentage']:.1f}%)")
    
    return result

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Sales Prediction System')
    parser.add_argument('--mode', 
                       choices=['setup', 'predict', 'batch', 'fulfillment'], 
                       default='setup',
                       help='Execution mode')
    parser.add_argument('--sku', type=str, help='SKU for single prediction')
    parser.add_argument('--channel', type=str, help='Channel for fulfillment query')
    parser.add_argument('--city', type=str, help='City for fulfillment query')
    parser.add_argument('--months', type=int, default=3, help='Months ahead to predict')
    parser.add_argument('--no-hybrid', action='store_true', help='Disable hybrid approach')
    parser.add_argument('--no-export', action='store_true', help='Skip export step')
    
    args = parser.parse_args()
    
    data_dir = Path(__file__).parent / "data"
    use_hybrid = not args.no_hybrid
    
    try:
        if args.mode == 'setup':
            setup_results = setup_system(data_dir)
            print(f"\n💡 System ready! Use these commands:")
            print(f"   • Single SKU: python main.py --mode predict --sku SKU_NAME")
            print(f"   • Batch predictions: python main.py --mode batch")
            print(f"   • Query fulfillment: python main.py --mode fulfillment --channel Amazon --city HYDERABAD")
            
        elif args.mode == 'predict':
            if not args.sku:
                print("❌ Please provide --sku parameter")
                return
            predict_single_sku(args.sku, data_dir, args.months, use_hybrid)
            
        elif args.mode == 'batch':
            batch_predictions(data_dir, use_hybrid, not args.no_export)
            
        elif args.mode == 'fulfillment':
            if not args.channel or not args.city:
                print("❌ Please provide both --channel and --city parameters")
                return
            query_fulfillment_center(args.channel, args.city, data_dir)
            
    except Exception as e:
        print(f"❌ System error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
