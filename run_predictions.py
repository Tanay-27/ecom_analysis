#!/usr/bin/env python3
"""
Complete Prediction System Runner
Executes full pipeline: data loading → predictions → multi-dimensional analysis → export
"""

import sys
from pathlib import Path
from datetime import datetime

# Add core modules to path
sys.path.append(str(Path(__file__).parent / "core"))

from data_coordinator import DataCoordinator
from sales_prediction_engine import SalesPredictionEngine
from multi_dimensional_predictor import MultiDimensionalPredictor
from data_export_manager import DataExportManager
from prediction_api import PredictionAPI

def run_complete_prediction_pipeline(data_dir: Path, export_results: bool = True):
    """Run the complete prediction pipeline with multi-dimensional analysis"""
    
    print("=" * 80)
    print("🚀 COMPLETE SALES PREDICTION PIPELINE")
    print("=" * 80)
    
    # Step 1: Data Coordination
    print("\n📊 Step 1: Data Coordination")
    coordinator = DataCoordinator(data_dir)
    unified_data = coordinator.unify_all_data()
    
    if unified_data.empty:
        print("❌ No data available. Please check data files.")
        return None
    
    coordinator.save_unified_data()
    summary = coordinator.get_data_summary()
    print(f"✅ Unified {summary['total_records']:,} records, {summary['unique_skus']} SKUs")
    
    # Step 2: Main Prediction Engine
    print("\n🤖 Step 2: Sales Prediction Engine")
    engine = SalesPredictionEngine(data_dir)
    engine.load_and_prepare_data()
    engine.train_models()
    engine.save_model()
    
    performance = engine.get_system_performance()
    print(f"✅ Trained models for {performance['active_skus']} SKUs")
    print(f"   Median MAPE: {performance['median_mape']:.2f}%")
    
    # Step 3: Multi-Dimensional Analysis
    print("\n🌍 Step 3: Multi-Dimensional Analysis")
    multi_predictor = MultiDimensionalPredictor(data_dir)
    setup_results = multi_predictor.analyze_and_setup()
    
    print(f"✅ Analyzed {setup_results['channels_analyzed']} channels, "
          f"{setup_results['states_analyzed']} states, "
          f"{setup_results['godowns_analyzed']} fulfillment centers")
    
    # Step 4: Generate Predictions for All Active SKUs
    print("\n🔮 Step 4: Generating Predictions")
    api = PredictionAPI(data_dir)
    active_skus = api.get_active_skus()
    
    all_predictions = {}
    successful_predictions = 0
    
    for i, sku in enumerate(active_skus, 1):
        print(f"   Processing {i}/{len(active_skus)}: {sku}", end="")
        
        try:
            # Get main prediction
            prediction = api.get_prediction(sku, months_ahead=3)
            
            if 'error' not in prediction:
                total_pred = prediction['predictions'][0]  # Next month
                
                # Get multi-dimensional breakdown
                multi_result = multi_predictor.get_inventory_allocation_matrix(
                    sku, total_pred, months_ahead=3
                )
                
                all_predictions[sku] = multi_result
                successful_predictions += 1
                print(" ✅")
            else:
                print(f" ❌ {prediction['error']}")
                
        except Exception as e:
            print(f" ❌ Error: {str(e)}")
    
    print(f"✅ Generated predictions for {successful_predictions}/{len(active_skus)} SKUs")
    
    # Step 5: Export Results
    if export_results and all_predictions:
        print("\n📤 Step 5: Exporting Results")
        export_manager = DataExportManager(data_dir)
        
        # Store in database
        for sku, result in all_predictions.items():
            export_manager.store_prediction_results(sku, result)
        
        # Export to Excel
        excel_path = export_manager.export_to_excel(all_predictions)
        print(f"✅ Excel export: {excel_path}")
        print(f"✅ Database: {export_manager.db_path}")
        
        return {
            'predictions': all_predictions,
            'excel_file': excel_path,
            'database': export_manager.db_path,
            'export_manager': export_manager
        }
    
    return {
        'predictions': all_predictions,
        'export_manager': None
    }

def run_single_sku_analysis(data_dir: Path, sku: str):
    """Run detailed analysis for a single SKU"""
    
    print(f"🔍 Single SKU Analysis: {sku}")
    print("=" * 50)
    
    # Initialize components
    api = PredictionAPI(data_dir)
    multi_predictor = MultiDimensionalPredictor(data_dir)
    multi_predictor.analyze_and_setup()
    
    # Get prediction
    prediction = api.get_prediction(sku, months_ahead=3)
    
    if 'error' in prediction:
        print(f"❌ Error: {prediction['error']}")
        return None
    
    # Get multi-dimensional analysis
    total_pred = prediction['predictions'][0]
    multi_result = multi_predictor.get_inventory_allocation_matrix(
        sku, total_pred, months_ahead=3
    )
    
    # Display results
    print(f"\n📈 Total Prediction: {total_pred:.1f} units")
    print(f"🎯 Confidence: {prediction['confidence']['confidence_level']}")
    print(f"📊 MAPE: {prediction['historical_accuracy']['mape']:.2f}%")
    
    print(f"\n📺 Channel Breakdown:")
    for channel, volume in multi_result['channel_breakdown']['predictions'].items():
        if volume > 0:
            pct = (volume / total_pred * 100) if total_pred > 0 else 0
            print(f"   • {channel}: {volume:.1f} units ({pct:.1f}%)")
    
    print(f"\n🗺️  Top States:")
    geo_items = sorted(multi_result['geographical_breakdown']['predictions'].items(),
                      key=lambda x: x[1], reverse=True)[:5]
    for state, volume in geo_items:
        if volume > 0 and state != 'Others':
            pct = (volume / total_pred * 100) if total_pred > 0 else 0
            print(f"   • {state}: {volume:.1f} units ({pct:.1f}%)")
    
    print(f"\n🏭 Top Fulfillment Centers:")
    godown_items = sorted(multi_result['godown_breakdown']['predictions'].items(),
                         key=lambda x: x[1], reverse=True)[:5]
    for godown, volume in godown_items:
        if volume > 0:
            channel, city = godown.split('_', 1)
            pct = (volume / total_pred * 100) if total_pred > 0 else 0
            print(f"   • {channel} in {city}: {volume:.1f} units ({pct:.1f}%)")
    
    print(f"\n🎯 Inventory Allocation Matrix:")
    for allocation_key, allocation_data in multi_result['inventory_allocation_matrix'].items():
        channel, state = allocation_key.split('_', 1)
        total_alloc = allocation_data['total_allocation']
        pct = (total_alloc / total_pred * 100) if total_pred > 0 else 0
        print(f"\n   📦 {channel} → {state}: {total_alloc:.1f} units ({pct:.1f}%)")
        
        for godown_info in allocation_data['godowns']:
            city = godown_info['city']
            units = godown_info['allocation']
            godown_pct = godown_info['percentage']
            print(f"      └── {city}: {units:.1f} units ({godown_pct:.1f}% of {state})")
    
    return multi_result

def query_fulfillment_center(data_dir: Path, channel: str, city: str):
    """Query predictions for a specific fulfillment center"""
    
    export_manager = DataExportManager(data_dir)
    result = export_manager.query_by_fulfillment_center(channel, city)
    
    if 'error' in result:
        print(f"❌ {result['error']}")
        return None
    
    print(f"🏭 Fulfillment Center Analysis: {channel} in {city}")
    print("=" * 60)
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Sales Prediction System')
    parser.add_argument('--mode', choices=['full', 'sku', 'fulfillment'], default='full',
                       help='Execution mode')
    parser.add_argument('--sku', type=str, help='SKU for single analysis')
    parser.add_argument('--channel', type=str, help='Channel for fulfillment center query')
    parser.add_argument('--city', type=str, help='City for fulfillment center query')
    parser.add_argument('--no-export', action='store_true', help='Skip export step')
    
    args = parser.parse_args()
    
    data_dir = Path(__file__).parent / "data"
    
    if args.mode == 'full':
        result = run_complete_prediction_pipeline(data_dir, export_results=not args.no_export)
        
        if result and result['export_manager']:
            print(f"\n💡 Next Steps:")
            print(f"   • View Excel file: {result['excel_file']}")
            print(f"   • Query database: {result['database']}")
            print(f"   • Run single SKU analysis: python run_predictions.py --mode sku --sku SKU_NAME")
            print(f"   • Query fulfillment center: python run_predictions.py --mode fulfillment --channel Amazon --city HYDERABAD")
    
    elif args.mode == 'sku':
        if not args.sku:
            print("❌ Please provide --sku parameter")
            return
        
        run_single_sku_analysis(data_dir, args.sku)
    
    elif args.mode == 'fulfillment':
        if not args.channel or not args.city:
            print("❌ Please provide both --channel and --city parameters")
            return
        
        query_fulfillment_center(data_dir, args.channel, args.city)

if __name__ == "__main__":
    main()
