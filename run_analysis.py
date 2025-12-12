#!/usr/bin/env python3
"""
Master Script for E-commerce Sales Analysis
Runs the complete analysis pipeline: data exploration -> prediction -> ordering -> dashboard
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print(result.stdout)
            print(f"✅ {description} completed successfully!")
            return True
        else:
            print(f"❌ Error in {description}:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Failed to run {description}: {e}")
        return False

def main():
    """Run the complete analysis pipeline"""
    print("🏪 E-COMMERCE SALES ANALYSIS PIPELINE")
    print("="*60)
    print("This will run the complete analysis in sequence:")
    print("1. Data Exploration & Analysis")
    print("2. Sales Prediction Algorithm")
    print("3. Ordering Optimization")
    print("4. Business Dashboard")
    print("="*60)
    
    # Define scripts to run
    scripts = [
        ("core/data_exploration.py", "Data Exploration & Analysis"),
        ("core/sales_predictor.py", "Sales Prediction Algorithm"),
        ("core/ordering_optimizer.py", "Ordering Optimization"),
        ("core/business_dashboard.py", "Business Dashboard")
    ]
    
    # Run each script
    success_count = 0
    for script_path, description in scripts:
        if run_script(script_path, description):
            success_count += 1
        else:
            print(f"\n⚠️  Pipeline stopped due to error in {description}")
            break
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"📊 PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Completed: {success_count}/{len(scripts)} steps")
    
    if success_count == len(scripts):
        print("🎉 All analysis completed successfully!")
        print("\n📁 Results available in:")
        print("   - core/analysis_results/")
        print("   - Check CSV files for detailed data")
        print("\n💡 Key outputs:")
        print("   - next_month_predictions.csv: Sales forecasts")
        print("   - ordering_schedule.csv: What to order and when")
        print("   - reorder_analysis.csv: Detailed inventory analysis")
    else:
        print("⚠️  Pipeline incomplete. Check error messages above.")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
