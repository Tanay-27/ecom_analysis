#!/usr/bin/env python3
"""
Ordering Optimization System
Focus: Generate optimal ordering recommendations based on sales predictions and MOQ/lead time constraints
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up paths
DATA_DIR = Path("/Users/tanayshah/Desktop/personal/projects/ecom_analysis/data")
RAW_DIR = DATA_DIR / "raw"
CORE_DIR = Path("/Users/tanayshah/Desktop/personal/projects/ecom_analysis/core")

class OrderingOptimizer:
    def __init__(self):
        self.predictions = None
        self.moq_data = None
        self.sku_master = None
        self.current_inventory = None
        
    def load_data(self):
        """Load all required data for ordering optimization"""
        print("=== Loading Data for Ordering Optimization ===")
        
        # Load predictions
        pred_file = CORE_DIR / "analysis_results" / "next_month_predictions.csv"
        if pred_file.exists():
            self.predictions = pd.read_csv(pred_file, index_col=0)
            print(f"Predictions loaded for {len(self.predictions)} SKUs")
        else:
            print("No predictions file found. Run sales_predictor.py first.")
            return False
        
        # Load MOQ and lead time data
        try:
            moq_file = RAW_DIR / "moq_leadtime.xlsx"
            moq_raw = pd.read_excel(moq_file)
            print(f"MOQ file loaded with columns: {moq_raw.columns.tolist()}")
            
            # Check if it has actual MOQ data or just categories
            if 'moq' not in moq_raw.columns:
                print("MOQ file doesn't contain MOQ/lead time data. Creating estimates...")
                self.create_dummy_moq_data()
            else:
                self.moq_data = moq_raw
                print(f"MOQ/Lead time data loaded: {len(self.moq_data)} records")
        except Exception as e:
            print(f"Error loading MOQ data: {e}")
            # Create dummy MOQ data for testing
            self.create_dummy_moq_data()
        
        # Load SKU master data
        sku_file = RAW_DIR / "sku_list.csv"
        self.sku_master = pd.read_csv(sku_file)
        print(f"SKU master data loaded: {len(self.sku_master)} records")
        
        return True
    
    def create_dummy_moq_data(self):
        """Create dummy MOQ data for testing purposes"""
        print("Creating dummy MOQ/Lead time data for testing...")
        
        # Get SKUs from predictions
        skus = self.predictions.index.tolist()
        
        # Create realistic MOQ and lead time data
        dummy_data = []
        for sku in skus:
            # Estimate MOQ based on predicted volume
            predicted_monthly = self.predictions.loc[sku, 'predicted_monthly_quantity']
            
            if predicted_monthly > 1000:
                moq = max(100, int(predicted_monthly * 0.3))  # 30% of monthly demand
                lead_time = 15  # 15 days for high volume items
            elif predicted_monthly > 500:
                moq = max(50, int(predicted_monthly * 0.4))   # 40% of monthly demand
                lead_time = 12  # 12 days for medium volume
            else:
                moq = max(20, int(predicted_monthly * 0.5))   # 50% of monthly demand
                lead_time = 10  # 10 days for low volume
            
            dummy_data.append({
                'sku': sku,
                'moq': moq,
                'lead_time_days': lead_time,
                'supplier': 'Default_Supplier',
                'unit_cost': 100  # Dummy cost
            })
        
        self.moq_data = pd.DataFrame(dummy_data)
        print(f"Created dummy MOQ data for {len(self.moq_data)} SKUs")
    
    def estimate_current_inventory(self):
        """Estimate current inventory levels (dummy data for now)"""
        print("=== Estimating Current Inventory Levels ===")
        
        # In a real system, this would come from inventory management system
        # For now, create reasonable estimates
        inventory_data = []
        
        for sku in self.predictions.index:
            predicted_daily = self.predictions.loc[sku, 'predicted_daily_average']
            
            # Assume current inventory is between 5-15 days of demand
            current_stock = int(predicted_daily * np.random.uniform(5, 15))
            
            inventory_data.append({
                'sku': sku,
                'current_stock': max(0, current_stock),
                'reserved_stock': int(current_stock * 0.1),  # 10% reserved
                'available_stock': max(0, int(current_stock * 0.9))
            })
        
        self.current_inventory = pd.DataFrame(inventory_data).set_index('sku')
        print(f"Inventory estimates created for {len(self.current_inventory)} SKUs")
    
    def calculate_reorder_points(self):
        """Calculate reorder points for each SKU"""
        print("\n=== Calculating Reorder Points ===")
        
        reorder_data = []
        
        for sku in self.predictions.index:
            # Get data
            pred_daily = self.predictions.loc[sku, 'predicted_daily_average']
            current_stock = self.current_inventory.loc[sku, 'available_stock']
            
            # Get MOQ and lead time
            moq_row = self.moq_data[self.moq_data['sku'] == sku]
            if len(moq_row) > 0:
                moq = moq_row.iloc[0]['moq']
                lead_time = moq_row.iloc[0]['lead_time_days']
            else:
                moq = max(20, int(pred_daily * 10))  # Default: 10 days demand
                lead_time = 14  # Default lead time
            
            # Calculate safety stock (3-5 days of demand)
            safety_stock = int(pred_daily * 4)
            
            # Calculate reorder point
            reorder_point = int((pred_daily * lead_time) + safety_stock)
            
            # Calculate days of inventory remaining
            days_remaining = current_stock / pred_daily if pred_daily > 0 else 999
            
            # Determine if reorder is needed
            needs_reorder = current_stock <= reorder_point
            
            # Calculate optimal order quantity
            if needs_reorder:
                # Order enough for lead time + buffer
                target_stock = int(pred_daily * (lead_time + 30))  # 30 days buffer
                order_quantity = max(moq, target_stock - current_stock)
            else:
                order_quantity = 0
            
            reorder_data.append({
                'sku': sku,
                'current_stock': current_stock,
                'predicted_daily_demand': round(pred_daily, 2),
                'reorder_point': reorder_point,
                'safety_stock': safety_stock,
                'moq': moq,
                'lead_time_days': lead_time,
                'days_remaining': round(days_remaining, 1),
                'needs_reorder': needs_reorder,
                'recommended_order_qty': order_quantity,
                'order_priority': 'High' if days_remaining < lead_time else 'Medium' if needs_reorder else 'Low'
            })
        
        reorder_df = pd.DataFrame(reorder_data)
        print(f"Reorder points calculated for {len(reorder_df)} SKUs")
        
        return reorder_df
    
    def generate_ordering_schedule(self, reorder_df):
        """Generate optimal ordering schedule"""
        print("\n=== Generating Ordering Schedule ===")
        
        # Filter SKUs that need reordering
        needs_order = reorder_df[reorder_df['needs_reorder'] == True].copy()
        
        if len(needs_order) == 0:
            print("No SKUs currently need reordering")
            return pd.DataFrame()
        
        # Sort by priority and days remaining
        priority_order = {'High': 1, 'Medium': 2, 'Low': 3}
        needs_order['priority_rank'] = needs_order['order_priority'].map(priority_order)
        needs_order = needs_order.sort_values(['priority_rank', 'days_remaining'])
        
        # Add ordering schedule
        ordering_schedule = []
        
        for idx, row in needs_order.iterrows():
            sku = row['sku']
            days_remaining = row['days_remaining']
            lead_time = row['lead_time_days']
            
            # Determine when to place order
            if days_remaining <= lead_time:
                order_date = "URGENT - Order Today"
                urgency = "CRITICAL"
            elif days_remaining <= lead_time + 3:
                order_date = "Order within 3 days"
                urgency = "HIGH"
            else:
                days_to_order = max(1, int(days_remaining - lead_time - 2))
                order_date = f"Order in {days_to_order} days"
                urgency = "MEDIUM"
            
            # Calculate order cost (dummy calculation)
            order_cost = row['recommended_order_qty'] * 100  # Assume ₹100 per unit
            
            ordering_schedule.append({
                'sku': sku,
                'urgency': urgency,
                'order_date': order_date,
                'recommended_qty': row['recommended_order_qty'],
                'moq': row['moq'],
                'estimated_cost': order_cost,
                'days_remaining': row['days_remaining'],
                'lead_time': row['lead_time_days'],
                'current_stock': row['current_stock'],
                'daily_demand': row['predicted_daily_demand']
            })
        
        schedule_df = pd.DataFrame(ordering_schedule)
        print(f"Ordering schedule created for {len(schedule_df)} SKUs")
        
        return schedule_df
    
    def create_summary_report(self, reorder_df, schedule_df):
        """Create summary report for management"""
        print("\n=== Creating Summary Report ===")
        
        total_skus = len(reorder_df)
        need_reorder = len(reorder_df[reorder_df['needs_reorder'] == True])
        urgent_orders = len(schedule_df[schedule_df['urgency'] == 'CRITICAL'])
        total_order_value = schedule_df['estimated_cost'].sum()
        
        summary = {
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'total_skus_analyzed': total_skus,
            'skus_need_reorder': need_reorder,
            'urgent_orders': urgent_orders,
            'total_estimated_order_value': f"₹{total_order_value:,.2f}",
            'avg_days_inventory': round(reorder_df['days_remaining'].mean(), 1),
            'stockout_risk_skus': len(reorder_df[reorder_df['days_remaining'] < 7])
        }
        
        print("Summary Report:")
        for key, value in summary.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        return summary
    
    def save_results(self, reorder_df, schedule_df, summary):
        """Save all results to files"""
        results_dir = CORE_DIR / "analysis_results"
        results_dir.mkdir(exist_ok=True)
        
        # Save detailed reorder analysis
        reorder_df.to_csv(results_dir / "reorder_analysis.csv", index=False)
        
        # Save ordering schedule
        schedule_df.to_csv(results_dir / "ordering_schedule.csv", index=False)
        
        # Save summary report
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(results_dir / "ordering_summary.csv", index=False)
        
        print(f"\nResults saved to: {results_dir}")
        print("  - reorder_analysis.csv")
        print("  - ordering_schedule.csv")
        print("  - ordering_summary.csv")
    
    def run_optimization(self):
        """Run the complete ordering optimization pipeline"""
        print("Starting Ordering Optimization Pipeline...")
        print("=" * 60)
        
        # Load data
        if not self.load_data():
            return None
        
        # Estimate current inventory
        self.estimate_current_inventory()
        
        # Calculate reorder points
        reorder_df = self.calculate_reorder_points()
        
        # Generate ordering schedule
        schedule_df = self.generate_ordering_schedule(reorder_df)
        
        # Create summary report
        summary = self.create_summary_report(reorder_df, schedule_df)
        
        # Save results
        self.save_results(reorder_df, schedule_df, summary)
        
        return reorder_df, schedule_df, summary

def main():
    """Main function to run ordering optimization"""
    optimizer = OrderingOptimizer()
    results = optimizer.run_optimization()
    
    if results:
        reorder_df, schedule_df, summary = results
        
        print("\n=== TOP PRIORITY ORDERS ===")
        if len(schedule_df) > 0:
            urgent = schedule_df[schedule_df['urgency'].isin(['CRITICAL', 'HIGH'])].head(5)
            for _, row in urgent.iterrows():
                print(f"{row['sku']}: {row['recommended_qty']} units - {row['order_date']} ({row['urgency']})")
        else:
            print("No urgent orders needed at this time.")
    
    return results

if __name__ == "__main__":
    results = main()
