#!/usr/bin/env python3
"""
Comprehensive Ordering Analysis with Multiple Godowns/States Consideration
Creates detailed tabular view and exports to CSV/Excel
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Set up paths
DATA_DIR = Path("/Users/tanayshah/Desktop/personal/projects/ecom_analysis/data")
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = Path("/Users/tanayshah/Desktop/personal/projects/ecom_analysis/core/analysis_results")

def load_all_data():
    """Load all required data files"""
    print("=== Loading Data for Comprehensive Analysis ===")
    
    # Load sales data with godown information
    sales_file = PROCESSED_DIR / "sales_data_jan_june_2025.csv"
    sales_data = pd.read_csv(sales_file)
    sales_data['Date'] = pd.to_datetime(sales_data['Date'])
    print(f"Sales data loaded: {len(sales_data)} records")
    
    # Load predictions
    pred_file = RESULTS_DIR / "next_month_predictions.csv"
    predictions = pd.read_csv(pred_file)
    print(f"Predictions loaded: {len(predictions)} SKUs")
    
    # Load ordering schedule
    order_file = RESULTS_DIR / "ordering_schedule.csv"
    ordering = pd.read_csv(order_file)
    print(f"Ordering schedule loaded: {len(ordering)} SKUs")
    
    # Load SKU mapping
    sku_file = DATA_DIR / "raw" / "sku_list.csv"
    sku_df = pd.read_csv(sku_file)
    sku_mapping = dict(zip(sku_df['sku'], sku_df['category']))
    print(f"SKU mapping loaded: {len(sku_mapping)} products")
    
    return sales_data, predictions, ordering, sku_mapping

def analyze_godown_distribution(sales_data):
    """Analyze distribution across godowns/states"""
    print("\n=== Godown/State Distribution Analysis ===")
    
    # Godown analysis
    godown_summary = sales_data.groupby(['Godown', 'Statefrom']).agg({
        'SKU': 'count',
        'Quantity': 'sum',
        'Amount': 'sum'
    }).reset_index()
    godown_summary.columns = ['Godown', 'State_From', 'Orders', 'Total_Quantity', 'Total_Revenue']
    
    print("Godown Distribution:")
    print(godown_summary.to_string(index=False))
    
    # SKU by godown analysis
    sku_godown = sales_data.groupby(['SKU', 'Godown']).agg({
        'Quantity': 'sum',
        'Amount': 'sum'
    }).reset_index()
    
    return godown_summary, sku_godown

def create_comprehensive_ordering_table(sales_data, predictions, ordering, sku_mapping):
    """Create comprehensive ordering analysis table"""
    print("\n=== Creating Comprehensive Ordering Table ===")
    
    # Calculate current quantities by SKU and Godown
    current_stock_by_godown = sales_data.groupby(['SKU', 'Godown']).agg({
        'Quantity': 'sum',
        'Amount': 'sum'
    }).reset_index()
    
    # Calculate overall current quantities
    current_quantities = sales_data.groupby('SKU').agg({
        'Quantity': 'sum',
        'Amount': 'sum'
    }).reset_index()
    current_quantities['current_monthly_avg'] = current_quantities['Quantity'] / 6  # 6 months
    
    # Merge all data
    comprehensive_table = []
    
    for _, pred_row in predictions.iterrows():
        sku = pred_row['SKU']
        
        # Get current data
        current_data = current_quantities[current_quantities['SKU'] == sku]
        current_qty = current_data['current_monthly_avg'].iloc[0] if len(current_data) > 0 else 0
        
        # Get ordering data
        order_data = ordering[ordering['sku'] == sku]
        if len(order_data) > 0:
            order_row = order_data.iloc[0]
            lead_time = order_row['lead_time']
            order_qty = order_row['recommended_qty']
            urgency = order_row['urgency']
            current_stock = order_row['current_stock']
            estimated_cost = order_row['estimated_cost']
            days_remaining = order_row['days_remaining']
        else:
            lead_time = 'N/A'
            order_qty = 0
            urgency = 'N/A'
            current_stock = 0
            estimated_cost = 0
            days_remaining = 'N/A'
        
        # Get godown distribution for this SKU
        sku_godowns = current_stock_by_godown[current_stock_by_godown['SKU'] == sku]
        godown_info = []
        for _, godown_row in sku_godowns.iterrows():
            godown_info.append(f"{godown_row['Godown']}({godown_row['Quantity']})")
        godown_distribution = "; ".join(godown_info) if godown_info else "No godown data"
        
        comprehensive_table.append({
            'Product_Name': pred_row['product_name'],
            'SKU': sku,
            'Current_Monthly_Quantity': round(current_qty, 1),
            'Predicted_Monthly_Quantity': pred_row['predicted_monthly_quantity'],
            'Growth_Rate_Percent': round(pred_row['growth_rate'] * 100, 1),
            'Lead_Time_Days': lead_time,
            'Recommended_Order_Quantity': order_qty,
            'Current_Stock': current_stock,
            'Days_Remaining': days_remaining,
            'Urgency': urgency,
            'Estimated_Cost_INR': estimated_cost,
            'Confidence': pred_row['confidence'],
            'Godown_Distribution': godown_distribution,
            'Seasonal_Multiplier': pred_row['seasonal_multiplier']
        })
    
    df = pd.DataFrame(comprehensive_table)
    
    # Sort by urgency and then by order quantity
    urgency_order = {'CRITICAL': 1, 'HIGH': 2, 'MEDIUM': 3, 'N/A': 4}
    df['urgency_rank'] = df['Urgency'].map(urgency_order)
    df = df.sort_values(['urgency_rank', 'Recommended_Order_Quantity'], ascending=[True, False])
    df = df.drop('urgency_rank', axis=1)
    
    return df

def export_to_files(comprehensive_df, godown_summary, sku_godown):
    """Export analysis to CSV and Excel files"""
    print("\n=== Exporting Analysis to Files ===")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export comprehensive table to CSV
    csv_file = f"comprehensive_ordering_analysis_{timestamp}.csv"
    comprehensive_df.to_csv(csv_file, index=False)
    print(f"✅ Comprehensive analysis exported to: {csv_file}")
    
    # Export to Excel with multiple sheets
    excel_file = f"ordering_analysis_complete_{timestamp}.xlsx"
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # Main comprehensive table
        comprehensive_df.to_excel(writer, sheet_name='Comprehensive_Analysis', index=False)
        
        # Godown summary
        godown_summary.to_excel(writer, sheet_name='Godown_Summary', index=False)
        
        # SKU by godown breakdown
        sku_godown.to_excel(writer, sheet_name='SKU_by_Godown', index=False)
        
        # Critical orders only
        critical_orders = comprehensive_df[comprehensive_df['Urgency'] == 'CRITICAL']
        critical_orders.to_excel(writer, sheet_name='Critical_Orders_Only', index=False)
        
        # High growth SKUs
        high_growth = comprehensive_df[comprehensive_df['Growth_Rate_Percent'] > 50]
        high_growth.to_excel(writer, sheet_name='High_Growth_SKUs', index=False)
    
    print(f"✅ Complete Excel analysis exported to: {excel_file}")
    
    return csv_file, excel_file

def main():
    """Main analysis function"""
    print("🚀 Starting Comprehensive Ordering Analysis with Multiple Godowns")
    print("=" * 70)
    
    # Load data
    sales_data, predictions, ordering, sku_mapping = load_all_data()
    
    # Analyze godown distribution
    godown_summary, sku_godown = analyze_godown_distribution(sales_data)
    
    # Create comprehensive table
    comprehensive_df = create_comprehensive_ordering_table(sales_data, predictions, ordering, sku_mapping)
    
    # Display sample of comprehensive table
    print("\n=== Comprehensive Ordering Analysis (Sample) ===")
    print(comprehensive_df.head(10).to_string(index=False))
    
    print(f"\n📊 Summary Statistics:")
    print(f"Total SKUs analyzed: {len(comprehensive_df)}")
    print(f"Critical orders: {len(comprehensive_df[comprehensive_df['Urgency'] == 'CRITICAL'])}")
    print(f"High priority orders: {len(comprehensive_df[comprehensive_df['Urgency'] == 'HIGH'])}")
    print(f"Total estimated cost: ₹{comprehensive_df['Estimated_Cost_INR'].sum():,.2f}")
    print(f"Average growth rate: {comprehensive_df['Growth_Rate_Percent'].mean():.1f}%")
    
    # Export files
    csv_file, excel_file = export_to_files(comprehensive_df, godown_summary, sku_godown)
    
    print(f"\n🎯 Key Insights:")
    print(f"1. Multiple godowns identified: {sales_data['Godown'].nunique()} unique godowns")
    print(f"2. States served from: {sales_data['Statefrom'].nunique()} states")
    print(f"3. Destination states: {sales_data['Stateto'].nunique()} states")
    print(f"4. Godown distribution is included in the analysis")
    
    print(f"\n📁 Files created:")
    print(f"   📄 CSV: {csv_file}")
    print(f"   📊 Excel: {excel_file}")
    
    return comprehensive_df, csv_file, excel_file

if __name__ == "__main__":
    comprehensive_df, csv_file, excel_file = main()
