#!/usr/bin/env python3
"""
Business Dashboard for E-commerce Sales Analysis
Simple interface for business owner to get insights and ordering recommendations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up paths
DATA_DIR = Path("/Users/tanayshah/Desktop/personal/projects/ecom_analysis/data")
PROCESSED_DIR = DATA_DIR / "processed"
CORE_DIR = Path("/Users/tanayshah/Desktop/personal/projects/ecom_analysis/core")
RESULTS_DIR = CORE_DIR / "analysis_results"

class BusinessDashboard:
    def __init__(self):
        self.sales_data = None
        self.predictions = None
        self.ordering_schedule = None
        self.reorder_analysis = None
        
    def load_all_data(self):
        """Load all analysis results"""
        try:
            # Load recent sales data
            sales_file = PROCESSED_DIR / "sales_data_jan_june_2025.csv"
            self.sales_data = pd.read_csv(sales_file)
            self.sales_data['Date'] = pd.to_datetime(self.sales_data['Date'])
            
            # Load predictions if available
            pred_file = RESULTS_DIR / "next_month_predictions.csv"
            if pred_file.exists():
                self.predictions = pd.read_csv(pred_file, index_col=0)
            
            # Load ordering schedule if available
            order_file = RESULTS_DIR / "ordering_schedule.csv"
            if order_file.exists():
                self.ordering_schedule = pd.read_csv(order_file)
            
            # Load reorder analysis if available
            reorder_file = RESULTS_DIR / "reorder_analysis.csv"
            if reorder_file.exists():
                self.reorder_analysis = pd.read_csv(reorder_file)
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def show_business_overview(self):
        """Show high-level business overview"""
        print("=" * 80)
        print("🏪 BUSINESS OVERVIEW DASHBOARD")
        print("=" * 80)
        
        if self.sales_data is None:
            print("❌ No sales data available")
            return
        
        # Calculate key metrics
        total_revenue = self.sales_data['Amount'].sum()
        total_orders = len(self.sales_data)
        unique_skus = self.sales_data['SKU'].nunique()
        date_range = f"{self.sales_data['Date'].min().strftime('%b %Y')} - {self.sales_data['Date'].max().strftime('%b %Y')}"
        
        # Monthly performance
        monthly_data = self.sales_data.groupby(self.sales_data['Date'].dt.to_period('M')).agg({
            'Amount': 'sum',
            'Quantity': 'sum'
        })
        
        current_month_revenue = monthly_data['Amount'].iloc[-1]
        previous_month_revenue = monthly_data['Amount'].iloc[-2] if len(monthly_data) > 1 else 0
        growth_rate = ((current_month_revenue - previous_month_revenue) / previous_month_revenue * 100) if previous_month_revenue > 0 else 0
        
        print(f"📊 PERFORMANCE SUMMARY ({date_range})")
        print(f"   💰 Total Revenue: ₹{total_revenue:,.2f}")
        print(f"   📦 Total Orders: {total_orders:,}")
        print(f"   🏷️  Active SKUs: {unique_skus}")
        print(f"   📈 Latest Month Growth: {growth_rate:+.1f}%")
        print(f"   💵 Average Order Value: ₹{total_revenue/total_orders:.2f}")
        
        # Top performing SKUs
        top_skus = self.sales_data.groupby('SKU')['Amount'].sum().sort_values(ascending=False).head(5)
        print(f"\n🏆 TOP 5 REVENUE GENERATORS:")
        for i, (sku, revenue) in enumerate(top_skus.items(), 1):
            print(f"   {i}. {sku}: ₹{revenue:,.0f}")
    
    def show_sales_predictions(self):
        """Show sales predictions for next month"""
        print("\n" + "=" * 80)
        print("🔮 NEXT MONTH SALES PREDICTIONS")
        print("=" * 80)
        
        if self.predictions is None:
            print("❌ No predictions available. Run sales prediction first.")
            return
        
        print("📈 PREDICTED SALES (Next 30 Days):")
        print("-" * 60)
        
        # Sort by predicted quantity (descending)
        sorted_predictions = self.predictions.sort_values('predicted_monthly_quantity', ascending=False)
        
        total_predicted_qty = 0
        for sku, row in sorted_predictions.head(10).iterrows():
            monthly_qty = int(row['predicted_monthly_quantity'])
            daily_avg = row['predicted_daily_average']
            confidence = row['confidence']
            
            # Confidence emoji
            conf_emoji = "🟢" if confidence == "High" else "🟡" if confidence == "Medium" else "🔴"
            
            print(f"   {conf_emoji} {sku:<12}: {monthly_qty:>4} units/month ({daily_avg:.1f}/day)")
            total_predicted_qty += monthly_qty
        
        print("-" * 60)
        print(f"   📊 TOTAL PREDICTED: {total_predicted_qty:,} units next month")
        
        # Show accuracy info
        print(f"\n💡 PREDICTION NOTES:")
        print(f"   🟢 High Confidence: Stable demand pattern")
        print(f"   🟡 Medium Confidence: Some volatility in demand")
        print(f"   🔴 Low Confidence: High volatility or limited data")
    
    def show_ordering_recommendations(self):
        """Show ordering recommendations"""
        print("\n" + "=" * 80)
        print("🛒 ORDERING RECOMMENDATIONS")
        print("=" * 80)
        
        if self.ordering_schedule is None:
            print("❌ No ordering schedule available. Run ordering optimizer first.")
            return
        
        # Separate by urgency
        critical = self.ordering_schedule[self.ordering_schedule['urgency'] == 'CRITICAL']
        high = self.ordering_schedule[self.ordering_schedule['urgency'] == 'HIGH']
        medium = self.ordering_schedule[self.ordering_schedule['urgency'] == 'MEDIUM']
        
        if len(critical) > 0:
            print("🚨 CRITICAL - ORDER IMMEDIATELY:")
            print("-" * 50)
            for _, row in critical.iterrows():
                days_left = row['days_remaining']
                qty = int(row['recommended_qty'])
                cost = row['estimated_cost']
                print(f"   ⚠️  {row['sku']:<12}: {qty:>4} units (₹{cost:,.0f}) - {days_left:.1f} days left")
        
        if len(high) > 0:
            print(f"\n🔶 HIGH PRIORITY - ORDER WITHIN 3 DAYS:")
            print("-" * 50)
            for _, row in high.iterrows():
                days_left = row['days_remaining']
                qty = int(row['recommended_qty'])
                cost = row['estimated_cost']
                print(f"   🔸 {row['sku']:<12}: {qty:>4} units (₹{cost:,.0f}) - {days_left:.1f} days left")
        
        if len(medium) > 0:
            print(f"\n🔵 MEDIUM PRIORITY - PLAN AHEAD:")
            print("-" * 50)
            for _, row in medium.head(5).iterrows():  # Show top 5 only
                days_left = row['days_remaining']
                qty = int(row['recommended_qty'])
                cost = row['estimated_cost']
                print(f"   🔹 {row['sku']:<12}: {qty:>4} units (₹{cost:,.0f}) - {days_left:.1f} days left")
        
        # Summary
        total_orders = len(self.ordering_schedule)
        total_cost = self.ordering_schedule['estimated_cost'].sum()
        
        print(f"\n📋 ORDERING SUMMARY:")
        print(f"   📦 Total SKUs to order: {total_orders}")
        print(f"   💰 Total estimated cost: ₹{total_cost:,.2f}")
        print(f"   🚨 Critical orders: {len(critical)}")
        print(f"   🔶 High priority: {len(high)}")
    
    def show_inventory_status(self):
        """Show current inventory status"""
        print("\n" + "=" * 80)
        print("📦 INVENTORY STATUS")
        print("=" * 80)
        
        if self.reorder_analysis is None:
            print("❌ No inventory analysis available.")
            return
        
        # Categorize by days remaining
        critical_stock = self.reorder_analysis[self.reorder_analysis['days_remaining'] < 7]
        low_stock = self.reorder_analysis[(self.reorder_analysis['days_remaining'] >= 7) & 
                                         (self.reorder_analysis['days_remaining'] < 14)]
        healthy_stock = self.reorder_analysis[self.reorder_analysis['days_remaining'] >= 14]
        
        print(f"📊 INVENTORY HEALTH:")
        print(f"   🚨 Critical (< 7 days): {len(critical_stock)} SKUs")
        print(f"   🟡 Low (7-14 days): {len(low_stock)} SKUs")
        print(f"   🟢 Healthy (14+ days): {len(healthy_stock)} SKUs")
        
        if len(critical_stock) > 0:
            print(f"\n🚨 CRITICAL STOCK LEVELS:")
            print("-" * 50)
            for _, row in critical_stock.iterrows():
                sku = row['sku']
                current = int(row['current_stock'])
                days = row['days_remaining']
                daily_demand = row['predicted_daily_demand']
                print(f"   ⚠️  {sku:<12}: {current:>3} units ({days:.1f} days) - {daily_demand:.1f}/day demand")
        
        # Average inventory metrics
        avg_days = self.reorder_analysis['days_remaining'].mean()
        total_inventory_value = (self.reorder_analysis['current_stock'] * 100).sum()  # Assuming ₹100 per unit
        
        print(f"\n📈 INVENTORY METRICS:")
        print(f"   📅 Average days of inventory: {avg_days:.1f} days")
        print(f"   💰 Estimated inventory value: ₹{total_inventory_value:,.2f}")
    
    def show_actionable_insights(self):
        """Show actionable business insights"""
        print("\n" + "=" * 80)
        print("💡 ACTIONABLE INSIGHTS & RECOMMENDATIONS")
        print("=" * 80)
        
        insights = []
        
        # Sales trend insights
        if self.sales_data is not None:
            monthly_data = self.sales_data.groupby(self.sales_data['Date'].dt.to_period('M'))['Amount'].sum()
            if len(monthly_data) >= 2:
                recent_growth = ((monthly_data.iloc[-1] - monthly_data.iloc[-2]) / monthly_data.iloc[-2] * 100)
                if recent_growth > 10:
                    insights.append(f"📈 Strong growth trend: {recent_growth:.1f}% month-over-month increase")
                elif recent_growth < -10:
                    insights.append(f"📉 Declining trend: {recent_growth:.1f}% month-over-month decrease - investigate causes")
        
        # Inventory insights
        if self.reorder_analysis is not None:
            critical_count = len(self.reorder_analysis[self.reorder_analysis['days_remaining'] < 7])
            if critical_count > 0:
                insights.append(f"🚨 URGENT: {critical_count} SKUs have critical stock levels - immediate action required")
            
            avg_days = self.reorder_analysis['days_remaining'].mean()
            if avg_days < 10:
                insights.append(f"⚠️  Overall inventory is low ({avg_days:.1f} days average) - consider increasing safety stock")
        
        # Ordering insights
        if self.ordering_schedule is not None:
            total_cost = self.ordering_schedule['estimated_cost'].sum()
            critical_orders = len(self.ordering_schedule[self.ordering_schedule['urgency'] == 'CRITICAL'])
            
            if critical_orders > 5:
                insights.append(f"📦 High ordering workload: {critical_orders} critical orders - consider bulk ordering")
            
            if total_cost > 500000:  # ₹5 lakh
                insights.append(f"💰 Large ordering requirement: ₹{total_cost:,.0f} - plan cash flow accordingly")
        
        # Prediction insights
        if self.predictions is not None:
            low_confidence = len(self.predictions[self.predictions['confidence'] == 'Low'])
            if low_confidence > 5:
                insights.append(f"🔍 {low_confidence} SKUs have low prediction confidence - monitor closely")
        
        # Display insights
        if insights:
            for i, insight in enumerate(insights, 1):
                print(f"   {i}. {insight}")
        else:
            print("   ✅ No critical issues identified - business operations look healthy")
        
        # General recommendations
        print(f"\n🎯 GENERAL RECOMMENDATIONS:")
        print(f"   1. Review critical stock items daily")
        print(f"   2. Place urgent orders within 24 hours")
        print(f"   3. Monitor prediction accuracy and adjust as needed")
        print(f"   4. Consider supplier negotiations for high-volume items")
        print(f"   5. Implement automated reorder points for top SKUs")
    
    def run_dashboard(self):
        """Run the complete business dashboard"""
        print("\n🚀 Loading Business Dashboard...")
        
        if not self.load_all_data():
            print("❌ Failed to load data. Please run data analysis first.")
            return
        
        # Show all sections
        self.show_business_overview()
        self.show_sales_predictions()
        self.show_ordering_recommendations()
        self.show_inventory_status()
        self.show_actionable_insights()
        
        print("\n" + "=" * 80)
        print("📊 DASHBOARD COMPLETE")
        print("=" * 80)
        print(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("For detailed analysis, check the CSV files in core/analysis_results/")

def main():
    """Main function to run the business dashboard"""
    dashboard = BusinessDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
