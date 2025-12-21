#!/usr/bin/env python3
"""
Data Exploration Script for E-commerce Sales Forecasting
Focus: Understanding data structure and identifying top SKUs from recent sales
"""

import pandas as pd
import sqlite3
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
DATA_DIR = Path("/Users/tanayshah/Desktop/personal/projects/ecom_analysis/data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CORE_DIR = Path("/Users/tanayshah/Desktop/personal/projects/ecom_analysis/core")

def load_recent_sales_data():
    """Load and explore recent sales data (Jan-June 2025)"""
    print("=== Loading Recent Sales Data (Jan-June 2025) ===")
    
    sales_file = PROCESSED_DIR / "sales_data_jan_june_2025.csv"
    df = pd.read_csv(sales_file)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Columns: {list(df.columns)}")
    print(f"Unique SKUs: {df['SKU'].nunique()}")
    print(f"Total transactions: {len(df)}")
    print(f"Total revenue: ₹{df['Amount'].sum():,.2f}")
    
    # Convert date column
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.strftime('%Y-%m')  # Convert to string for SQLite compatibility
    df['Week'] = df['Date'].dt.strftime('%Y-W%U')  # Convert to string for SQLite compatibility
    
    return df

def analyze_top_skus(df, top_n=20):
    """Identify top SKUs based on recent 4-5 months sales"""
    print(f"\n=== Analyzing Top {top_n} SKUs (Last 4-5 Months) ===")
    
    # SKU performance analysis
    sku_analysis = df.groupby('SKU').agg({
        'Quantity': ['sum', 'count', 'mean'],
        'Amount': ['sum', 'mean'],
        'Date': ['min', 'max']
    }).round(2)
    
    # Flatten column names
    sku_analysis.columns = ['_'.join(col).strip() for col in sku_analysis.columns]
    sku_analysis = sku_analysis.rename(columns={
        'Quantity_sum': 'Total_Quantity',
        'Quantity_count': 'Total_Orders',
        'Quantity_mean': 'Avg_Quantity_Per_Order',
        'Amount_sum': 'Total_Revenue',
        'Amount_mean': 'Avg_Order_Value',
        'Date_min': 'First_Sale',
        'Date_max': 'Last_Sale'
    })
    
    # Calculate additional metrics
    sku_analysis['Revenue_Per_Unit'] = sku_analysis['Total_Revenue'] / sku_analysis['Total_Quantity']
    sku_analysis['Days_Active'] = (sku_analysis['Last_Sale'] - sku_analysis['First_Sale']).dt.days + 1
    sku_analysis['Daily_Avg_Quantity'] = sku_analysis['Total_Quantity'] / sku_analysis['Days_Active']
    
    # Sort by total revenue (primary importance metric)
    top_skus = sku_analysis.sort_values('Total_Revenue', ascending=False).head(top_n)
    
    print("Top SKUs by Revenue:")
    print(top_skus[['Total_Revenue', 'Total_Quantity', 'Total_Orders', 'Daily_Avg_Quantity']].head(10))
    
    return top_skus, sku_analysis

def analyze_monthly_trends(df):
    """Analyze monthly sales trends"""
    print("\n=== Monthly Sales Trends Analysis ===")
    
    monthly_sales = df.groupby('Month').agg({
        'Quantity': 'sum',
        'Amount': 'sum',
        'SKU': 'nunique',
        'Orderid': 'nunique'
    }).round(2)
    
    monthly_sales.columns = ['Total_Quantity', 'Total_Revenue', 'Unique_SKUs', 'Unique_Orders']
    
    print("Monthly Performance:")
    print(monthly_sales)
    
    # Calculate month-over-month growth
    monthly_sales['Revenue_Growth'] = monthly_sales['Total_Revenue'].pct_change() * 100
    monthly_sales['Quantity_Growth'] = monthly_sales['Total_Quantity'].pct_change() * 100
    
    print("\nMonth-over-Month Growth:")
    print(monthly_sales[['Revenue_Growth', 'Quantity_Growth']].round(2))
    
    return monthly_sales

def load_sku_master_data():
    """Load SKU master data"""
    print("\n=== Loading SKU Master Data ===")
    
    sku_file = RAW_DIR / "sku_list.csv"
    sku_df = pd.read_csv(sku_file)
    
    print(f"SKU Master shape: {sku_df.shape}")
    print(f"Categories: {sku_df['category'].value_counts().head(10)}")
    
    return sku_df

def load_moq_leadtime_data():
    """Load MOQ and Lead Time data"""
    print("\n=== Loading MOQ and Lead Time Data ===")
    
    try:
        moq_file = RAW_DIR / "moq_leadtime.xlsx"
        moq_df = pd.read_excel(moq_file)
        
        print(f"MOQ/Lead Time shape: {moq_df.shape}")
        print(f"Columns: {list(moq_df.columns)}")
        print("Sample data:")
        print(moq_df.head())
        
        return moq_df
    except Exception as e:
        print(f"Error loading MOQ data: {e}")
        return None

def analyze_seasonality_patterns(df):
    """Analyze seasonal patterns in sales data"""
    print("\n=== Seasonality Analysis ===")
    
    # Day of week patterns
    df['DayOfWeek'] = df['Date'].dt.day_name()
    daily_patterns = df.groupby('DayOfWeek')['Amount'].sum().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ])
    
    print("Daily Sales Patterns:")
    print(daily_patterns.round(2))
    
    # Weekly patterns
    weekly_patterns = df.groupby('Week').agg({
        'Amount': 'sum',
        'Quantity': 'sum'
    })
    
    print(f"\nWeekly patterns - Weeks analyzed: {len(weekly_patterns)}")
    print("Weekly Revenue Stats:")
    print(weekly_patterns['Amount'].describe().round(2))
    
    return daily_patterns, weekly_patterns

def create_sqlite_database(df, sku_df, moq_df=None):
    """Create SQLite database for analysis"""
    print("\n=== Creating SQLite Database ===")
    
    db_path = CORE_DIR / "sales_analysis.db"
    conn = sqlite3.connect(db_path)
    
    # Save sales data
    df.to_sql('sales_data', conn, if_exists='replace', index=False)
    print(f"Saved {len(df)} sales records to database")
    
    # Save SKU master data
    sku_df.to_sql('sku_master', conn, if_exists='replace', index=False)
    print(f"Saved {len(sku_df)} SKU records to database")
    
    # Save MOQ data if available
    if moq_df is not None:
        moq_df.to_sql('moq_leadtime', conn, if_exists='replace', index=False)
        print(f"Saved {len(moq_df)} MOQ/Lead time records to database")
    
    conn.close()
    print(f"Database created at: {db_path}")
    
    return db_path

def main():
    """Main exploration function"""
    print("Starting E-commerce Data Exploration...")
    print("=" * 60)
    
    # Load recent sales data
    sales_df = load_recent_sales_data()
    
    # Analyze top SKUs
    top_skus, all_sku_analysis = analyze_top_skus(sales_df, top_n=20)
    
    # Analyze monthly trends
    monthly_trends = analyze_monthly_trends(sales_df)
    
    # Load master data
    sku_master = load_sku_master_data()
    moq_data = load_moq_leadtime_data()
    
    # Analyze seasonality
    daily_patterns, weekly_patterns = analyze_seasonality_patterns(sales_df)
    
    # Create database
    db_path = create_sqlite_database(sales_df, sku_master, moq_data)
    
    # Save analysis results
    results_dir = CORE_DIR / "analysis_results"
    results_dir.mkdir(exist_ok=True)
    
    # Save top SKUs analysis
    top_skus.to_csv(results_dir / "top_skus_analysis.csv")
    all_sku_analysis.to_csv(results_dir / "complete_sku_analysis.csv")
    monthly_trends.to_csv(results_dir / "monthly_trends.csv")
    
    print("\n=== Analysis Complete ===")
    print(f"Results saved to: {results_dir}")
    print(f"Database created at: {db_path}")
    
    # Summary insights
    print("\n=== Key Insights ===")
    print(f"1. Total SKUs in recent data: {sales_df['SKU'].nunique()}")
    print(f"2. Top 10 SKUs contribute: ₹{top_skus.head(10)['Total_Revenue'].sum():,.2f}")
    print(f"3. Average daily sales: ₹{sales_df.groupby('Date')['Amount'].sum().mean():,.2f}")
    print(f"4. Most active category: {sku_master['category'].value_counts().index[0]}")
    
    return {
        'sales_df': sales_df,
        'top_skus': top_skus,
        'monthly_trends': monthly_trends,
        'sku_master': sku_master,
        'moq_data': moq_data,
        'db_path': db_path
    }

if __name__ == "__main__":
    results = main()
