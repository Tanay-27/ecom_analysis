#!/usr/bin/env python3
"""
Multi-Source Data Coordinator for Sales Prediction Pipeline
Handles unification of different data formats and sources
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCoordinator:
    """Coordinates data from multiple sources with unified schema"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.collected_dir = self.data_dir / "collected"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(exist_ok=True)
        
        # Unified schema columns
        self.unified_schema = {
            'Date': 'datetime64[ns]',
            'OrderID': 'object',
            'Platform': 'object',  # Amazon, Flipkart, etc.
            'SKU': 'object',
            'ASIN': 'object',
            'Quantity': 'int64',
            'Rate': 'float64',
            'Amount': 'float64',
            'State': 'object',
            'City': 'object',
            'Pincode': 'object',
            'Category': 'object'
        }
        
        self.data_sources = {}
        self.unified_data = None
        
    def detect_data_sources(self) -> Dict[str, List[Path]]:
        """Detect and categorize available data sources"""
        logger.info("Detecting available data sources...")
        
        sources = {
            'historical': [],
            'recent': [],
            'amazon': [],
            'flipkart': [],
            'database': []
        }
        
        for file_path in self.collected_dir.glob("*"):
            if file_path.is_file():
                filename = file_path.name.lower()
                
                if 'historical' in filename or '2018' in filename or '2024' in filename:
                    sources['historical'].append(file_path)
                elif 'jan_june_2025' in filename or 'recent' in filename:
                    sources['recent'].append(file_path)
                elif 'amazon' in filename:
                    sources['amazon'].append(file_path)
                elif 'flipkart' in filename:
                    sources['flipkart'].append(file_path)
                elif 'db_sales_cache' in filename:
                    sources['database'].append(file_path)
        
        self.data_sources = sources
        
        for source_type, files in sources.items():
            logger.info(f"{source_type.title()}: {len(files)} files")
            for file in files:
                logger.info(f"  - {file.name}")
        
        return sources
    
    def load_historical_data(self) -> pd.DataFrame:
        """Load and standardize historical data (2018-Nov 2024)"""
        logger.info("Loading historical data...")
        
        historical_files = self.data_sources.get('historical', [])
        if not historical_files:
            logger.warning("No historical data files found")
            return pd.DataFrame()
        
        dfs = []
        for file_path in historical_files:
            try:
                df = pd.read_csv(file_path)
                df = self._standardize_historical_format(df)
                df['Source'] = f"Historical_{file_path.stem}"
                dfs.append(df)
                logger.info(f"Loaded {len(df)} records from {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {e}")
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Total historical records: {len(combined_df)}")
            return combined_df
        
        return pd.DataFrame()
    
    def load_recent_data(self) -> pd.DataFrame:
        """Load and standardize recent data (Jan-June 2025)"""
        logger.info("Loading recent data...")
        
        recent_files = self.data_sources.get('recent', [])
        if not recent_files:
            logger.warning("No recent data files found")
            return pd.DataFrame()
        
        dfs = []
        for file_path in recent_files:
            try:
                df = pd.read_csv(file_path)
                df = self._standardize_recent_format(df)
                df['Source'] = f"Recent_{file_path.stem}"
                dfs.append(df)
                logger.info(f"Loaded {len(df)} records from {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {e}")
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Total recent records: {len(combined_df)}")
            return combined_df
        
        return pd.DataFrame()
    
    def load_amazon_data(self) -> pd.DataFrame:
        """Load and standardize Amazon sales data"""
        logger.info("Loading Amazon data...")
        
        amazon_files = self.data_sources.get('amazon', [])
        if not amazon_files:
            logger.warning("No Amazon data files found")
            return pd.DataFrame()
        
        dfs = []
        for file_path in amazon_files:
            try:
                df = pd.read_csv(file_path)
                df = self._standardize_amazon_format(df)
                df['Source'] = f"Amazon_{file_path.stem}"
                dfs.append(df)
                logger.info(f"Loaded {len(df)} records from {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {e}")
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Total Amazon records: {len(combined_df)}")
            return combined_df
        
        return pd.DataFrame()
    
    def load_flipkart_data(self) -> pd.DataFrame:
        """Load and standardize Flipkart sales data"""
        logger.info("Loading Flipkart data...")
        
        flipkart_files = self.data_sources.get('flipkart', [])
        if not flipkart_files:
            logger.warning("No Flipkart data files found")
            return pd.DataFrame()
        
        dfs = []
        for file_path in flipkart_files:
            try:
                if file_path.suffix.lower() == '.xlsx':
                    df = pd.read_excel(file_path)
                else:
                    df = pd.read_csv(file_path)
                
                df = self._standardize_flipkart_format(df)
                df['Source'] = f"Flipkart_{file_path.stem}"
                dfs.append(df)
                logger.info(f"Loaded {len(df)} records from {file_path.name}")
            except Exception as e:
                logger.error(f"Error loading {file_path.name}: {e}")
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Total Flipkart records: {len(combined_df)}")
            return combined_df
        
        return pd.DataFrame()
    
    def _standardize_historical_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize historical data format to unified schema"""
        standardized = pd.DataFrame()
        
        # Map columns to unified schema
        column_mapping = {
            'Date': 'Date',
            'Orderid': 'OrderID',
            'SKU': 'SKU',
            'sku': 'SKU',
            'Asin': 'ASIN',
            'Quantity': 'Quantity',
            'Rate': 'Rate',
            'Amount': 'Amount',
            'Stateto': 'State',
            'City': 'City',
            'Pincode': 'Pincode'
        }
        
        for unified_col, col_type in self.unified_schema.items():
            if unified_col in column_mapping:
                source_col = column_mapping[unified_col]
                if source_col in df.columns:
                    standardized[unified_col] = df[source_col]
                else:
                    standardized[unified_col] = None
            else:
                standardized[unified_col] = None
        
        # Set platform
        standardized['Platform'] = df.get('Partyname', 'Unknown')
        
        # Parse dates
        standardized['Date'] = pd.to_datetime(standardized['Date'], errors='coerce')
        
        # Clean numeric columns
        standardized['Quantity'] = pd.to_numeric(standardized['Quantity'], errors='coerce').fillna(0).astype(int)
        standardized['Rate'] = pd.to_numeric(standardized['Rate'], errors='coerce').fillna(0.0)
        standardized['Amount'] = pd.to_numeric(standardized['Amount'], errors='coerce').fillna(0.0)
        
        return standardized.dropna(subset=['Date', 'SKU'])
    
    def _standardize_recent_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize recent data format to unified schema"""
        return self._standardize_historical_format(df)  # Same format
    
    def _standardize_amazon_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize Amazon data format to unified schema"""
        standardized = pd.DataFrame()
        
        # Map Amazon columns to unified schema
        standardized['Date'] = pd.to_datetime(df.get('Purchase Date', df.get('Shipment Date')), errors='coerce')
        standardized['OrderID'] = df.get('Amazon Order Id', '')
        standardized['Platform'] = 'Amazon'
        standardized['SKU'] = df.get('Merchant SKU', '')
        standardized['ASIN'] = None  # Not available in this format
        standardized['Quantity'] = pd.to_numeric(df.get('Shipped Quantity', 1), errors='coerce').fillna(1).astype(int)
        standardized['Rate'] = pd.to_numeric(df.get('Item Price', 0), errors='coerce').fillna(0.0)
        standardized['Amount'] = standardized['Rate'] * standardized['Quantity']
        standardized['State'] = df.get('Shipping State', '')
        standardized['City'] = df.get('Shipping City', '')
        standardized['Pincode'] = df.get('Shipping Postal Code', '')
        standardized['Category'] = None  # Will be mapped later
        
        return standardized.dropna(subset=['Date', 'SKU'])
    
    def _standardize_flipkart_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize Flipkart data format to unified schema"""
        standardized = pd.DataFrame()
        
        # This will need to be adjusted based on actual Flipkart file structure
        # For now, assuming similar structure to historical data
        return self._standardize_historical_format(df)
    
    def validate_data_quality(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """Validate and clean data quality issues"""
        logger.info(f"Validating data quality for {source_name}...")
        
        initial_count = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['Date', 'OrderID', 'SKU'], keep='first')
        
        # Remove invalid dates
        df = df.dropna(subset=['Date'])
        df = df[df['Date'] >= '2018-01-01']
        # Handle timezone-aware dates by converting to naive
        current_date = pd.Timestamp.now()
        if hasattr(df['Date'].dtype, 'tz') and df['Date'].dtype.tz is not None:
            df['Date'] = df['Date'].dt.tz_convert(None)
        df = df[df['Date'] <= current_date]
        
        # Remove invalid quantities and amounts
        df = df[df['Quantity'] > 0]
        df = df[df['Amount'] >= 0]
        
        # Remove empty SKUs
        df = df[df['SKU'].notna()]
        df = df[df['SKU'] != '']
        
        final_count = len(df)
        removed_count = initial_count - final_count
        
        logger.info(f"{source_name}: Removed {removed_count} invalid records ({removed_count/initial_count*100:.1f}%)")
        logger.info(f"{source_name}: {final_count} valid records remaining")
        
        return df
    
    def unify_all_data(self) -> pd.DataFrame:
        """Load and unify all data sources"""
        logger.info("Starting data unification process...")
        
        # Detect sources
        self.detect_data_sources()
        
        # Load all data sources
        historical_df = self.load_historical_data()
        recent_df = self.load_recent_data()
        amazon_df = self.load_amazon_data()
        flipkart_df = self.load_flipkart_data()
        
        # Validate data quality
        dfs_to_combine = []
        
        if not historical_df.empty:
            historical_df = self.validate_data_quality(historical_df, "Historical")
            dfs_to_combine.append(historical_df)
        
        if not recent_df.empty:
            recent_df = self.validate_data_quality(recent_df, "Recent")
            dfs_to_combine.append(recent_df)
        
        if not amazon_df.empty:
            amazon_df = self.validate_data_quality(amazon_df, "Amazon")
            dfs_to_combine.append(amazon_df)
        
        if not flipkart_df.empty:
            flipkart_df = self.validate_data_quality(flipkart_df, "Flipkart")
            dfs_to_combine.append(flipkart_df)
        
        # Combine all data
        if dfs_to_combine:
            self.unified_data = pd.concat(dfs_to_combine, ignore_index=True)
            self.unified_data = self.unified_data.sort_values('Date').reset_index(drop=True)
            
            logger.info(f"Unified dataset created: {len(self.unified_data)} total records")
            logger.info(f"Date range: {self.unified_data['Date'].min()} to {self.unified_data['Date'].max()}")
            logger.info(f"Unique SKUs: {self.unified_data['SKU'].nunique()}")
            logger.info(f"Platforms: {self.unified_data['Platform'].value_counts().to_dict()}")
            
            return self.unified_data
        else:
            logger.error("No valid data sources found")
            return pd.DataFrame()
    
    def save_unified_data(self, filename: str = "unified_sales_data.csv") -> Path:
        """Save unified data to processed directory"""
        if self.unified_data is None or self.unified_data.empty:
            logger.error("No unified data to save")
            return None
        
        output_path = self.processed_dir / filename
        self.unified_data.to_csv(output_path, index=False)
        logger.info(f"Unified data saved to: {output_path}")
        
        return output_path
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of unified data"""
        if self.unified_data is None or self.unified_data.empty:
            return {}
        
        summary = {
            'total_records': len(self.unified_data),
            'date_range': {
                'start': self.unified_data['Date'].min().strftime('%Y-%m-%d'),
                'end': self.unified_data['Date'].max().strftime('%Y-%m-%d')
            },
            'unique_skus': self.unified_data['SKU'].nunique(),
            'platforms': self.unified_data['Platform'].value_counts().to_dict(),
            'total_revenue': self.unified_data['Amount'].sum(),
            'total_quantity': self.unified_data['Quantity'].sum(),
            'avg_order_value': self.unified_data['Amount'].mean(),
            'top_skus': self.unified_data.groupby('SKU')['Amount'].sum().nlargest(10).to_dict()
        }
        
        return summary

def main():
    """Main function to run data coordination"""
    data_dir = Path("/Users/tanayshah/Desktop/personal/projects/ecom_analysis/data")
    
    coordinator = DataCoordinator(data_dir)
    unified_data = coordinator.unify_all_data()
    
    if not unified_data.empty:
        # Save unified data
        output_path = coordinator.save_unified_data()
        
        # Print summary
        summary = coordinator.get_data_summary()
        print("\n=== Data Unification Summary ===")
        print(f"Total Records: {summary['total_records']:,}")
        print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"Unique SKUs: {summary['unique_skus']:,}")
        print(f"Total Revenue: ₹{summary['total_revenue']:,.2f}")
        print(f"Platforms: {summary['platforms']}")
        
        return coordinator
    else:
        logger.error("Data unification failed")
        return None

if __name__ == "__main__":
    coordinator = main()
