# Data Directory Structure

## Overview
This directory contains all datasets used for the e-commerce analytics dashboard project. Data is organized into three main categories: raw, processed, and external.

## Directory Structure

```
data/
├── raw/                    # Original, unmodified source data
├── processed/              # Cleaned and transformed datasets
├── external/               # Third-party or supplementary data
└── README.md              # This documentation file
```

## Raw Data Files

### 1. SKU Master Data
**File**: `raw/sku_list.csv`
- **Description**: Complete product catalog with SKU specifications
- **Size**: ~6KB
- **Records**: Product inventory list
- **Key Fields**: SKU codes, product names, categories
- **Source**: Internal product management system
- **Last Updated**: Historical snapshot

### 2. Historical Sales Data
**File**: `raw/sales_data_historical.csv`
- **Description**: Complete sales transaction history (2018-2024)
- **Size**: ~85MB (compressed)
- **Records**: ~85M+ transactions
- **Key Fields**: Date, SKU, Quantity, Revenue, Customer segments
- **Source**: E-commerce transaction system
- **Coverage**: 2018 - November 2024

### 3. MOQ and Lead Time Data
**File**: `raw/moq_leadtime.xlsx`
- **Description**: Supplier constraints and ordering parameters
- **Size**: ~15KB
- **Records**: Per-SKU inventory parameters
- **Key Fields**: SKU, MOQ (Minimum Order Quantity), Lead Time, Supplier info
- **Source**: Procurement management system
- **Last Updated**: Current supplier agreements

## Processed Data Files

### 1. Recent Sales Performance
**File**: `processed/sales_data_jan_june_2025.csv`
- **Description**: Latest sales data for model training and validation
- **Size**: ~7.5MB
- **Records**: January - June 2025 transactions
- **Key Fields**: Cleaned and validated sales records
- **Processing**: Data quality checks, outlier removal, standardization

### 2. Returns Analysis Data
**File**: `processed/returns_jan_june_2025.csv`
- **Description**: Product return patterns and reasons
- **Size**: ~1.8MB
- **Records**: Return transactions with categorized reasons
- **Key Fields**: Return date, SKU, quantity, reason codes, refund amounts
- **Use Case**: Quality analysis and demand adjustment

### 3. Historical Consolidated Data
**File**: `processed/historical_data_2018_nov2024.csv`
- **Description**: Cleaned and consolidated historical dataset
- **Size**: ~82MB
- **Records**: Multi-year sales history with data quality improvements
- **Key Fields**: Standardized schema across all historical periods
- **Processing**: Duplicate removal, data type standardization, missing value handling

## Data Quality Standards

### Validation Rules
- **Date Formats**: ISO 8601 standard (YYYY-MM-DD)
- **SKU Codes**: Alphanumeric, consistent formatting
- **Numeric Fields**: No negative quantities, reasonable price ranges
- **Missing Values**: Documented and handled appropriately

### Data Lineage
- **Raw → Processed**: All transformations documented in processing scripts
- **Version Control**: Data snapshots with timestamps
- **Audit Trail**: Change logs for all data modifications

## Usage Guidelines

### For Data Scientists
- Use `processed/` files for model training and analysis
- Refer to `raw/` files only for data exploration and validation
- Document all feature engineering in the processing pipeline

### For Business Users
- Sales performance reports use `processed/sales_data_jan_june_2025.csv`
- Inventory planning references `raw/moq_leadtime.xlsx`
- Historical trends analysis uses `processed/historical_data_2018_nov2024.csv`

### For Developers
- API endpoints should primarily serve processed data
- Implement caching for frequently accessed datasets
- Use streaming for large file processing (>50MB files)

## Data Refresh Schedule

### Real-time Updates
- **Sales Data**: Daily incremental updates
- **Returns Data**: Weekly batch processing
- **Inventory Levels**: Real-time via API integration

### Periodic Updates
- **SKU Master**: Monthly or on product catalog changes
- **MOQ/Lead Times**: Quarterly or on supplier agreement updates
- **Historical Consolidation**: Monthly aggregation of recent data

## Security & Privacy

### Access Controls
- **Raw Data**: Restricted to data engineers and analysts
- **Processed Data**: Available to business users and applications
- **External Data**: Governed by third-party agreements

### Data Protection
- **PII Handling**: Customer identifiers anonymized in processed datasets
- **Backup Strategy**: Daily backups with 30-day retention
- **Encryption**: All files encrypted at rest and in transit

## Contact Information

For questions about data structure, quality, or access:
- **Data Engineering**: [Contact details]
- **Business Intelligence**: [Contact details]
- **Data Governance**: [Contact details]

---
*Last Updated: December 2025*
*Version: 1.0*
