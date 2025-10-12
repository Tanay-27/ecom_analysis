# 📊 Ecommerce Analysis Dashboard

A comprehensive data-driven predictive system for e-commerce supply chain analysis and forecasting.

## 🚀 Quick Start

### Start the Dashboard
```bash
# Activate virtual environment
source .venv/bin/activate

# Start the dashboard
python3 start_dashboard.py
```

The dashboard will open at: http://localhost:8000

## 📁 Clean Project Structure

```
ecommerce-analysis/
├── src/                     # Source code
│   ├── api/                # FastAPI backend
│   │   └── fastapi_service.py
│   └── dashboard/          # Dashboard files
│       ├── index.html      # Frontend
│       └── start_dashboard.py
├── datasets/               # Data files
│   ├── raw/               # Original data files
│   │   ├── Sales-Table 1.csv
│   │   └── List of sku.csv
│   └── processed/         # Processed data files
│       ├── sales_data_jan_june_2025.csv
│       ├── returns_jan_june_2025.csv
│       └── historical_data_2018_nov2024.csv
├── scripts/               # Analysis scripts
│   ├── analysis/          # Business intelligence
│   └── utilities/         # Helper utilities
├── config/                # Configuration files
├── cache/                 # Cached patterns
├── start_dashboard.py     # Quick start script
└── README.md             # This file
```

## 🎯 Key Features

- **📈 Sales Forecasting**: Monthly predictions for July 2025
- **🔄 Returns Analysis**: Return rate analysis and risk assessment
- **📊 Business Intelligence**: Comprehensive analytics and insights
- **🎨 Interactive Dashboard**: Modern web-based UI with ECharts
- **🤖 ML Pipeline**: Multiple prediction models with ensemble methods
- **📱 Real-time API**: FastAPI service for live predictions

## 📊 Data

The system uses fixed Jan-June 2025 data:

### Sales Data
- **Records**: 63,036 transactions
- **Revenue**: ₹56,415,966
- **Quantity**: 32,159 units
- **SKUs**: 91 unique products
- **Date Range**: January 1, 2025 to June 30, 2025

### Returns Data
- **Records**: 15,482 returns
- **Return Amount**: ₹23,526,714
- **Return Quantity**: 15,508 units
- **Date Range**: January 1, 2025 to June 30, 2025

## 🔧 API Endpoints

- `GET /health` - Health check
- `GET /analysis` - Business analysis
- `GET /skus` - SKU list
- `GET /charts/*` - Chart data
- `POST /predict` - Sales predictions
- `GET /returns/*` - Returns analysis
- `GET /predictions/monthly` - Monthly predictions

## 🎯 Usage

### Dashboard Tabs
1. **📊 Overview**: Sales trends, SKU performance, geographic distribution
2. **🔄 Returns**: Return analysis, risk assessment, temporal patterns
3. **🔮 Predictions**: Monthly predictions for July 2025
4. **📊 Forecasting**: Model performance and accuracy metrics
5. **📋 Analysis**: Business intelligence and recommendations

### Generate Predictions
1. Go to **Predictions** tab
2. Select SKU from dropdown
3. View predictions with confidence scores
4. Check monthly prediction table for all SKUs

## 🔧 Development

### Setup
```bash
# Install dependencies
pip install -e .

# Activate virtual environment
source .venv/bin/activate
```

### Run in Development Mode
```bash
uvicorn src.api.fastapi_service:app --reload --host 0.0.0.0 --port 8000
```

## 📈 Performance

- **All APIs Working**: 19/19 endpoints functional (100% success rate)
- **Response Time**: <1 second for most predictions
- **Data Processing**: Handles 78K+ records efficiently
- **Cache System**: Pre-computed patterns for fast predictions

## 🎉 Success Metrics

- ✅ **Clean Structure**: Organized, professional project layout
- ✅ **No Upload Complexity**: Fixed data, reliable system
- ✅ **All APIs Working**: 100% endpoint success rate
- ✅ **User-Friendly**: Modern web dashboard
- ✅ **Scalable**: Handles large datasets efficiently

---

**Ready to analyze your e-commerce data!** 🚀