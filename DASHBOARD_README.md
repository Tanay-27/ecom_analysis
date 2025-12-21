# E-commerce Analytics Dashboard

A professional, minimalistic web-based analytics dashboard for your e-commerce sales prediction and inventory optimization system.

## Features

### 📊 **Key Metrics Cards**
- **Total Revenue**: Current period revenue with growth indicators
- **Total Orders**: Order count with average order value
- **Active SKUs**: Number of products being tracked
- **Next Month Forecast**: Predicted sales volume

### 📈 **Interactive Charts**
- **Revenue Trend**: Monthly revenue progression with smooth line chart
- **Top Performing SKUs**: Doughnut chart showing revenue distribution
- **Prediction Confidence**: Bar chart showing confidence levels
- **Inventory Health**: Pie chart showing stock status distribution

### 🚨 **Critical Alerts**
- Real-time insights and recommendations
- Color-coded alerts (Critical, Warning, Info, Positive)
- Actionable business intelligence

### 📋 **Data Tables**
- **Critical Orders**: Items that need immediate ordering
- **Sales Predictions**: Next month forecasts with confidence levels
- **Critical Inventory**: Items with dangerously low stock
- **High Priority Orders**: Items to order within 3 days

## How to Use

### 1. **Start the Dashboard**
```bash
# Navigate to the project directory
cd /Users/tanayshah/Desktop/personal/projects/ecom_analysis

# Install Flask if not already installed
pip install Flask==3.0.0

# Start the dashboard server
python dashboard_app.py
```

### 2. **Access the Dashboard**
- Open your web browser
- Navigate to: `http://127.0.0.1:8080`
- The dashboard will automatically load your latest analysis data

### 3. **Dashboard Features**

#### **Real-time Updates**
- Dashboard automatically refreshes every 5 minutes
- Manual refresh: Press `Ctrl+R` (Windows) or `Cmd+R` (Mac)

#### **Keyboard Shortcuts**
- `Ctrl/Cmd + R`: Refresh dashboard
- `Ctrl/Cmd + P`: Print dashboard

#### **Mobile Responsive**
- Fully responsive design works on all devices
- Tables scroll horizontally on mobile devices

### 4. **Understanding the Data**

#### **Confidence Levels**
- 🟢 **High**: Stable demand pattern, reliable predictions
- 🟡 **Medium**: Some volatility, moderate confidence
- 🔴 **Low**: High volatility or limited data, monitor closely

#### **Urgency Levels**
- 🚨 **Critical**: Order immediately (< 7 days stock)
- 🔶 **High Priority**: Order within 3 days (7-14 days stock)
- 🔵 **Medium Priority**: Plan ahead (14+ days stock)

#### **Status Indicators**
- Days remaining until stockout
- Current stock levels vs. reorder points
- Daily demand rates

## Data Sources

The dashboard automatically loads data from:
- `core/analysis_results/next_month_predictions.csv`
- `core/analysis_results/ordering_schedule.csv`
- `core/analysis_results/reorder_analysis.csv`
- `data/processed/sales_data_jan_june_2025.csv`

## Troubleshooting

### **Dashboard Won't Load**
1. Ensure Flask is installed: `pip install Flask==3.0.0`
2. Check if port 8080 is available
3. Verify analysis results files exist in `core/analysis_results/`

### **No Data Showing**
1. Run your analysis pipeline first: `python run_analysis.py`
2. Check that CSV files are generated in `core/analysis_results/`
3. Refresh the browser page

### **Charts Not Displaying**
1. Check browser console for JavaScript errors
2. Ensure internet connection (Chart.js loads from CDN)
3. Try refreshing the page

## Technical Details

### **Technology Stack**
- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **Charts**: Chart.js
- **Icons**: Font Awesome
- **Styling**: Custom CSS with modern design principles

### **File Structure**
```
├── dashboard_app.py          # Flask backend server
├── templates/
│   └── dashboard.html        # Main dashboard template
├── static/
│   ├── css/
│   │   └── dashboard.css     # Professional styling
│   └── js/
│       └── dashboard.js      # Interactive functionality
└── requirements_dashboard.txt # Python dependencies
```

### **API Endpoints**
- `/api/business-overview` - Key metrics and revenue data
- `/api/sales-predictions` - Next month forecasts
- `/api/ordering-recommendations` - What to order and when
- `/api/inventory-status` - Current stock levels
- `/api/key-insights` - Business insights and alerts

## Customization

### **Colors and Styling**
Edit `static/css/dashboard.css` to customize:
- Color palette (CSS variables at top of file)
- Card layouts and spacing
- Chart colors and styling

### **Data Refresh Rate**
Edit `static/js/dashboard.js` line ~25 to change auto-refresh interval:
```javascript
// Current: 5 minutes (300000ms)
setInterval(() => {
    this.refreshData();
}, 300000); // Change this value
```

### **Adding New Metrics**
1. Add API endpoint in `dashboard_app.py`
2. Update HTML template in `templates/dashboard.html`
3. Add JavaScript rendering in `static/js/dashboard.js`

## Support

For issues or questions:
1. Check the browser console for error messages
2. Verify all data files are present and properly formatted
3. Ensure Flask server is running without errors

---

**Dashboard URL**: http://127.0.0.1:8080

**Last Updated**: December 2025
