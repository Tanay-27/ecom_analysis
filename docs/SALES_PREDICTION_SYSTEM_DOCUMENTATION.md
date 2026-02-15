# Sales Prediction System - Complete Documentation

## Executive Summary

This document outlines the complete sales prediction system developed to address post-business-disruption forecasting challenges. The system achieved a **13.30% median MAPE** on active SKUs through intelligent filtering, confidence scoring, and production-ready architecture.

---

## 1. Problem Statement & Context

### Business Challenge
- **Business Disruption**: Operations halted in November 2024, restarted January 2025
- **Data Challenge**: 12+ months of post-restart data available, but prediction accuracy remained poor (~98% MAPE initially)
- **Objective**: Build production-ready system with realistic accuracy targets (≤32% MAPE)

### Key Constraints
- Mixed data sources (Historical, Amazon, Flipkart, Recent sales)
- Varying SKU lifecycles (continuing, new, discontinued)
- Business volatility post-restart
- Need for confidence scoring and risk management

---

## 2. Solution Development Process

### Phase 1: Data Unification & Analysis
**Steps Taken:**
1. **Multi-source Data Integration** (`data_coordinator.py`)
   - Unified 4 different data formats
   - Standardized schema across platforms
   - Quality validation and deduplication

2. **Business-Aware Analysis** 
   - Identified disruption impact patterns
   - Segmented data: pre-disruption vs post-restart
   - Analyzed SKU lifecycle patterns

### Phase 2: Model Experimentation
**Approaches Tested:**
1. **Complex Ensemble Model** - Initial attempt with ARIMA + RandomForest + XGBoost
2. **Business-Aware Segmentation** - Separate models for continuing/new SKUs
3. **Simplified Trend-Based** - Focus on post-restart patterns only
4. **Deep Analysis** - Investigation of data quality issues

**Key Learning:** Complexity didn't improve accuracy due to data sparsity and business volatility.

### Phase 3: Production System Development
**Final Approach:**
- **Single RandomForest per SKU** with optimized features
- **Intelligent filtering** to focus on predictable SKUs
- **Confidence scoring** for risk management
- **Monthly retraining** capability

---

## 3. Final Production Architecture

### System Components

#### 3.1 Data Processing Pipeline
```
Raw Data Sources → Data Coordinator → Feature Engineering → Model Training → Predictions
```

**Data Sources:**
- Historical sales (2018-Nov 2024)
- Recent sales (Jan-Jun 2025)
- Amazon sales data (Aug, Dec 2025)
- Flipkart sales data

#### 3.2 Core Engine (`sales_prediction_engine.py`)
**Architecture:**
- **Model Type**: RandomForest Regressor per SKU
- **Feature Engineering**: 17 optimized features
- **Scaling**: StandardScaler for feature normalization
- **Persistence**: Model saving/loading for production use

#### 3.3 API Interface (`prediction_api.py`)
**Simple Interface:**
```python
api = PredictionAPI(data_dir)
prediction = api.get_prediction('SKU_NAME', months_ahead=3)
```

---

## 4. Final Production Architecture

### 4.1 Hybrid Prediction System

The production system uses an **intelligent hybrid approach** that combines individual SKU predictions with category-based predictions:

**Three-Tier Architecture**:
1. **Individual SKU Models**: RandomForest per SKU for established products
2. **Category-Level Models**: Aggregate predictions for product categories  
3. **Hybrid Decision Engine**: Intelligent selection between approaches

**Decision Logic Rules**:
- **Individual approach** when: High data quality (>6 months), dominant SKU (>30% category share)
- **Category approach** when: New SKUs, limited data, stable category patterns
- **Hybrid blend** when: Non-dominant SKU in stable category
- **Fallback** when: No suitable approach available

### 4.2 Category-Based Prediction System

**Problem Solved**: Multiple SKUs per product due to different packaging and combinations

**Implementation**:
- **235 SKUs** mapped to **33 product categories** 
- **Category models** trained on aggregated monthly data
- **SKU distribution patterns** analyzed within each category
- **Automatic fallback** for new product launches

**Key Benefits**:
- **New product forecasting** using category patterns
- **Reduced model complexity**: 19 category models vs 90+ individual models
- **Portfolio-level insights** for business planning

### 4.3 Individual SKU Models

**Single Model per SKU**: RandomForest regression with optimized hyperparameters
- **Rationale**: Complex ensembles did not improve accuracy significantly  
- **Benefit**: Simpler, more reliable, easier to maintain and explain
- **Performance**: Achieved target accuracy with reduced complexity

### 4.4 Feature Engineering Pipeline

**17 Optimized Features** across 5 categories:
1. **Lag Features** (3): quantity_lag_1, quantity_lag_2, quantity_lag_3
2. **Rolling Statistics** (6): 3-month and 6-month rolling means and standard deviations
3. **Trend Features** (2): 3-month quantity and amount trends
4. **Seasonal Features** (4): Cyclical encodings for month and quarter
5. **Volume Features** (2): Recent volume stability and average monthly volume

---

## 5. Filtering System

### 5.1 Inactive SKU Filter
**Purpose:** Remove SKUs with no recent sales to improve system accuracy

**Implementation:**
- **Threshold**: No sales in last 3 months
- **Impact**: Reduced from 90 to 47 active SKUs (48% reduction)
- **Result**: Improved median MAPE from 37% to 13.30%

### 5.2 Stability Filter
**Criteria for Training:**
- **Minimum Data**: ≥6 months of post-restart data
- **Volume Threshold**: Reasonable sales volume
- **Quality Check**: Data completeness validation

**Results:**
- **Active SKUs**: 47 identified
- **Trainable SKUs**: 46 models successfully trained
- **Success Rate**: 98%

---

## 6. Confidence Scoring System

### 6.1 Confidence Components
**Multi-factor scoring (0-1 scale):**

1. **Data Quality (40% weight):**
   - Volume consistency (CV-based)
   - Trend stability
   - Data completeness
   - Outlier ratio

2. **Historical Performance (35% weight):**
   - Based on historical MAPE:
     - ≤20%: Score 1.0
     - 21-30%: Score 0.8
     - 31-40%: Score 0.6
     - >40%: Score 0.4

3. **Volume Stability (25% weight):**
   - Based on average monthly volume:
     - ≥100: Score 1.0
     - 50-99: Score 0.8
     - 20-49: Score 0.6
     - <20: Score 0.4

### 6.2 Confidence Levels
- **High (≥0.8)**: Tight intervals (±25%), suitable for critical decisions
- **Medium (0.6-0.8)**: Standard intervals (±35%), regular planning
- **Low (<0.6)**: Wide intervals (±50%), directional planning only

---

## 7. Historical Accuracy Tracking

### 7.1 Accuracy Metrics
**Primary Metrics:**
- **MAPE**: Mean Absolute Percentage Error
- **RMSE**: Root Mean Square Error
- **Volume Accuracy**: Total volume prediction accuracy
- **Directional Accuracy**: Trend direction prediction rate

### 7.2 Validation Approach
- **Time Series Split**: 80% train, 20% test
- **Minimum Test Size**: ≥3 months for validation
- **Cross-validation**: Historical performance tracking

### 7.3 Current Performance
**System-wide Metrics:**
- **Median MAPE**: 13.30%
- **Best MAPE**: 2.68%
- **SKUs ≤30% MAPE**: 31/46 (67%)
- **SKUs ≤20% MAPE**: 27/46 (59%)

---

## 8. Multi-Dimensional Prediction System

### 8.1 Godown/Fulfillment Center Predictions

**Problem Solved:**
The system now provides **godown-specific predictions** to address the critical "what to stock where" challenge, since Amazon and Flipkart have different warehouse locations and fulfillment strategies.

**Implementation:**
- **Godown Identification**: Channel-City combinations (e.g., "Amazon_HYDERABAD", "Flipkart_MUMBAI")
- **111 fulfillment centers** analyzed across all channels
- **Capacity utilization** and growth tracking per godown
- **State-wise service mapping** for each fulfillment center

**Key Features:**
1. **Channel-Godown Mapping**: Automatic identification of fulfillment centers
2. **Capacity Analysis**: Growth rates and utilization patterns per godown
3. **Service Area Mapping**: Which states each godown serves
4. **Inventory Allocation Matrix**: Channel × State × Godown optimization

### 8.2 Complete Dimensional Breakdown

**Three-Dimensional Analysis:**
1. **Channel Predictions**: Amazon vs Flipkart vs Amazon Stores
2. **Geographical Predictions**: State-wise distribution
3. **Godown Predictions**: Fulfillment center allocation

**Inventory Allocation Matrix:**
```
Channel → State → Godown Allocation
├── Amazon → MAHARASHTRA → Amazon_MUMBAI (15.2 units)
├── Amazon → TELANGANA → Amazon_HYDERABAD (13.6 units)
├── Amazon → KARNATAKA → Amazon_BENGALURU (11.1 units)
└── Flipkart → TAMIL NADU → Flipkart_CHENNAI (8.3 units)
```

### 8.3 Input/Output Specifications

#### Input Requirements

**Data Format:**
```
Required Columns:
- Date: datetime (YYYY-MM-DD format)
- SKU: string (product identifier)
- Quantity: integer (units sold)
- Amount: float (sales value)
- Platform: string (Amazon/Flipkart/etc.)
- State: string (delivery state)
- City: string (delivery city - used for godown mapping)
```

#### File Structure
```
data/
├── collected/
│   ├── historical_data_2018_nov2024.csv
│   ├── sales_data_jan_june_2025.csv
│   ├── amazonsalesaug25.csv
│   └── amazonsalesdec.csv
├── processed/
│   └── unified_sales_data.csv
└── models/
    ├── sales_prediction_engine.pkl
    └── multi_dimensional_predictor.pkl
```

### 8.4 Multi-Dimensional Output Specifications

#### Complete Inventory Allocation Response
```json
{
  "sku": "SKU_NAME",
  "total_prediction": 150.0,
  "channel_breakdown": {
    "approach": "pattern_with_growth",
    "predictions": {
      "Amazon": 137.0,
      "Amazon Stores": 13.0,
      "Flipkart Store": 0.0
    },
    "confidence": "Medium",
    "method": "Historical patterns + growth trends"
  },
  "geographical_breakdown": {
    "approach": "geo_model_simplified",
    "predictions": {
      "MAHARASHTRA": 24.5,
      "TELANGANA": 18.6,
      "TAMIL NADU": 17.2,
      "KARNATAKA": 17.0,
      "Others": 72.7
    },
    "confidence": "Medium"
  },
  "godown_breakdown": {
    "approach": "godown_pattern_with_capacity",
    "predictions": {
      "Amazon_HYDERABAD": 13.6,
      "Amazon_BENGALURU": 11.1,
      "Amazon_CHENNAI": 5.6,
      "Amazon_MUMBAI": 4.2
    },
    "confidence": "Medium",
    "method": "Godown patterns + growth + capacity utilization"
  },
  "inventory_allocation_matrix": {
    "Amazon_MAHARASHTRA": {
      "total_allocation": 22.4,
      "godowns": [
        {
          "godown_id": "Amazon_MUMBAI",
          "city": "MUMBAI",
          "allocation": 15.2,
          "percentage": 67.9
        },
        {
          "godown_id": "Amazon_PUNE",
          "city": "PUNE",
          "allocation": 7.2,
          "percentage": 32.1
        }
      ]
    },
    "Amazon_TELANGANA": {
      "total_allocation": 17.0,
      "godowns": [
        {
          "godown_id": "Amazon_HYDERABAD",
          "city": "HYDERABAD",
          "allocation": 17.0,
          "percentage": 100.0
        }
      ]
    }
  },
  "godown_mapping": {
    "Amazon": {
      "HYDERABAD": {
        "godown_id": "Amazon_HYDERABAD",
        "state": "TELANGANA",
        "volume": 1791,
        "serves_states": ["TELANGANA", "ANDHRA PRADESH", "KARNATAKA"]
      },
      "MUMBAI": {
        "godown_id": "Amazon_MUMBAI",
        "state": "MAHARASHTRA",
        "volume": 836,
        "serves_states": ["MAHARASHTRA", "GUJARAT", "GOA"]
      }
    }
  },
  "metadata": {
    "months_ahead": 3,
    "analysis_date": "2026-02-15",
    "allocation_method": "Channel x State x Godown optimization"
  }
}
```

#### Standard SKU Prediction Response
```json
{
  "sku": "SKU_NAME",
  "predictions": [45.2, 47.1, 44.8],
  "lower_bounds": [29.4, 30.6, 29.1],
  "upper_bounds": [61.0, 63.6, 60.5],
  "confidence": {
    "confidence_score": 0.782,
    "confidence_level": "Medium",
    "factors": {
      "data_quality": 0.887,
      "historical_performance": 0.800,
      "volume_stability": 0.600
    }
  },
  "historical_accuracy": {
    "mape": 19.46,
    "rmse": 12.3,
    "months_of_data": 12,
    "expected_tier": "High"
  },
  "sku_metrics": {
    "total_volume": 168,
    "avg_monthly_volume": 14.0,
    "volume_consistency": 0.777,
    "trend_stability": 0.645
  },
  "prediction_metadata": {
    "months_ahead": 3,
    "confidence_interval": "±35%",
    "model_training_date": "2026-02-15"
  }
}
```

---

## 9. API Usage Guide

### 9.1 Hybrid Prediction API
```python
from core.prediction_api import PredictionAPI
from pathlib import Path

# Initialize with hybrid approach (default)
data_dir = Path("/path/to/data")
api = PredictionAPI(data_dir, use_hybrid=True)

# Get hybrid prediction with intelligent approach selection
result = api.get_prediction('ACIB6DS', months_ahead=3)

# Result includes hybrid decision information:
# result['hybrid_info']['approach_used'] = 'individual' | 'category' | 'hybrid_blend'
# result['hybrid_info']['decision_reason'] = 'High quality individual data'

# Use individual-only mode if needed
api_individual = PredictionAPI(data_dir, use_hybrid=False)

# List active SKUs
active_skus = api.get_active_skus()

# System performance
performance = api.get_system_performance()

# Monthly retraining
api.retrain_models()
```

### 9.2 Category-Based Predictions
```python
from core.category_prediction_engine import CategoryPredictionEngine

# Initialize category engine
cat_engine = CategoryPredictionEngine(data_dir)
cat_engine.load_and_prepare_data()
cat_engine.analyze_and_setup()

# Get category-based prediction
result = cat_engine.get_category_based_sku_prediction('CMSM06', months_ahead=3)

# Access category information
category = result['category']
category_share = result['sku_share_in_category']
distribution_stability = result['distribution_stability']
```

### 9.3 Multi-Dimensional Inventory Allocation
```python
from core.multi_dimensional_predictor import MultiDimensionalPredictor

# Initialize multi-dimensional predictor
multi_predictor = MultiDimensionalPredictor(data_dir)
multi_predictor.analyze_and_setup()

# Get complete inventory allocation
total_prediction = 150  # From main prediction API
allocation = multi_predictor.get_inventory_allocation_matrix(
    sku='CMSM06', 
    total_prediction=total_prediction, 
    months_ahead=3
)

# Access different breakdowns
channel_breakdown = allocation['channel_breakdown']
geo_breakdown = allocation['geographical_breakdown']
godown_breakdown = allocation['godown_breakdown']
allocation_matrix = allocation['inventory_allocation_matrix']

# Use for inventory planning
for channel_state, allocation_data in allocation_matrix.items():
    channel, state = channel_state.split('_', 1)
    total_units = allocation_data['total_allocation']
    
    print(f"Allocate {total_units:.1f} units for {channel} in {state}")
    
    for godown in allocation_data['godowns']:
        city = godown['city']
        units = godown['allocation']
        print(f"  └── Stock {units:.1f} units at {city} warehouse")
```

### 9.4 Production Deployment with Hybrid + Multi-Dimensional Analysis
```python
# Complete production workflow with hybrid approach
api = PredictionAPI(data_dir, use_hybrid=True)  # Use hybrid by default
multi_predictor = MultiDimensionalPredictor(data_dir)
multi_predictor.analyze_and_setup()

# Get predictions for all active SKUs
inventory_plan = {}

for sku in api.get_active_skus():
    # Get hybrid prediction with approach selection
    prediction = api.get_prediction(sku, months_ahead=3)
    
    if 'error' not in prediction:
        total_pred = prediction['predictions'][0]  # Next month
        confidence = prediction['confidence']['confidence_level']
        approach_used = prediction.get('hybrid_info', {}).get('approach_used', 'unknown')
        
        # Get dimensional breakdown
        allocation = multi_predictor.get_inventory_allocation_matrix(
            sku, total_pred, months_ahead=1
        )
        
        inventory_plan[sku] = {
            'total_prediction': total_pred,
            'confidence': confidence,
            'approach_used': approach_used,
            'channel_breakdown': allocation['channel_breakdown']['predictions'],
            'godown_allocation': allocation['godown_breakdown']['predictions'],
            'allocation_matrix': allocation['inventory_allocation_matrix']
        }

# Use inventory_plan for:
# - Channel-specific procurement
# - Godown-wise stock allocation  
# - State-wise distribution planning
```

### 9.5 Hybrid Approach Decision Making
```python
# Decision framework based on hybrid approach and confidence levels
for sku, plan in inventory_plan.items():
    confidence = plan['confidence']
    approach = plan['approach_used']
    total_pred = plan['total_prediction']
    
    print(f"\n📦 {sku} ({approach} approach):")
    
    if approach == 'individual' and confidence == 'High':
        # High confidence individual predictions - use precise allocation
        safety_margin = 1.1  # 10% safety stock
        print(f"  ✅ High confidence individual model - precise allocation")
        
    elif approach == 'category':
        # Category-based predictions - use broader margins
        safety_margin = 1.3  # 30% safety stock
        print(f"  📊 Category-based prediction - conservative allocation")
        
    elif approach == 'hybrid_blend':
        # Blended predictions - moderate margins
        safety_margin = 1.2  # 20% safety stock
        print(f"  🔀 Hybrid blend - balanced allocation")
        
    else:
        # Fallback or low confidence - high margins
        safety_margin = 1.5  # 50% safety stock
        print(f"  ⚠️  Fallback approach - high safety margins")
    
    # Apply safety margins to godown allocations
    for godown, units in plan['godown_allocation'].items():
        if units > 0:
            final_allocation = units * safety_margin
            print(f"    • {godown}: {final_allocation:.0f} units")
```

---

## 10. Performance Benchmarks

### 10.1 Accuracy Improvement Timeline
| Metric | Before Filtering | After Filtering | Improvement |
|--------|------------------|-----------------|-------------|
| Median MAPE | 37.09% | 13.30% | 64% better |
| SKUs ≤30% MAPE | ~40% | 67% | 68% more |
| Active SKUs | 90 | 47 | Focused scope |
| Model Success Rate | ~60% | 98% | Much higher |

### 10.2 Confidence Distribution
- **High Confidence**: 19 SKUs (41%)
- **Medium Confidence**: 21 SKUs (46%)
- **Low Confidence**: 6 SKUs (13%)

### 10.3 Expected Future Performance
**Timeline Projections:**
- **Months 18-24**: Target 10-15% median MAPE
- **Months 24-36**: Target 8-12% median MAPE
- **Stabilization**: After 24+ months of data

---

## 11. Business Impact & Recommendations

### 11.1 Immediate Benefits
- **Realistic Accuracy**: 13.30% median MAPE vs previous 98%
- **Risk Management**: Confidence-based decision making
- **Focused Scope**: 47 active SKUs vs 90 total
- **Production Ready**: Automated training and prediction

### 11.2 Usage Recommendations
1. **High Confidence SKUs**: Use for critical inventory decisions
2. **Medium Confidence SKUs**: Standard business planning
3. **Low Confidence SKUs**: Directional planning with wide margins
4. **Monthly Retraining**: Recommended for optimal performance
5. **Inactive Monitoring**: Review filtered SKUs quarterly

### 11.3 Success Metrics
- **67% of SKUs** achieve ≤30% MAPE (production target met)
- **59% of SKUs** achieve ≤20% MAPE (excellent performance)
- **98% model training success** rate
- **Automatic inactive filtering** improves system focus
- **111 godown/fulfillment centers** analyzed for inventory allocation
- **3-dimensional predictions** (Channel × State × Godown) for complete inventory planning

### 11.4 Godown/Fulfillment Center Benefits
- **Solves "what to stock where" problem** for different Amazon/Flipkart warehouses
- **Capacity utilization tracking** prevents overallocation to constrained facilities
- **Service area mapping** ensures optimal distribution coverage
- **Growth-adjusted allocation** accounts for expanding/contracting fulfillment centers
- **Channel-specific optimization** recognizes different fulfillment strategies

---

## 12. Execution Guide

### 12.1 System Setup and Installation

#### Prerequisites
```bash
# Python 3.8+ required
python --version

# Install dependencies
pip install pandas numpy scikit-learn openpyxl
```

#### Project Structure
```
ecom_analysis/
├── core/
│   ├── data_coordinator.py           # Data unification
│   ├── sales_prediction_engine.py    # Main prediction engine
│   ├── multi_dimensional_predictor.py # Channel/State/Godown analysis
│   ├── prediction_api.py             # API interface
│   └── data_export_manager.py        # Export & database management
├── data/
│   ├── collected/                    # Raw data files
│   ├── processed/                    # Unified data
│   ├── models/                       # Saved models
│   └── exports/                      # Excel exports
├── run_predictions.py                # Main execution script
└── docs/
    └── SALES_PREDICTION_SYSTEM_DOCUMENTATION.md
```

### 12.2 Running the Complete System

#### Full Pipeline Execution
```bash
# Run complete prediction pipeline
python run_predictions.py --mode full

# Run without Excel export (faster)
python run_predictions.py --mode full --no-export
```

**What this does:**
1. **Data Coordination**: Unifies all data sources
2. **Model Training**: Trains prediction models for all SKUs
3. **Multi-Dimensional Analysis**: Analyzes channels, states, godowns
4. **Prediction Generation**: Creates predictions for all active SKUs
5. **Data Export**: Saves to database and Excel

#### Single SKU Analysis
```bash
# Detailed analysis for specific SKU
python run_predictions.py --mode sku --sku CMSM06

# Example output:
# 🔍 Single SKU Analysis: CMSM06
# 📈 Total Prediction: 45.2 units
# 🎯 Confidence: Medium
# 📊 MAPE: 19.46%
# 📺 Channel Breakdown:
#    • Amazon: 41.3 units (91.4%)
#    • Amazon Stores: 3.9 units (8.6%)
```

#### Fulfillment Center Query
```bash
# Query specific fulfillment center
python run_predictions.py --mode fulfillment --channel Amazon --city HYDERABAD

# Example output:
# 🏭 Fulfillment Center Analysis: Amazon in HYDERABAD
# 📊 Total Predicted Volume: 156.8 units
# 📦 SKUs Handled: 23
# 🔝 Top SKUs:
#    • CMSM06: 13.6 units (8.7%)
#    • LRM02: 12.4 units (7.9%)
```

### 12.3 Data Visualization and Storage

#### Excel Export Structure
The system generates a comprehensive Excel file with 5 sheets:

**Sheet 1: Summary**
- SKU overview with confidence levels
- Top channels and states per SKU
- Analysis dates and metadata

**Sheet 2: Channel_Breakdown**
- Channel-wise predictions for all SKUs
- Volume, percentage, approach, confidence

**Sheet 3: Geographical_Breakdown**
- State-wise predictions for all SKUs
- Regional distribution patterns

**Sheet 4: Fulfillment_Centers**
- Godown-wise predictions
- Channel-City combinations with volumes

**Sheet 5: Inventory_Allocation**
- Complete allocation matrix
- Channel → State → Godown breakdown
- Percentage allocations at each level

#### Database Schema
```sql
-- Main predictions table
sku_predictions (sku, prediction_date, total_prediction, confidence_level, mape)

-- Dimensional breakdowns
channel_predictions (sku, channel, predicted_volume, percentage, confidence)
geo_predictions (sku, state, predicted_volume, percentage, confidence)
fulfillment_predictions (sku, channel, city, godown_id, predicted_volume, percentage)

-- Allocation matrix
inventory_allocation (sku, channel, state, godown_city, godown_allocation, allocation_percentage)
```

### 12.4 Database Queries and Analysis

#### Python Query Interface
```python
from core.data_export_manager import DataExportManager
from pathlib import Path

export_manager = DataExportManager(Path("data"))

# Query by SKU
sku_data = export_manager.query_by_sku('CMSM06')
print(f"SKU: {sku_data['sku']}")
print(f"Channels: {len(sku_data['channel_breakdown'])}")
print(f"Fulfillment Centers: {len(sku_data['fulfillment_centers'])}")

# Query by fulfillment center
fc_data = export_manager.query_by_fulfillment_center('Amazon', 'HYDERABAD')
print(f"Total Volume: {fc_data['total_predicted_volume']}")
print(f"SKUs: {fc_data['sku_count']}")

# Query by channel-state combination
cs_data = export_manager.query_by_channel_state('Amazon', 'MAHARASHTRA')
print(f"Godowns: {len(cs_data['godown_summary'])}")
print(f"Total SKUs: {cs_data['total_skus']}")
```

#### SQL Direct Queries
```sql
-- Top fulfillment centers by volume
SELECT channel, city, SUM(predicted_volume) as total_volume, COUNT(DISTINCT sku) as sku_count
FROM fulfillment_predictions 
GROUP BY channel, city 
ORDER BY total_volume DESC 
LIMIT 10;

-- SKU performance by confidence level
SELECT confidence_level, COUNT(*) as sku_count, AVG(total_prediction) as avg_prediction
FROM sku_predictions 
GROUP BY confidence_level;

-- Channel distribution analysis
SELECT channel, SUM(predicted_volume) as total_volume, 
       SUM(predicted_volume) * 100.0 / (SELECT SUM(predicted_volume) FROM channel_predictions) as percentage
FROM channel_predictions 
GROUP BY channel 
ORDER BY total_volume DESC;
```

### 12.5 Production Deployment Commands

#### Daily Execution
```bash
# Daily prediction update (recommended)
python run_predictions.py --mode full

# Check specific fulfillment center capacity
python run_predictions.py --mode fulfillment --channel Amazon --city MUMBAI
```

#### Monthly Retraining
```python
from core.prediction_api import PredictionAPI
from pathlib import Path

api = PredictionAPI(Path("data"))
api.retrain_models()  # Retrains all models with latest data
```

#### Batch SKU Analysis
```bash
# Create script for multiple SKUs
for sku in CMSM06 LRM02 ACIB6DS; do
    echo "Analyzing $sku..."
    python run_predictions.py --mode sku --sku $sku
done
```

### 12.6 File Locations and Outputs

#### Key Files Generated
```
data/
├── processed/unified_sales_data.csv     # Unified dataset
├── models/sales_prediction_engine.pkl   # Trained models
├── predictions.db                       # SQLite database
└── exports/
    └── fulfillment_predictions_YYYYMMDD_HHMMSS.xlsx
```

#### Log Files
```bash
# View execution logs
tail -f logs/prediction_system.log

# Check model training progress
grep "Training model" logs/prediction_system.log
```

---

## 13. Technical Specifications

### 12.1 System Requirements
- **Python**: 3.8+
- **Key Libraries**: pandas, numpy, scikit-learn
- **Memory**: ~500MB for full model training
- **Storage**: ~50MB for model persistence

### 12.2 Performance Characteristics
- **Training Time**: ~3 minutes for 46 SKUs
- **Prediction Time**: <1 second per SKU
- **Model Size**: ~10MB saved model
- **Scalability**: Linear with number of active SKUs

### 12.3 Monitoring & Maintenance
- **Monthly Retraining**: Automated capability
- **Performance Tracking**: Built-in accuracy monitoring
- **Data Quality**: Automatic validation and filtering
- **Model Persistence**: Automatic save/load functionality

---

## Conclusion

The sales prediction system successfully transformed a 98% MAPE challenge into a production-ready solution with 13.30% median MAPE through intelligent filtering, confidence scoring, and focused architecture. The system provides actionable predictions with risk assessment, enabling data-driven inventory and business planning decisions.

**Key Success Factors:**
1. **Inactive SKU filtering** - Major accuracy breakthrough
2. **Confidence scoring** - Risk-appropriate decision making
3. **Simple architecture** - Robust and maintainable
4. **Production focus** - Real-world usability over complexity

The system is ready for immediate deployment and will continue improving as more post-restart data accumulates.
