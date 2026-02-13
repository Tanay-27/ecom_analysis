# Task
I want to use the existing model to predict the sales for the Tenant1 database.
TABLE NAME: SALESTRANSACTIONS

DB CONNECTION STRING:
    "Tenant1Database": "Data Source=103.87.173.236,1433;Initial Catalog=Tenant1_SalesDB;User ID=apsr;Password=Apsr@389;Encrypt=False;TrustServerCertificate=True;Connect Timeout=120;ConnectRetryCount=3;ConnectRetryInterval=10;Pooling=False;"


There is a pattern in which i want you to test our model
    
## PROMPT: E-Commerce Demand Forecasting Engine

### 1. Objective
Predict SKU-level sales for a rolling 60-day horizon (Lead Time: 45-60 days). The system must bridge historical seasonal intelligence with current operational velocity while ignoring a structural business break in Nov-Dec 2024.

### 2. Data Logic & Segmentation
- **Segment A (Historical: 2019 - Oct 2024):** - Purpose: Extract **Seasonal Multipliers** and **Category Growth Benchmarks**.
    - Action: Aggregate data by 'Product Category'. Calculate monthly seasonality indexes. Ignore individual SKU IDs as they are deprecated.
- **Segment B (The Break: Nov 2024 - Dec 2024):** - Action: **HARD EXCLUSION.** Do not calculate trends across this gap to prevent the model from learning a false "crash and recovery" slope.
- **Segment C (Current: Jan 2025 - Present):** - Purpose: Establish **Base Velocity** and **Current SKU Growth**.
    - Action: This is the primary training set for active SKU IDs.

### 3. Feature Engineering Requirements
- **Category Bridge:** Map all 2025 SKUs to their 2019-2024 Category counterparts. Apply the 'Seasonal Multipliers' from Segment A as a feature to Segment C.
- **Temporal Features:** Include Month-of-year, Payday cycles (e.g., 1st and 15th), and Days-since-restart (starting Jan 1, 2025).

### 4. Testing & Optimization Protocol
- **Backtesting (Walk-Forward):** - Window 1: Train Jan-June 2025 -> Predict July (Compare vs. Actual).
    - Window 2: Train Feb-July 2025 -> Predict August (Compare vs. Actual).
    - Continue monthly through Dec 2025.
- **Chunk Size Sensitivity Test:** - Iteratively test training lookback windows of 5, 6, 7, 8, and 9 months. 
    - Identify the window size that minimizes **Weighted MAPE (Mean Absolute Percentage Error)** for top-tier SKUs.
- **Multi-Step Strategy:** - Implement **Direct Forecasting** for the 60-day horizon (T+30 and T+60 predicted as independent targets). 
    - Avoid recursive forecasting to prevent error compounding from the volatile 2025 restart period.

### 5. Success Metrics
- **Primary:** MAPE < 15% for Top 20% SKUs (Pareto).
- **Secondary:** Directional Accuracy (predicting the correct 'up' or 'down' trend) > 80%.
