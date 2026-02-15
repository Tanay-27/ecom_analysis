# API & Integration Specifications

## Overview
The system must ingest data from Amazon/Flipkart and communicate internal decisions between the C# Core and Python Inference scripts.

## Requirements
- **Channel Sync**: Automated fetching of Sales and Closing Stock.
- **Bridge Strategy**: C# to call Python script via Subprocess for forecasting.
- **ERP Integration**: Ability to export "Manufacturing Orders" to existing systems.

## Pending Questions
1. **API Credentials**: Will the client provide SP-API (Amazon) / Seller API (Flipkart) credentials directly?
2. **Historical Backfill**: For a new tenant, how many months/years of historical data should we attempt to pull initially?
3. **Data Format**: Are the Amazon/Flipkart reports fetched via API, or is there a need to support "Excel Uploads" for manual reporting?
4. **Sync Frequency**: How often should Sales data be pulled? (Hourly, Daily, or 4x a day?)
5. **Failure Handling**: If the Python script fails to return a forecast, should C# use a "Last Known Forecast" or a simple Average?


## Answers
1. API Credentials:  Yes, we will have them in environment variables, your focus needs to be on parsing and saving to db properly.
2. Out of scope for now. We will have a script that can backfill upon manually adding dates.
3. Lets have excel uploads for now, format will be provided. We can make it automated in next phase.
4. Daily sync is fine.
5. Easier to show NA.

