# UI & Dashboard Requirements

## Overview
The frontend must convert complex forecasting data into actionable business decisions.

## Requirements
- **Action Center**: A screen showing exactly "What to order today."
- **Scenario Simulator**: "What happens if I delay this shipment by 10 days?"
- **Tenant Dashboard**: High-level view of stock health across all godowns.

## Pending Questions
1. **Primary User**: Is the dashboard mainly for the Business Owner (Profit/Loss focus) or the Inventory Manager (Stock/Lead Time focus)?
2. **Alerts**: Do we need Email/WhatsApp notifications for "Critical Stockouts"?
3. **Visualization**: Besides Bar/Line charts, do we need Map views to show regional stock distribution?
4. **Manual Overrides**: Should the user be able to manually "Adjust" a forecast on the screen if they know about an upcoming marketing campaign?
5. **Downloadable Reports**: Which formats are required for export (PDF for orders, Excel for raw data)?

## Answers
1. Business Owner (Profit/Loss focus) but the stock and lead time need equal focus.
2. We can have an option to enable/disable email alerts on weekly basis.
3. Simple bar/line charts are enough. No need for complex visualizations.
4. No manual update for now. 
5. Excel is good.


