# Inventory & Warehousing Requirements

## Overview
Define how stock is tracked across various stages: Raw Materials, Manufacturing, Personal Godowns, and Channel Godowns (FBA, Flipkart FA, etc.).

## Requirements
- **Location Hierarchy**: Support for multiple states and multiple godowns per state.
- **Stock Types**: Track Raw Material (for manufacturing) and Finished Goods (ready to sell).
- **Movement Tracking**: Log transfers between "Personal Godown" and "Channel Godown."

## Pending Questions
1. **Stock Granularity**: Do we track inventory at the Batch/Lot level or just total SKU quantity per location?
2. **Virtual Stock**: Should we account for "In-Transit" stock (shipped but not yet received by the channel) in the ordering logic?
3. **Bin Management**: Within a godown, do we need to track specific shelf/bin locations, or just the godown totals?
4. **Stock Reconciliation**: How often is the "System Stock" synced with physical/channel stock? (Real-time vs Daily Batch).
5. **Returns Handling**: How do customer returns at Channel Godowns affect the "Closing Stock" for reordering?
6. **Damage/QC**: Do we need a "Damaged" or "QC-Pending" status that excludes items from "Available to Sell" stock?

## Answers
1. Stock Granularity: We need to maintain total SKU quantity per location and overall. But the orders will be tracked at lot level. Since we'll order in chunk on total requirement.
2. Virtual Stock: Maintaining in order books is fine, like reuqirement 100, ordered 80 something like that.
3. Bin Management: We don't need to track specific shelf/bin locations, just the godown totals.
4. Daily updates on stock data ( closing stock updated every day)
5. Returns Handling: Returns at channel godown will be updated in closing stock. Returns at personal godown will be updated in closing stock.
6. For purpose of simplicity we will consider them as sold.
