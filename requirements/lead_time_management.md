# Lead Time Management Requirements

## Overview
Lead time is the most critical variable in the reordering engine. It must be dynamic and reflect real-world delays.

## Requirements
- **Components**: Breakdown lead time into Procurement, Manufacturing, Transfer, and Channel Intake.
- **Dynamic Calculation**: Ability to adjust lead times based on recent performance (e.g., if a supplier is consistently late).

## Pending Questions
1. **Default Values**: What are the "Standard" lead times for Procurement vs Manufacturing if no historical data exists?
2. **Buffer/Safety Days**: Should the system automatically add a "Safety Buffer" (e.g., +5 days) to the lead time to prevent stockouts?
3. **Holiday/Weekend Awareness**: Should the lead time calculator skip certain days (e.g., Sundays or Public Holidays) for manufacturing/shipping?
4. **Supplier-Specific**: Do different manufacturers for the same SKU have different lead times?
5. **Route-Specific**: Does the "Transfer Lead Time" vary significantly between different State Godowns?


## Answers
1. All values for leads times and processing times will be provided. No DEFAULT values are required.
2. No Buffer/Safety Days are required.
3. No Holiday/Weekend Awareness is required.
4. No Supplier-Specific lead times are required.
5. No Route-Specific lead times are required.
