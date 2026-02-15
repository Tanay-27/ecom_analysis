# Business Rules & Constraints Requirements

## Overview
The "Decision Engine" must operate within fixed business constraints like minimum orders, packaging groups, and capital limits.

## Requirements
- **MOQ (Minimum Order Quantity)**: Support for SKU-level and Supplier-level MOQs.
- **Group MOQs**: Scenarios where a group of SKUs together must meet a weight or value threshold.
- **Priority Logic**: When capital is limited, which SKUs get reordered first?

## Pending Questions
1. **MOQ Priority**: If we have a Group MOQ, do we prioritize "Highest Selling" items or "Lowest Stock" items to fill the gap?
2. **Price Breaks**: Does the system need to account for Tiered Pricing (e.g., 5% discount if ordering >1000 units)?
3. **Manufacturing Capacity**: Is there a maximum limit on how much the "Personal Plant" can produce per week/month?
4. **Auto-Approval**: Should the system automatically "Execute" orders below a certain value, or must every suggestion be manually approved?
5. **Discard Rules**: Are there cases where we should NOT reorder an item even if stock is low (e.g., SKU being phased out/EOL)?


## Answers
1. MOQ Priority: Group MOQ is usually for a combination of same main item (eg. a sewing machine, with extra threads, needles, etc. as variable items across different SKUs).
2. You can skip tiered pricing for now.

### NOTE: For now we will be using this system only as recommendation reference, and manually ordering the items. These below items are beyond scope for first iteration.
3. Manufacturing capacity is not a constraint for now.
4. Auto approval is not a constraint for now.
5. Discard rules are not a constraint for now.
