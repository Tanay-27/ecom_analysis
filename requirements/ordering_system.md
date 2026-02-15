	Ordering System			
	The Organisation can be in Trading ,Trading + Manufacturing , Manufacturing			
	The Company maintains stock at own Godown, Channels Godown			
	Channels Godown can be located in Multiple States and Multiple Godown in Same State			
				
	Rules in Ordering System			
				
1	Based on Sales Prediction for next One Month, Two Months and Three Months depending on LeadTime + Manufacturing Lead Time the Quantity to be Reordered needs to be calculated			
				
2	The Shortfall has to Be Calculated Channelwise/Godownwise, Overall 			
				
	if Stock is lying at own Godown or Raw Material for Sku's are there in Closing Stock then based on ItemBOM 			
				
1	Give Manufacturing Order			
2	Give Stock Transfer order to ChannelGodownwise			
3	Give Details of Order to be given for Procurement in case of Procurement if there is MOQ and there is Group of items for MOQ then the items under MOQ would be given preference on basis of Sales			
				


## Functional Requirements
- **Support Multiple Business Models**: Trading, Manufacturing, and Mixed (Trading + Manufacturing).
- **Multi-Tier Stock Management**: Track stock at own Godowns and Channel Godowns (Multiple States/Multiple Godowns per State).
- **Intelligent Reordering**:
    - Calculate Reorder Quantity based on 1/2/3 month sales predictions.
    - Incorporate Lead Time and Manufacturing Lead Time in calculations.
- **Automated Shortfall Calculation**: Compute shortages at Godown, Channel, and Overall levels.
- **Order Generation Logic**:
    - **Manufacturing Order**: Triggered if Stock/Raw Material is available via ItemBOM.
    - **Stock Transfer Order**: Direct transfer to specific Channel Godowns.
    - **Procurement Order**: Detailed procurement list considering MOQ and MOQ item groups (prioritized by sales).
