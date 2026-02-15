Data Import						
a. Sales		b. Closing Stock			c. Stock Analysis Report	
Amazon		Amazon			Amazon	
Flipkart		Flipkart			Flipkart	
						
						
a. Sales						
It will have Multiple Channels , Each Channels Supply Data in Different Formats and Need adjustments before importing the same						
We need to store the Raw Data in one file and Main Data for Sales should be saved Seperately						
						
Duplicate Checking needs to be introduced for the Same						
Unique ID Would be OrderID+ sku+Statefrom+Date						
						
b. Closing Stock						
						
It will have Multiple Channels , Each Channels Supply Data in Different Formats and Need adjustments before importing the same						
We need to store the Raw Data in one file and Main Data for Stock which should be datewise and should be saved Seperately						
						
c. Stock Analysis Report						
						
It will have Multiple Channels , Each Channels Supply Data in Different Formats and Need adjustments before importing the same						
We need to store the Raw Data in one file and Main Data for Stock which should be datewise and should be saved Seperately						



## Importing Data
Data is of various types:

1. Sales Data
2. Closing Stock
3. Stock Analysis Report


### Sales Data
Data will be imported from multiple channels and each channel will have different formats and need adjustments before importing the same

We need to accept data in raw format and create a parser for each channel that will convert into the format we require to store. Thus keeping the source data intact while maintaining correctness of data in db.


Format for different channels:-

Amazon:

Flipkart:


### Closing Stock
This is the daily update for the inventory for each channel at different locations
This needs to be stored in such a way that we can get the closing stock for each channel at different locations as well as be able to store historical data so as to have data of how the stock moves over time in each location individually.

### Stock Analysis Report
This is the analysis of the stock for each channel at different locations


## Functional Requirements
- **Support Multiple Channels**: Integration for Amazon, Flipkart, and other channels with varying data formats.
- **Data Segregation**: Maintain separate storage for Raw Data and processed Main Data for all imports.
- **Sales Data Management**:
    - Implement channel-specific parsers/adjustments for varying formats.
    - Robust duplicate detection using a composite Unique ID (`OrderID + SKU + StateFrom + Date`).
- **Closing Stock Management**:
    - Date-wise and location-wise (Channel/Godown) inventory tracking.
    - Historical data retention for stock movement analysis over time.
- **Stock Analysis Reporting**:
    - Process and store analysis reports for each channel and location.
    - Distinct storage for raw reporting data vs. processed main data.
