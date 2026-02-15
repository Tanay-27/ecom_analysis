
Masters										
										
a. Item Master with Link to other Software										
b. Itemwise MOQ										
c. Itemwise BOM										
d. GodownMaster										
e. Ecommerce Channels										
f. Organisation Master										
g. UserMaster										
										
a. Item Masters 										
if user wants to sync from Other Software we have to give an option for other Software Database access details or User can Edit /update Delete from this Software also . Reverse sync is not allowed										
Existing Format in Other Software										
	ItemID,ItemName,Amazonsku,Flipkartsku,ASIN,OpQty,OpDate,OpValue,IsCombo,qtyin,qtyout,qtybalance,repairin,repairout,repairbal,IsActive,RepopQty,Group,IsManu,prodqty,rejqty									
										
We can use the Same Fields and Update from the Software daily once based on user input										
										
	ItemID,ItemName,Amazonsku,Flipkartsku,ASIN,OpQty,OpDate,OpValue,IsCombo,qtyin,qtyout,qtybalance,repairin,repairout,repairbal,IsActive,RepopQty,Group,IsManu,prodqty,rejqty									
										
b. ITEM MOQ										
										
Should Contain Order Lead Time, Manufacturing Lead time, Minimum order Qty , and clubbing of items for MOQ										
										
c. Itemwise BOM  can be synced or Edit Delete Update from the Software.  One Master should be given if require sync by default it would be know										
										
Existing Format in Other Software										
	ID,ItemIDManu,ItemID,ItemQty,IsActive									
										
d. GodownMaster										
										
Should Contain Following Fields										
	ID	Channel	GodownNAme	Alias1	Alias2	State	IsActive			
										
Here The Godown May have two Alias will have to Check with Both the Alias to arrive at Godown Name										
										
e. Ecommerce Channel Master										
										
	Should Contain Following Fields									
										
	ID	Channel Name	Type 	Main State	IsActive					
										
Type would be Amazon/Flipkart/Others as Dropdown										
										
f. Organisation Master										
										
You Suggest the Fields										
										
g. UserMaster										
										
You Suggest the Fields										
										
										

# Master Tables Requirements

## Item Master
- **Purpose**: Central repository for product information with integration capabilities.
- **Sync Options**: 
  - Provide option for external software database access details if user wants to sync.
  - Manual CRUD operations (edit/update/delete) within this system.
  - Reverse sync not allowed.
  - Use the same fields and update from external software daily once based on user input.
- **Fields** (based on existing format):
  | Field | Type | Description |
  |-------|------|-------------|
  | ItemID | String | Unique identifier |
  | ItemName | String | Product name |
  | Amazonsku | String | Amazon SKU |
  | Flipkartsku | String | Flipkart SKU |
  | ASIN | String | Amazon ASIN |
  | OpQty | Numeric | Opening quantity |
  | OpDate | Date | Opening date |
  | OpValue | Numeric | Opening value |
  | IsCombo | Boolean | Combo product flag |
  | qtyin/qtyout/qtybalance | Numeric | Inventory quantities |
  | repairin/repairout/repairbal | Numeric | Repair quantities |
  | IsActive | Boolean | Active status |
  | RepopQty | Numeric | Replenishment quantity |
  | Group | String | Product category/group |
  | IsManu | Boolean | Manufactured item flag |
  | prodqty/rejqty | Numeric | Production/rejected quantities |

## Item MOQ (Minimum Order Quantity)
- **Purpose**: Define ordering parameters per item.
- **Requirements**: Order lead time, manufacturing lead time, minimum order quantity, item clubbing for MOQ fulfillment.

## Item BOM (Bill of Materials)
- **Purpose**: Component relationships for manufactured items.
- **Sync Options**: Can be synced or edited/deleted/updated from the software. One master should be given if sync is required; by default, it would be known.
- **Fields** (based on existing format):
  | Field | Type | Description |
  |-------|------|-------------|
  | ID | String | Unique BOM identifier |
  | ItemIDManu | String | Manufactured item ID |
  | ItemID | String | Component item ID |
  | ItemQty | Numeric | Required quantity |
  | IsActive | Boolean | Active status |

## Godown Master
- **Purpose**: Warehouse/godown information with alias support for matching.
- **Requirements**: Support two aliases for flexible godown identification. The godown may have two aliases and will have to check with both aliases to arrive at the godown name.
- **Fields**:
  | Field | Type | Description |
  |-------|------|-------------|
  | ID | String | Unique identifier |
  | Channel | String | Associated channel |
  | GodownName | String | Primary name |
  | Alias1/Alias2 | String | Alternative names |
  | State | String | Location state |
  | IsActive | Boolean | Active status |

## Ecommerce Channel Master
- **Purpose**: Define sales channels and their properties.
- **Fields**:
  | Field | Type | Description |
  |-------|------|-------------|
  | ID | String | Unique identifier |
  | Channel Name | String | Channel name |
  | Type | Dropdown | Amazon/Flipkart/Others |
  | Main State | String | Primary operating state |
  | IsActive | Boolean | Active status |

## Organisation Master
- **Purpose**: Company/organization details.
- **Suggested Fields**:
  | Field | Type | Description |
  |-------|------|-------------|
  | ID | String | Unique identifier |
  | Name | String | Organization name |
  | Address | String | Full address |
  | Contact | String | Phone/email |
  | TaxID | String | Tax registration number |
  | GSTIN | String | GST number |
  | IsActive | Boolean | Active status |

## User Master
- **Purpose**: System users with access control.
- **Suggested Fields**:
  | Field | Type | Description |
  |-------|------|-------------|
  | ID | String | Unique identifier |
  | Name | String | Full name |
  | Email | String | Login email |
  | Role | String | User role (admin/user) |
  | PasswordHash | String | Hashed password |
  | IsActive | Boolean | Active status |
  | CreatedAt | DateTime | Account creation date |

**My Inputs**: 
- Align User Master fields with authentication system (already implemented).
- Add audit fields (created/updated timestamps) to all masters.
- Consider soft deletes (IsActive) for data integrity.
- Ensure field naming consistency (e.g., ItemID vs ID).
