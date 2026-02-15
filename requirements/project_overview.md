# Sales Forecasting & Inventory Optimization System  
*(Multi-tenant · Multi-channel · Operations-aware)*

---

## 1. What We’re Building (In Simple Terms)

We are building a **decision-making system for e-commerce sellers** who sell across:
- Multiple channels (Flipkart, Amazon, etc.)
- Multiple states and warehouses
- Complex sourcing and manufacturing workflows

It is a **planning and execution support system** that connects:

**Sales → Inventory → Lead Times → Ordering Decisions**

The platform is **multi-tenant**, meaning:
- Each client’s data, logic, and rules are isolated
- Forecasting and optimization adapt per business, not one-size-fits-all

---

## 2. Typical Client Setup (Mental Model)

A client:
- Sells on **Flipkart + Amazon**
- Has **multiple seller godowns** (channel warehouses) across India
- May also have:
  - A **personal manufacturing plant**
  - A **personal raw-material godown**

Inventory flows like this:

**Raw Material → Personal Plant → Personal Godown → Channel Godown → Customer**

Every movement has **time, cost, and constraints**.

---

## 3. Core Objective

> **Spend inventory in the right place, at the right time, with the least risk and capital lock-in.**

To do this, the system must:
- Predict demand accurately
- Understand lead times end-to-end
- Recommend concrete ordering and transfer actions

---

## 4. Inventory Sourcing Logic

Inventory can be sourced in **two primary ways**:

### 4.1 Ready Product (Outsourced Manufacturing)
- Procure finished goods from external manufacturers
- Lead time: **~60 days**
- Flow:
  - Manufacturer → Seller Godown (Amazon/Flipkart)

### 4.2 In-house Manufacturing
- Import or procure raw materials
- Manufacture in personal plant
- Lead time: **~75 days**
- Flow:
  - Raw Material → Personal Plant → Seller Godown

### 4.3 Lead Time Awareness
The system must calculate **effective lead time** based on:
- Procurement time
- Manufacturing time
- Inter-godown transfer time
- Channel intake time

Lead time is **dynamic**, not a fixed number.

---

## 5. Sales Prediction Engine

### 5.1 Available Data

#### Historical Data (2019–2024)
Used to learn:
- Seasonality
- Long-term growth trends
- Event-driven spikes (sales, festivals, campaigns)

#### Current / New SKU Data
Used when:
- SKUs are newly launched
- Limited or no sales history exists

Estimation here relies on:
- Similar SKUs
- Category-level patterns
- Channel behavior
- Early sales signals

---

### 5.2 Forecast Granularity

Sales must be predicted at **multiple levels simultaneously**:

- **SKU**
- **Channel** (Amazon / Flipkart)
- **State / Region**
- **Seller Godown (storage unit)**

Each SKU may:
- Sell on multiple channels
- Be stocked in multiple warehouses
- Show different demand behavior per region

> **Open Design Question (TBD):**  
> How do we best aggregate and split demand so that:
> - Overall SKU growth is captured correctly
> - Individual warehouse forecasts remain accurate and actionable

---

### 5.3 Forecast Horizon

- Forecast frequency: **Weekly**
- Forecast window: **2.5–3 months**
  - This is mandatory to cover procurement + manufacturing lead times

---

## 6. Inventory & Ordering Decision System

The forecasting engine feeds into an **ordering planner**, which decides *what action to take next*.

### 6.1 Ordering Scenarios

#### 1. Direct Manufacturer Ordering
- Finished goods ordered externally
- Accounts for:
  - Manufacturer lead time
  - Channel-wise demand
  - Current stock at seller godowns

#### 2. Internal Stock Transfers
- Move ready products from personal godown → seller godowns
- Optimized by:
  - Regional demand
  - Transfer cost and time
  - Risk of stockouts

#### 3. Manufacture + Ship
- Order raw material
- Manufacture in plant
- Ship finished goods to seller godowns
- Accounts for:
  - Raw material availability
  - Manufacturing capacity
  - Production timelines

#### 4. Manufacture Using Existing Raw Material
- If raw material already exists:
  - Procurement time = 0
  - Only manufacturing + shipping time applies
- System should prefer this path when feasible

---

## 7. Output the User Actually Cares About

The system should not just show charts.

It should clearly say:
- **What to order**
- **How much to order**
- **From where**
- **When**
- **For which channel and godown**
- **What happens if they delay or skip**

In short:
> “If you do nothing, this SKU will stock out in X days in Y locations.”

---

## 8. Why This System Matters

Without this system, clients rely on:
- Gut feel
- Static Excel sheets
- Disconnected reports per channel

With this system, they get:
- Predictive clarity
- Lower inventory holding cost
- Fewer stockouts
- Faster reaction to demand changes

---

## 9. North Star

> **Every inventory decision should be explainable, forecast-backed, and timed correctly.**

This platform exists to make that possible.
