# Sales Prediction System Documentation

## Overview

The Sales Prediction System is a modular component designed to forecast future product demand based on historical sales data. It is integrated into the Stock Ordering Analysis application, providing insights for inventory management and stock ordering.

Current Implementation: **C# .NET 8.0** using **ML.NET**.

## Technology Stack

*   **Language**: C# (NET 8.0)
*   **Machine Learning Framework**: ML.NET (`Microsoft.ML`, `Microsoft.ML.TimeSeries`)
*   **Algorithm**: Singular Spectrum Analysis (SSA) for Time Series Forecasting.
*   **Architecture**: Clean Architecture with a dedicated service layer for forecasting.

## Architecture & Integration

The system follows a defined interface pattern, making it easy to swap or upgrade the underlying forecasting logic.

### Interface Definition

The core entry point for any forecasting logic is the `IDemandForecaster` interface.

```csharp
namespace SalesForecasting.MLModels.Training
{
    public interface IDemandForecaster
    {
        /// <summary>
        /// Generates a forecast for a specific SKU based on historical data.
        /// </summary>
        /// <param name="historicalData">List of historical sales records.</param>
        /// <param name="horizon">Number of days to forecast (default 30).</param>
        /// <returns>ModelOutput containing forecast values and confidence intervals.</returns>
        ModelOutput Forecast(IEnumerable<SalesData> historicalData, int horizon = 30);
    }
}
```

To integrate your Python-based model steps, you would implement this interface in a new class (e.g., `PythonModelAdapter` or `EnsembleForecaster`) and register it in the Dependency Injection container.

## Data Structures

The system relies on strict data contracts for input and output.

### 1. Input Data (`SalesData`)

The input to the forecaster is a simple time-series of daily sales quantities.

**C# Class:**
```csharp
public class SalesData
{
    public DateTime Date { get; set; }
    public float Quantity { get; set; }
}
```

**JSON Representation:**
```json
[
  {
    "Date": "2025-01-01T00:00:00",
    "Quantity": 15.0
  },
  {
    "Date": "2025-01-02T00:00:00",
    "Quantity": 8.0
  }
]
```

### 2. Output Data (`ModelOutput`)

The output contains parallel arrays for the forecasted values and their confidence bounds.

**C# Class:**
```csharp
public class ModelOutput
{
    // Array of forecasted public float values for the requested horizon
    public float[] Forecast { get; set; }
    
    // Lower confidence bound (e.g., 90% or 95% confidence)
    public float[] LowerBound { get; set; }
    
    // Upper confidence bound
    public float[] UpperBound { get; set; }
}
```

**JSON Representation (Horizon = 3):**
```json
{
  "Forecast": [12.5, 13.1, 11.8],
  "LowerBound": [10.2, 11.0, 9.5],
  "UpperBound": [14.8, 15.2, 14.1]
}
```

### 3. Application Domain Models

Once the forecast is generated, it is often mapped to domain models for storage or display.

**Forecast Result (Storage/UI):**
```csharp
public class ForecastResult
{
    public string SKU { get; set; }
    public DateTime ForecastDate { get; set; }
    public float PredictedQuantity { get; set; }
    public float ConfidenceLower { get; set; }
    public float ConfidenceUpper { get; set; }
    public string ModelVersion { get; set; }
}
```

**Forward Prediction (Reporting):**
Used for aggregating monthly and quarterly views.
```csharp
public class ForwardPrediction
{
    public string SKU { get; set; }
    public decimal Jan26Predicted { get; set; }
    public decimal Feb26Predicted { get; set; }
    public decimal Mar26Predicted { get; set; }
    public decimal Q1_2026_Total { get; set; }
    public decimal ConfidenceScore { get; set; }
}
```

## Forecasting Logic (Current Implementation)

The current `ImprovedDemandForecaster` implements the following pipeline:

1.  **Data Cleaning**:
    *   Removes outliers using the Interquartile Range (IQR) method.
    *   Replaces outliers with the median value.

2.  **Feature Engineering**:
    *   Adds temporal features: Day of Week, Week of Year, Payday Cycles (1st-7th, 15th-21st).

3.  **Adaptive Aggregation**:
    *   **Stable Data** (CV < 2.0): Uses **Daily** forecasting.
    *   **Volatile Data** (CV > 2.0): Aggregates to **Weekly** totals involved forecasting, then distributes back to daily using historical day-of-week weights.

4.  **Model**:
    *   Uses **SSA (Singular Spectrum Analysis)** via `ForecastBySsa`.
    *   Window size is dynamic based on data length (min 7, max 30 for daily).
    *   Confidence level: 90%.

5.  **Post-Processing**:
    *   **Safety Caps**: predictions are capped at reasonable multiples of history (e.g., 3x max historical) to prevent runaway exponential forecasts.
    *   **Flooring**: Negative values are clamped to 0.

## Integration Strategy for New Model

To integrate your new model ensemble:

1.  **Create a New Service**: Implement `IDemandForecaster`.
2.  **Map Inputs**: Convert the incoming `IEnumerable<SalesData>` to your model's required input format.
3.  **Execute Model**: Run your Python logic (e.g., via ONNX, pythonnet, or a microservice API).
4.  **Map Outputs**: Convert your model's result into the standard `ModelOutput` arrays.
5.  **Register**: Update the Dependency Injection in `Program.cs` to use your new class instead of `ImprovedDemandForecaster`.
