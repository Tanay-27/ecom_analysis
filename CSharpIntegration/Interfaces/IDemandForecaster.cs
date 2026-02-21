using SalesForecasting.Models;

namespace SalesForecasting.MLModels.Training
{
    /// <summary>
    /// Core interface for demand forecasting implementations
    /// Provides a contract for different forecasting approaches
    /// </summary>
    public interface IDemandForecaster
    {
        /// <summary>
        /// Generates a forecast for a specific SKU based on historical data.
        /// </summary>
        /// <param name="historicalData">List of historical sales records.</param>
        /// <param name="horizon">Number of days to forecast (default 30).</param>
        /// <returns>ModelOutput containing forecast values and confidence intervals.</returns>
        ModelOutput Forecast(IEnumerable<SalesData> historicalData, int horizon = 30);
        
        /// <summary>
        /// Gets the name/version of the forecasting model
        /// </summary>
        string ModelVersion { get; }
        
        /// <summary>
        /// Indicates if the model is ready for forecasting
        /// </summary>
        bool IsReady { get; }
    }
}
