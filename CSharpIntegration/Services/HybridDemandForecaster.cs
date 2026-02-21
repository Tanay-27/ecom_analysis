using SalesForecasting.Models;
using SalesForecasting.MLModels.Training;

namespace SalesForecasting.Services
{
    /// <summary>
    /// Enhanced demand forecaster that combines Python hybrid models with C# fallback
    /// Implements intelligent model selection and error handling
    /// </summary>
    public class HybridDemandForecaster : IDemandForecaster
    {
        private readonly PythonModelAdapter _pythonAdapter;
        private readonly IDemandForecaster? _fallbackForecaster;
        private readonly ILogger<HybridDemandForecaster> _logger;

        public string ModelVersion => "Hybrid Demand Forecaster v1.0.0";
        public bool IsReady => _pythonAdapter.IsReady || _fallbackForecaster?.IsReady == true;

        public HybridDemandForecaster(
            PythonModelAdapter pythonAdapter, 
            IDemandForecaster? fallbackForecaster,
            ILogger<HybridDemandForecaster> logger)
        {
            _pythonAdapter = pythonAdapter;
            _fallbackForecaster = fallbackForecaster;
            _logger = logger;
        }

        public ModelOutput Forecast(IEnumerable<SalesData> historicalData, int horizon = 30)
        {
            var dataList = historicalData.ToList();
            
            // Validate input data
            if (!dataList.Any())
            {
                _logger.LogWarning("No historical data provided for forecasting");
                return CreateEmptyForecast(horizon);
            }

            if (dataList.Count < 7)
            {
                _logger.LogWarning($"Insufficient data for forecasting: {dataList.Count} records (minimum 7 required)");
                return CreateEmptyForecast(horizon);
            }

            try
            {
                // Try Python hybrid model first
                if (_pythonAdapter.IsReady)
                {
                    _logger.LogInformation("Attempting forecast with Python hybrid model");
                    var pythonResult = _pythonAdapter.Forecast(dataList, horizon);
                    
                    // Validate Python result
                    if (IsValidForecast(pythonResult))
                    {
                        _logger.LogInformation($"Python forecast successful using {pythonResult.ApproachUsed} approach");
                        return pythonResult;
                    }
                    else
                    {
                        _logger.LogWarning("Python model returned invalid forecast, falling back");
                    }
                }
                else
                {
                    _logger.LogWarning("Python model not ready, using fallback");
                }

                // Fallback to C# model if available
                if (_fallbackForecaster?.IsReady == true)
                {
                    _logger.LogInformation("Using C# fallback forecaster");
                    var fallbackResult = _fallbackForecaster.Forecast(dataList, horizon);
                    
                    // Enhance fallback result with metadata
                    fallbackResult.ApproachUsed = "csharp_fallback";
                    fallbackResult.DecisionReason = "Python model unavailable - using C# SSA model";
                    
                    return fallbackResult;
                }

                // Last resort: statistical fallback
                _logger.LogWarning("All forecasting models unavailable, using statistical fallback");
                return CreateStatisticalFallback(dataList, horizon);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error during forecasting process");
                return CreateStatisticalFallback(dataList, horizon);
            }
        }

        private bool IsValidForecast(ModelOutput forecast)
        {
            return forecast.Forecast?.Length > 0 &&
                   forecast.LowerBound?.Length > 0 &&
                   forecast.UpperBound?.Length > 0 &&
                   forecast.Forecast.All(x => x >= 0) &&
                   !forecast.Forecast.All(x => x == 0);
        }

        private ModelOutput CreateEmptyForecast(int horizon)
        {
            return new ModelOutput
            {
                Forecast = new float[horizon],
                LowerBound = new float[horizon],
                UpperBound = new float[horizon],
                ConfidenceLevel = 0.50f,
                ApproachUsed = "empty",
                DecisionReason = "Insufficient historical data",
                HistoricalMAPE = null
            };
        }

        private ModelOutput CreateStatisticalFallback(List<SalesData> historicalData, int horizon)
        {
            // Calculate basic statistics from recent data
            var recentData = historicalData.OrderByDescending(x => x.Date).Take(30).ToList();
            var quantities = recentData.Select(x => x.Quantity).Where(x => x > 0).ToList();

            if (!quantities.Any())
            {
                return CreateEmptyForecast(horizon);
            }

            var mean = quantities.Average();
            var stdDev = quantities.Count > 1 ? CalculateStandardDeviation(quantities) : mean * 0.3f;

            // Create forecast arrays
            var forecast = new float[horizon];
            var lowerBound = new float[horizon];
            var upperBound = new float[horizon];

            for (int i = 0; i < horizon; i++)
            {
                // Apply slight decay over time
                var decayFactor = 1.0f - (i * 0.01f);
                var prediction = Math.Max(0, mean * decayFactor);
                
                forecast[i] = prediction;
                lowerBound[i] = Math.Max(0, prediction - stdDev);
                upperBound[i] = prediction + stdDev;
            }

            return new ModelOutput
            {
                Forecast = forecast,
                LowerBound = lowerBound,
                UpperBound = upperBound,
                ConfidenceLevel = 0.60f,
                ApproachUsed = "statistical_fallback",
                DecisionReason = "All ML models unavailable - using statistical approach",
                HistoricalMAPE = null
            };
        }

        private float CalculateStandardDeviation(List<float> values)
        {
            var mean = values.Average();
            var sumOfSquaredDifferences = values.Sum(x => Math.Pow(x - mean, 2));
            return (float)Math.Sqrt(sumOfSquaredDifferences / values.Count);
        }
    }
}
