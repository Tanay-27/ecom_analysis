using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using SalesForecasting.Extensions;
using SalesForecasting.MLModels.Training;
using SalesForecasting.Models;

namespace SalesForecasting.Examples
{
    /// <summary>
    /// Example program demonstrating how to integrate the hybrid forecasting system
    /// into your existing C# .NET 8.0 application
    /// </summary>
    class Program
    {
        static async Task Main(string[] args)
        {
            // Build configuration
            var configuration = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("Configuration/appsettings.json", optional: false)
                .AddEnvironmentVariables()
                .Build();

            // Build host with services
            var host = Host.CreateDefaultBuilder(args)
                .ConfigureServices((context, services) =>
                {
                    // Add logging
                    services.AddLogging(builder =>
                    {
                        builder.AddConsole();
                        builder.AddDebug();
                    });

                    // Add hybrid forecasting system
                    services.AddHybridForecasting(configuration);
                })
                .Build();

            // Get the forecasting service
            var forecaster = host.Services.GetRequiredService<IDemandForecaster>();
            var logger = host.Services.GetRequiredService<ILogger<Program>>();

            logger.LogInformation("Starting Hybrid Forecasting System Demo");
            logger.LogInformation($"Model Version: {forecaster.ModelVersion}");
            logger.LogInformation($"Model Ready: {forecaster.IsReady}");

            // Generate sample historical data
            var historicalData = GenerateSampleData();
            logger.LogInformation($"Generated {historicalData.Count} sample data points");

            try
            {
                // Test forecasting
                logger.LogInformation("Generating 30-day forecast...");
                var forecast = forecaster.Forecast(historicalData, horizon: 30);

                // Display results
                DisplayForecastResults(forecast, logger);

                // Test different horizons
                logger.LogInformation("\nTesting different forecast horizons:");
                
                var horizons = new[] { 7, 14, 30, 60 };
                foreach (var horizon in horizons)
                {
                    var testForecast = forecaster.Forecast(historicalData, horizon);
                    logger.LogInformation($"  {horizon}-day forecast: {testForecast.Forecast.Sum():F1} total units " +
                                        $"(Approach: {testForecast.ApproachUsed})");
                }

                // Test with insufficient data
                logger.LogInformation("\nTesting with insufficient data:");
                var limitedData = historicalData.Take(3).ToList();
                var limitedForecast = forecaster.Forecast(limitedData, horizon: 30);
                logger.LogInformation($"  Limited data forecast: {limitedForecast.ApproachUsed} - {limitedForecast.DecisionReason}");

                // Convert to domain models
                logger.LogInformation("\nConverting to domain models:");
                var forecastResults = ConvertToForecastResults("SAMPLE_SKU", forecast);
                var forwardPrediction = ConvertToForwardPrediction("SAMPLE_SKU", forecast);

                logger.LogInformation($"  Generated {forecastResults.Count} forecast result records");
                logger.LogInformation($"  Q1 2026 Total: {forwardPrediction.Q1_2026_Total:C}");

            }
            catch (Exception ex)
            {
                logger.LogError(ex, "Error during forecasting demonstration");
            }

            logger.LogInformation("Demo completed. Press any key to exit...");
            Console.ReadKey();
        }

        private static List<SalesData> GenerateSampleData()
        {
            var data = new List<SalesData>();
            var startDate = DateTime.Now.AddDays(-90);
            var random = new Random(42); // Fixed seed for reproducible results

            for (int i = 0; i < 90; i++)
            {
                // Generate realistic sales pattern with trend, seasonality, and noise
                var trend = 0.1 * i; // Slight upward trend
                var seasonality = 5 * Math.Sin(2 * Math.PI * i / 7); // Weekly pattern
                var noise = (random.NextDouble() - 0.5) * 4; // Random variation
                var baseQuantity = 15;

                var quantity = Math.Max(0, baseQuantity + trend + seasonality + noise);

                data.Add(new SalesData
                {
                    Date = startDate.AddDays(i),
                    Quantity = (float)quantity
                });
            }

            return data;
        }

        private static void DisplayForecastResults(ModelOutput forecast, ILogger logger)
        {
            logger.LogInformation("\n=== Forecast Results ===");
            logger.LogInformation($"Approach Used: {forecast.ApproachUsed}");
            logger.LogInformation($"Decision Reason: {forecast.DecisionReason}");
            logger.LogInformation($"Confidence Level: {forecast.ConfidenceLevel:P0}");
            
            if (forecast.HistoricalMAPE.HasValue)
            {
                logger.LogInformation($"Historical MAPE: {forecast.HistoricalMAPE:F2}%");
            }

            logger.LogInformation($"Total Forecast: {forecast.Forecast.Sum():F1} units");
            logger.LogInformation($"Average Daily: {forecast.Forecast.Average():F1} units");
            logger.LogInformation($"Confidence Range: {forecast.LowerBound.Sum():F1} - {forecast.UpperBound.Sum():F1} units");

            // Show first week details
            logger.LogInformation("\nFirst 7 days breakdown:");
            for (int i = 0; i < Math.Min(7, forecast.Forecast.Length); i++)
            {
                var date = DateTime.Now.AddDays(i + 1);
                logger.LogInformation($"  {date:MMM dd}: {forecast.Forecast[i]:F1} " +
                                    $"({forecast.LowerBound[i]:F1} - {forecast.UpperBound[i]:F1})");
            }
        }

        private static List<ForecastResult> ConvertToForecastResults(string sku, ModelOutput forecast)
        {
            var results = new List<ForecastResult>();
            var startDate = DateTime.Now.AddDays(1);

            for (int i = 0; i < forecast.Forecast.Length; i++)
            {
                results.Add(new ForecastResult
                {
                    SKU = sku,
                    ForecastDate = startDate.AddDays(i),
                    PredictedQuantity = forecast.Forecast[i],
                    ConfidenceLower = forecast.LowerBound[i],
                    ConfidenceUpper = forecast.UpperBound[i],
                    ModelVersion = "Hybrid Python Model v1.0.0",
                    ApproachUsed = forecast.ApproachUsed,
                    HistoricalMAPE = forecast.HistoricalMAPE,
                    CreatedAt = DateTime.UtcNow
                });
            }

            return results;
        }

        private static ForwardPrediction ConvertToForwardPrediction(string sku, ModelOutput forecast)
        {
            // Aggregate daily forecasts into monthly predictions
            var startDate = DateTime.Now.AddDays(1);
            var jan26 = forecast.Forecast.Take(31).Sum(); // Approximate month
            var feb26 = forecast.Forecast.Skip(31).Take(28).Sum();
            var mar26 = forecast.Forecast.Skip(59).Take(31).Sum();

            return new ForwardPrediction
            {
                SKU = sku,
                Jan26Predicted = (decimal)jan26,
                Feb26Predicted = (decimal)feb26,
                Mar26Predicted = (decimal)mar26,
                Q1_2026_Total = (decimal)(jan26 + feb26 + mar26),
                ConfidenceScore = (decimal)forecast.ConfidenceLevel,
                ApproachUsed = forecast.ApproachUsed,
                GeneratedAt = DateTime.UtcNow
            };
        }
    }
}
