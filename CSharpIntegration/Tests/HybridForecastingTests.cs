using Microsoft.Extensions.Logging;
using SalesForecasting.Models;
using SalesForecasting.Services;
using Xunit;

namespace SalesForecasting.Tests
{
    /// <summary>
    /// Unit tests for the hybrid forecasting system
    /// </summary>
    public class HybridForecastingTests
    {
        private readonly ILogger<PythonModelAdapter> _pythonLogger;
        private readonly ILogger<HybridDemandForecaster> _hybridLogger;

        public HybridForecastingTests()
        {
            var loggerFactory = LoggerFactory.Create(builder => builder.AddConsole());
            _pythonLogger = loggerFactory.CreateLogger<PythonModelAdapter>();
            _hybridLogger = loggerFactory.CreateLogger<HybridDemandForecaster>();
        }

        [Fact]
        public void TestBasicForecastingFlow()
        {
            // Arrange
            var testData = GenerateTestSalesData();
            var pythonScriptPath = "../Scripts/python_prediction_bridge.py";
            var dataDirectory = "../../../data";

            var pythonAdapter = new PythonModelAdapter(pythonScriptPath, dataDirectory, _pythonLogger);
            var hybridForecaster = new HybridDemandForecaster(pythonAdapter, null, _hybridLogger);

            // Act
            var result = hybridForecaster.Forecast(testData, horizon: 30);

            // Assert
            Assert.NotNull(result);
            Assert.Equal(30, result.Forecast.Length);
            Assert.Equal(30, result.LowerBound.Length);
            Assert.Equal(30, result.UpperBound.Length);
            Assert.True(result.Forecast.All(x => x >= 0));
        }

        [Fact]
        public void TestInsufficientDataHandling()
        {
            // Arrange
            var testData = GenerateTestSalesData().Take(3); // Only 3 data points
            var pythonScriptPath = "../Scripts/python_prediction_bridge.py";
            var dataDirectory = "../../../data";

            var pythonAdapter = new PythonModelAdapter(pythonScriptPath, dataDirectory, _pythonLogger);
            var hybridForecaster = new HybridDemandForecaster(pythonAdapter, null, _hybridLogger);

            // Act
            var result = hybridForecaster.Forecast(testData, horizon: 30);

            // Assert
            Assert.NotNull(result);
            Assert.Equal("empty", result.ApproachUsed);
            Assert.Contains("Insufficient", result.DecisionReason);
        }

        [Fact]
        public void TestEmptyDataHandling()
        {
            // Arrange
            var testData = new List<SalesData>();
            var pythonScriptPath = "../Scripts/python_prediction_bridge.py";
            var dataDirectory = "../../../data";

            var pythonAdapter = new PythonModelAdapter(pythonScriptPath, dataDirectory, _pythonLogger);
            var hybridForecaster = new HybridDemandForecaster(pythonAdapter, null, _hybridLogger);

            // Act
            var result = hybridForecaster.Forecast(testData, horizon: 30);

            // Assert
            Assert.NotNull(result);
            Assert.Equal("empty", result.ApproachUsed);
            Assert.All(result.Forecast, x => Assert.Equal(0f, x));
        }

        [Fact]
        public void TestModelOutputValidation()
        {
            // Arrange
            var validOutput = new ModelOutput
            {
                Forecast = new float[] { 10f, 12f, 8f },
                LowerBound = new float[] { 8f, 10f, 6f },
                UpperBound = new float[] { 12f, 14f, 10f },
                ApproachUsed = "individual"
            };

            var invalidOutput = new ModelOutput
            {
                Forecast = new float[] { -5f, 0f, 0f },
                LowerBound = new float[] { -7f, -2f, -2f },
                UpperBound = new float[] { -3f, 2f, 2f },
                ApproachUsed = "individual"
            };

            var pythonScriptPath = "../Scripts/python_prediction_bridge.py";
            var dataDirectory = "../../../data";

            var pythonAdapter = new PythonModelAdapter(pythonScriptPath, dataDirectory, _pythonLogger);
            var hybridForecaster = new HybridDemandForecaster(pythonAdapter, null, _hybridLogger);

            // Use reflection to access private method for testing
            var method = typeof(HybridDemandForecaster).GetMethod("IsValidForecast", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

            // Act & Assert
            var validResult = (bool)method!.Invoke(hybridForecaster, new object[] { validOutput })!;
            var invalidResult = (bool)method.Invoke(hybridForecaster, new object[] { invalidOutput })!;

            Assert.True(validResult);
            Assert.False(invalidResult);
        }

        private List<SalesData> GenerateTestSalesData()
        {
            var data = new List<SalesData>();
            var startDate = DateTime.Now.AddDays(-60);

            for (int i = 0; i < 60; i++)
            {
                data.Add(new SalesData
                {
                    Date = startDate.AddDays(i),
                    Quantity = 10f + (float)(Math.Sin(i * 0.1) * 5) + (float)(new Random().NextDouble() * 2)
                });
            }

            return data;
        }
    }
}
