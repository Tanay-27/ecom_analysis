namespace SalesForecasting.Configuration
{
    /// <summary>
    /// Configuration settings for the forecasting system
    /// </summary>
    public class ForecastingConfiguration
    {
        public const string SectionName = "Forecasting";

        /// <summary>
        /// Path to the Python prediction script
        /// </summary>
        public string PythonScriptPath { get; set; } = string.Empty;

        /// <summary>
        /// Path to the data directory containing models and data files
        /// </summary>
        public string DataDirectory { get; set; } = string.Empty;

        /// <summary>
        /// Python executable path (defaults to "python")
        /// </summary>
        public string PythonExecutable { get; set; } = "python";

        /// <summary>
        /// Timeout for Python script execution in seconds
        /// </summary>
        public int PythonTimeoutSeconds { get; set; } = 60;

        /// <summary>
        /// Whether to use hybrid Python models as primary forecasting method
        /// </summary>
        public bool UseHybridModels { get; set; } = true;

        /// <summary>
        /// Whether to enable fallback to C# SSA models
        /// </summary>
        public bool EnableFallback { get; set; } = true;

        /// <summary>
        /// Minimum number of data points required for forecasting
        /// </summary>
        public int MinimumDataPoints { get; set; } = 7;

        /// <summary>
        /// Default forecast horizon in days
        /// </summary>
        public int DefaultHorizonDays { get; set; } = 30;

        /// <summary>
        /// Default confidence level for predictions
        /// </summary>
        public float DefaultConfidenceLevel { get; set; } = 0.90f;
    }
}
