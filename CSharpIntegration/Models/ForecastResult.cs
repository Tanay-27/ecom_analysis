namespace SalesForecasting.Models
{
    /// <summary>
    /// Domain model for storing forecast results
    /// Used for database storage and UI display
    /// </summary>
    public class ForecastResult
    {
        public string SKU { get; set; } = string.Empty;
        public DateTime ForecastDate { get; set; }
        public float PredictedQuantity { get; set; }
        public float ConfidenceLower { get; set; }
        public float ConfidenceUpper { get; set; }
        public string ModelVersion { get; set; } = string.Empty;
        public string ApproachUsed { get; set; } = string.Empty;
        public float? HistoricalMAPE { get; set; }
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    }
}
