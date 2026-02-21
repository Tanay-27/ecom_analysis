namespace SalesForecasting.Models
{
    /// <summary>
    /// Monthly and quarterly aggregated predictions
    /// Used for reporting and business planning
    /// </summary>
    public class ForwardPrediction
    {
        public string SKU { get; set; } = string.Empty;
        public decimal Jan26Predicted { get; set; }
        public decimal Feb26Predicted { get; set; }
        public decimal Mar26Predicted { get; set; }
        public decimal Q1_2026_Total { get; set; }
        public decimal ConfidenceScore { get; set; }
        public string ApproachUsed { get; set; } = string.Empty;
        public DateTime GeneratedAt { get; set; } = DateTime.UtcNow;
    }
}
