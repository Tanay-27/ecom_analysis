namespace SalesForecasting.Models
{
    /// <summary>
    /// Output structure for forecasting results
    /// Contains forecast values and confidence intervals
    /// </summary>
    public class ModelOutput
    {
        /// <summary>
        /// Array of forecasted values for the requested horizon
        /// </summary>
        public float[] Forecast { get; set; } = Array.Empty<float>();
        
        /// <summary>
        /// Lower confidence bound (e.g., 90% or 95% confidence)
        /// </summary>
        public float[] LowerBound { get; set; } = Array.Empty<float>();
        
        /// <summary>
        /// Upper confidence bound
        /// </summary>
        public float[] UpperBound { get; set; } = Array.Empty<float>();
        
        /// <summary>
        /// Confidence level used (e.g., 0.90 for 90%)
        /// </summary>
        public float ConfidenceLevel { get; set; } = 0.90f;
        
        /// <summary>
        /// Approach used for prediction (individual, category, hybrid)
        /// </summary>
        public string ApproachUsed { get; set; } = string.Empty;
        
        /// <summary>
        /// Reason for approach selection
        /// </summary>
        public string DecisionReason { get; set; } = string.Empty;
        
        /// <summary>
        /// Historical accuracy (MAPE) if available
        /// </summary>
        public float? HistoricalMAPE { get; set; }
    }
}
