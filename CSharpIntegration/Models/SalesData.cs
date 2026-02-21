namespace SalesForecasting.Models
{
    /// <summary>
    /// Input data structure for sales forecasting
    /// Represents historical sales data points
    /// </summary>
    public class SalesData
    {
        public DateTime Date { get; set; }
        public float Quantity { get; set; }
    }
}
