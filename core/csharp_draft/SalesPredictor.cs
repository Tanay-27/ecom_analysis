using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using System.Data;
using System.Data.SqlClient;
using Newtonsoft.Json;

namespace EcomAnalysis.Core
{
    // Data Models
    public class SalesData
    {
        public DateTime Date { get; set; }
        public string OrderId { get; set; }
        public string SKU { get; set; }
        public int Quantity { get; set; }
        public decimal Rate { get; set; }
        public decimal Amount { get; set; }
        public string Destination { get; set; }
    }

    public class SKUPerformance
    {
        public string SKU { get; set; }
        public decimal TotalRevenue { get; set; }
        public int TotalQuantity { get; set; }
        public int TotalOrders { get; set; }
        public double DailyAvgQuantity { get; set; }
        public decimal RevenuePerUnit { get; set; }
        public int DaysActive { get; set; }
    }

    public class SalesPrediction
    {
        public string SKU { get; set; }
        public int PredictedMonthlyQuantity { get; set; }
        public double PredictedDailyAverage { get; set; }
        public double GrowthRate { get; set; }
        public double SeasonalMultiplier { get; set; }
        public string Confidence { get; set; }
        public DateTime PredictionDate { get; set; }
    }

    public class OrderRecommendation
    {
        public string SKU { get; set; }
        public string Urgency { get; set; }
        public string OrderDate { get; set; }
        public int RecommendedQty { get; set; }
        public int MOQ { get; set; }
        public decimal EstimatedCost { get; set; }
        public double DaysRemaining { get; set; }
        public int LeadTimeDays { get; set; }
        public int CurrentStock { get; set; }
        public double DailyDemand { get; set; }
    }

    // Main Sales Predictor Service
    public class SalesPredictorService
    {
        private readonly ILogger<SalesPredictorService> _logger;
        private readonly string _connectionString;

        public SalesPredictorService(ILogger<SalesPredictorService> logger, string connectionString)
        {
            _logger = logger;
            _connectionString = connectionString;
        }

        // Load recent sales data from database
        public async Task<List<SalesData>> LoadRecentSalesDataAsync(DateTime fromDate, DateTime toDate)
        {
            var salesData = new List<SalesData>();
            
            using var connection = new SqlConnection(_connectionString);
            await connection.OpenAsync();
            
            var query = @"
                SELECT Date, OrderId, SKU, Quantity, Rate, Amount, Destination
                FROM SalesData 
                WHERE Date BETWEEN @FromDate AND @ToDate
                ORDER BY Date DESC";
            
            using var command = new SqlCommand(query, connection);
            command.Parameters.AddWithValue("@FromDate", fromDate);
            command.Parameters.AddWithValue("@ToDate", toDate);
            
            using var reader = await command.ExecuteReaderAsync();
            while (await reader.ReadAsync())
            {
                salesData.Add(new SalesData
                {
                    Date = reader.GetDateTime("Date"),
                    OrderId = reader.GetString("OrderId"),
                    SKU = reader.GetString("SKU"),
                    Quantity = reader.GetInt32("Quantity"),
                    Rate = reader.GetDecimal("Rate"),
                    Amount = reader.GetDecimal("Amount"),
                    Destination = reader.IsDBNull("Destination") ? "" : reader.GetString("Destination")
                });
            }
            
            _logger.LogInformation($"Loaded {salesData.Count} sales records from {fromDate:yyyy-MM-dd} to {toDate:yyyy-MM-dd}");
            return salesData;
        }

        // Identify top performing SKUs
        public List<SKUPerformance> IdentifyTopSKUs(List<SalesData> salesData, int topN = 15)
        {
            _logger.LogInformation($"Analyzing top {topN} SKUs from {salesData.Count} sales records");
            
            var skuPerformance = salesData
                .GroupBy(s => s.SKU)
                .Select(g => new SKUPerformance
                {
                    SKU = g.Key,
                    TotalRevenue = g.Sum(s => s.Amount),
                    TotalQuantity = g.Sum(s => s.Quantity),
                    TotalOrders = g.Count(),
                    DaysActive = (g.Max(s => s.Date) - g.Min(s => s.Date)).Days + 1,
                    RevenuePerUnit = g.Sum(s => s.Amount) / Math.Max(g.Sum(s => s.Quantity), 1)
                })
                .Where(p => p.TotalQuantity > 0)
                .ToList();

            // Calculate daily average
            foreach (var perf in skuPerformance)
            {
                perf.DailyAvgQuantity = (double)perf.TotalQuantity / Math.Max(perf.DaysActive, 1);
            }

            var topSKUs = skuPerformance
                .OrderByDescending(p => p.TotalRevenue)
                .Take(topN)
                .ToList();

            _logger.LogInformation($"Top SKUs identified: {string.Join(", ", topSKUs.Take(5).Select(s => s.SKU))}");
            return topSKUs;
        }

        // Simple prediction algorithm (baseline implementation)
        public async Task<List<SalesPrediction>> PredictNextMonthSalesAsync(List<SKUPerformance> topSKUs, List<SalesData> historicalData)
        {
            _logger.LogInformation($"Generating predictions for {topSKUs.Count} SKUs");
            
            var predictions = new List<SalesPrediction>();
            
            foreach (var sku in topSKUs)
            {
                try
                {
                    // Get SKU-specific historical data
                    var skuData = historicalData
                        .Where(s => s.SKU == sku.SKU)
                        .OrderBy(s => s.Date)
                        .ToList();

                    if (skuData.Count < 30) // Need minimum data points
                        continue;

                    // Calculate baseline prediction using moving averages
                    var last30Days = skuData.TakeLast(30).ToList();
                    var avgDailyQuantity = last30Days.Average(s => s.Quantity);

                    // Calculate growth trend
                    double growthRate = 0;
                    if (skuData.Count >= 60)
                    {
                        var recent30 = skuData.TakeLast(30).Average(s => s.Quantity);
                        var previous30 = skuData.Skip(skuData.Count - 60).Take(30).Average(s => s.Quantity);
                        growthRate = previous30 > 0 ? (recent30 - previous30) / previous30 : 0;
                    }

                    // Apply growth trend
                    var predictedDaily = avgDailyQuantity * (1 + growthRate);
                    var predictedMonthly = (int)(predictedDaily * 30);

                    // Simple seasonality adjustment
                    var nextMonth = DateTime.Now.AddMonths(1).Month;
                    var seasonalMultiplier = GetSeasonalMultiplier(skuData, nextMonth);
                    predictedMonthly = (int)(predictedMonthly * seasonalMultiplier);

                    // Determine confidence level
                    var confidence = Math.Abs(growthRate) < 0.2 ? "Medium" : "Low";
                    if (skuData.Count > 100 && Math.Abs(growthRate) < 0.1)
                        confidence = "High";

                    predictions.Add(new SalesPrediction
                    {
                        SKU = sku.SKU,
                        PredictedMonthlyQuantity = Math.Max(0, predictedMonthly),
                        PredictedDailyAverage = Math.Max(0, predictedMonthly / 30.0),
                        GrowthRate = growthRate,
                        SeasonalMultiplier = seasonalMultiplier,
                        Confidence = confidence,
                        PredictionDate = DateTime.Now
                    });

                    _logger.LogDebug($"Prediction for {sku.SKU}: {predictedMonthly} units/month ({confidence} confidence)");
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, $"Error predicting sales for SKU: {sku.SKU}");
                }
            }

            _logger.LogInformation($"Generated {predictions.Count} predictions");
            return predictions;
        }

        // Calculate seasonal multiplier
        private double GetSeasonalMultiplier(List<SalesData> skuData, int targetMonth)
        {
            var monthlyAvg = skuData
                .GroupBy(s => s.Date.Month)
                .ToDictionary(g => g.Key, g => g.Average(s => s.Quantity));

            var overallAvg = skuData.Average(s => s.Quantity);

            if (monthlyAvg.ContainsKey(targetMonth) && overallAvg > 0)
                return monthlyAvg[targetMonth] / overallAvg;

            return 1.0; // No adjustment if no data
        }

        // Generate ordering recommendations
        public List<OrderRecommendation> GenerateOrderingRecommendations(
            List<SalesPrediction> predictions, 
            Dictionary<string, int> currentInventory,
            Dictionary<string, (int MOQ, int LeadTime)> moqData)
        {
            _logger.LogInformation("Generating ordering recommendations");
            
            var recommendations = new List<OrderRecommendation>();

            foreach (var prediction in predictions)
            {
                try
                {
                    var currentStock = currentInventory.GetValueOrDefault(prediction.SKU, 0);
                    var (moq, leadTime) = moqData.GetValueOrDefault(prediction.SKU, (50, 14));
                    
                    // Calculate safety stock (4 days of demand)
                    var safetyStock = (int)(prediction.PredictedDailyAverage * 4);
                    
                    // Calculate reorder point
                    var reorderPoint = (int)(prediction.PredictedDailyAverage * leadTime) + safetyStock;
                    
                    // Calculate days remaining
                    var daysRemaining = prediction.PredictedDailyAverage > 0 
                        ? currentStock / prediction.PredictedDailyAverage 
                        : 999;
                    
                    // Determine if reorder is needed
                    var needsReorder = currentStock <= reorderPoint;
                    
                    if (needsReorder)
                    {
                        // Calculate order quantity
                        var targetStock = (int)(prediction.PredictedDailyAverage * (leadTime + 30)); // 30 days buffer
                        var orderQuantity = Math.Max(moq, targetStock - currentStock);
                        
                        // Determine urgency
                        string urgency;
                        string orderDate;
                        
                        if (daysRemaining <= leadTime)
                        {
                            urgency = "CRITICAL";
                            orderDate = "URGENT - Order Today";
                        }
                        else if (daysRemaining <= leadTime + 3)
                        {
                            urgency = "HIGH";
                            orderDate = "Order within 3 days";
                        }
                        else
                        {
                            urgency = "MEDIUM";
                            var daysToOrder = Math.Max(1, (int)(daysRemaining - leadTime - 2));
                            orderDate = $"Order in {daysToOrder} days";
                        }
                        
                        recommendations.Add(new OrderRecommendation
                        {
                            SKU = prediction.SKU,
                            Urgency = urgency,
                            OrderDate = orderDate,
                            RecommendedQty = orderQuantity,
                            MOQ = moq,
                            EstimatedCost = orderQuantity * 100, // Assuming ₹100 per unit
                            DaysRemaining = daysRemaining,
                            LeadTimeDays = leadTime,
                            CurrentStock = currentStock,
                            DailyDemand = prediction.PredictedDailyAverage
                        });
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, $"Error generating recommendation for SKU: {prediction.SKU}");
                }
            }

            var orderedRecommendations = recommendations
                .OrderBy(r => r.Urgency == "CRITICAL" ? 1 : r.Urgency == "HIGH" ? 2 : 3)
                .ThenBy(r => r.DaysRemaining)
                .ToList();

            _logger.LogInformation($"Generated {recommendations.Count} ordering recommendations");
            return orderedRecommendations;
        }

        // Save predictions to database
        public async Task SavePredictionsAsync(List<SalesPrediction> predictions)
        {
            using var connection = new SqlConnection(_connectionString);
            await connection.OpenAsync();
            
            // Clear existing predictions
            await new SqlCommand("DELETE FROM SalesPredictions WHERE PredictionDate < @CutoffDate", connection)
                { Parameters = { new SqlParameter("@CutoffDate", DateTime.Now.AddDays(-30)) } }
                .ExecuteNonQueryAsync();
            
            // Insert new predictions
            foreach (var prediction in predictions)
            {
                var query = @"
                    INSERT INTO SalesPredictions 
                    (SKU, PredictedMonthlyQuantity, PredictedDailyAverage, GrowthRate, 
                     SeasonalMultiplier, Confidence, PredictionDate)
                    VALUES 
                    (@SKU, @PredictedMonthlyQuantity, @PredictedDailyAverage, @GrowthRate, 
                     @SeasonalMultiplier, @Confidence, @PredictionDate)";
                
                using var command = new SqlCommand(query, connection);
                command.Parameters.AddWithValue("@SKU", prediction.SKU);
                command.Parameters.AddWithValue("@PredictedMonthlyQuantity", prediction.PredictedMonthlyQuantity);
                command.Parameters.AddWithValue("@PredictedDailyAverage", prediction.PredictedDailyAverage);
                command.Parameters.AddWithValue("@GrowthRate", prediction.GrowthRate);
                command.Parameters.AddWithValue("@SeasonalMultiplier", prediction.SeasonalMultiplier);
                command.Parameters.AddWithValue("@Confidence", prediction.Confidence);
                command.Parameters.AddWithValue("@PredictionDate", prediction.PredictionDate);
                
                await command.ExecuteNonQueryAsync();
            }
            
            _logger.LogInformation($"Saved {predictions.Count} predictions to database");
        }
    }
}
