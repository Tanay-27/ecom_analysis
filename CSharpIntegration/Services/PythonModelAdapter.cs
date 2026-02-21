using System.Diagnostics;
using System.Text.Json;
using SalesForecasting.Models;
using SalesForecasting.MLModels.Training;

namespace SalesForecasting.Services
{
    /// <summary>
    /// Adapter that integrates Python-based hybrid prediction models with C# application
    /// Implements IDemandForecaster interface to provide seamless integration
    /// </summary>
    public class PythonModelAdapter : IDemandForecaster
    {
        private readonly string _pythonScriptPath;
        private readonly string _dataDirectory;
        private readonly ILogger<PythonModelAdapter>? _logger;

        public string ModelVersion => "Hybrid Python Model v1.0.0";
        public bool IsReady { get; private set; }

        public PythonModelAdapter(string pythonScriptPath, string dataDirectory, ILogger<PythonModelAdapter>? logger = null)
        {
            _pythonScriptPath = pythonScriptPath;
            _dataDirectory = dataDirectory;
            _logger = logger;
            
            // Check if Python environment and models are available
            IsReady = CheckPythonEnvironment();
        }

        public ModelOutput Forecast(IEnumerable<SalesData> historicalData, int horizon = 30)
        {
            try
            {
                if (!IsReady)
                {
                    _logger?.LogError("Python model adapter is not ready");
                    return CreateFallbackOutput(horizon);
                }

                // Convert C# data to Python-compatible format
                var pythonInput = ConvertToPythonFormat(historicalData, horizon);
                
                // Execute Python prediction script
                var pythonOutput = ExecutePythonScript(pythonInput);
                
                // Parse and convert Python output to C# format
                var modelOutput = ParsePythonOutput(pythonOutput, horizon);
                
                _logger?.LogInformation($"Successfully generated forecast using {modelOutput.ApproachUsed} approach");
                
                return modelOutput;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Error during Python model execution");
                return CreateFallbackOutput(horizon);
            }
        }

        private bool CheckPythonEnvironment()
        {
            try
            {
                // Check if Python script exists
                if (!File.Exists(_pythonScriptPath))
                {
                    _logger?.LogWarning($"Python script not found at: {_pythonScriptPath}");
                    return false;
                }

                // Check if data directory exists
                if (!Directory.Exists(_dataDirectory))
                {
                    _logger?.LogWarning($"Data directory not found at: {_dataDirectory}");
                    return false;
                }

                // Test Python environment
                var testProcess = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = "--version",
                    RedirectStandardOutput = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using var process = Process.Start(testProcess);
                process?.WaitForExit();
                
                return process?.ExitCode == 0;
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to check Python environment");
                return false;
            }
        }

        private string ConvertToPythonFormat(IEnumerable<SalesData> historicalData, int horizon)
        {
            var data = historicalData.OrderBy(x => x.Date).ToList();
            
            var pythonInput = new
            {
                historical_data = data.Select(d => new
                {
                    date = d.Date.ToString("yyyy-MM-dd"),
                    quantity = d.Quantity
                }).ToArray(),
                horizon_days = horizon,
                use_hybrid = true
            };

            return JsonSerializer.Serialize(pythonInput, new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            });
        }

        private string ExecutePythonScript(string inputJson)
        {
            var tempInputFile = Path.GetTempFileName();
            var tempOutputFile = Path.GetTempFileName();

            try
            {
                // Write input to temporary file
                File.WriteAllText(tempInputFile, inputJson);

                // Execute Python script
                var processInfo = new ProcessStartInfo
                {
                    FileName = "python",
                    Arguments = $"\"{_pythonScriptPath}\" \"{tempInputFile}\" \"{tempOutputFile}\" \"{_dataDirectory}\"",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                };

                using var process = Process.Start(processInfo);
                if (process == null)
                    throw new InvalidOperationException("Failed to start Python process");

                process.WaitForExit();

                if (process.ExitCode != 0)
                {
                    var error = process.StandardError.ReadToEnd();
                    throw new InvalidOperationException($"Python script failed: {error}");
                }

                // Read output from temporary file
                if (!File.Exists(tempOutputFile))
                    throw new FileNotFoundException("Python script did not generate output file");

                return File.ReadAllText(tempOutputFile);
            }
            finally
            {
                // Clean up temporary files
                if (File.Exists(tempInputFile)) File.Delete(tempInputFile);
                if (File.Exists(tempOutputFile)) File.Delete(tempOutputFile);
            }
        }

        private ModelOutput ParsePythonOutput(string pythonOutput, int horizon)
        {
            try
            {
                using var document = JsonDocument.Parse(pythonOutput);
                var root = document.RootElement;

                var forecast = ParseFloatArray(root.GetProperty("predictions"));
                var lowerBound = ParseFloatArray(root.GetProperty("confidence_lower"));
                var upperBound = ParseFloatArray(root.GetProperty("confidence_upper"));

                // Ensure arrays are the correct length
                if (forecast.Length != horizon)
                {
                    Array.Resize(ref forecast, horizon);
                    Array.Resize(ref lowerBound, horizon);
                    Array.Resize(ref upperBound, horizon);
                }

                return new ModelOutput
                {
                    Forecast = forecast,
                    LowerBound = lowerBound,
                    UpperBound = upperBound,
                    ConfidenceLevel = root.TryGetProperty("confidence_level", out var confLevel) 
                        ? confLevel.GetSingle() : 0.90f,
                    ApproachUsed = root.TryGetProperty("approach_used", out var approach) 
                        ? approach.GetString() ?? "unknown" : "unknown",
                    DecisionReason = root.TryGetProperty("decision_reason", out var reason) 
                        ? reason.GetString() ?? "" : "",
                    HistoricalMAPE = root.TryGetProperty("historical_mape", out var mape) 
                        ? mape.GetSingle() : null
                };
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "Failed to parse Python output");
                throw new InvalidOperationException("Invalid Python model output format", ex);
            }
        }

        private float[] ParseFloatArray(JsonElement element)
        {
            if (element.ValueKind == JsonValueKind.Array)
            {
                return element.EnumerateArray()
                    .Select(x => x.GetSingle())
                    .ToArray();
            }
            return Array.Empty<float>();
        }

        private ModelOutput CreateFallbackOutput(int horizon)
        {
            // Create a simple fallback when Python model fails
            var forecast = new float[horizon];
            var lowerBound = new float[horizon];
            var upperBound = new float[horizon];

            // Use zero forecast as fallback
            for (int i = 0; i < horizon; i++)
            {
                forecast[i] = 0f;
                lowerBound[i] = 0f;
                upperBound[i] = 0f;
            }

            return new ModelOutput
            {
                Forecast = forecast,
                LowerBound = lowerBound,
                UpperBound = upperBound,
                ConfidenceLevel = 0.50f,
                ApproachUsed = "fallback",
                DecisionReason = "Python model unavailable - using fallback",
                HistoricalMAPE = null
            };
        }
    }
}
