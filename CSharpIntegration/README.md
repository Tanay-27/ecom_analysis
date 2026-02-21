# C# Integration for Python Hybrid Prediction System

This folder contains a complete C# .NET 8.0 integration layer that allows your existing C# application to seamlessly use the Python-based hybrid prediction models while maintaining the `IDemandForecaster` interface contract.

## Architecture Overview

The integration follows a clean architecture pattern with these key components:

### Core Components

1. **Models** - Data contracts matching your existing system
   - `SalesData` - Input historical sales data
   - `ModelOutput` - Forecast results with confidence intervals
   - `ForecastResult` - Domain model for storage/display
   - `ForwardPrediction` - Monthly/quarterly aggregations

2. **Services** - Business logic and adapters
   - `PythonModelAdapter` - Bridges C# and Python systems
   - `HybridDemandForecaster` - Main forecaster with fallback logic

3. **Configuration** - Settings and dependency injection
   - `ForecastingConfiguration` - Configuration model
   - `ServiceCollectionExtensions` - DI registration helpers

4. **Scripts** - Python bridge
   - `python_prediction_bridge.py` - Command-line interface for Python models

## Integration Steps

### 1. Add NuGet Packages

```xml
<PackageReference Include="Microsoft.ML" Version="3.0.1" />
<PackageReference Include="Microsoft.ML.TimeSeries" Version="3.0.1" />
<PackageReference Include="Newtonsoft.Json" Version="13.0.3" />
<PackageReference Include="Python.Runtime" Version="3.0.3" />
```

### 2. Update Configuration

Add to your `appsettings.json`:

```json
{
  "Forecasting": {
    "PythonScriptPath": "path/to/python_prediction_bridge.py",
    "DataDirectory": "path/to/your/data/directory",
    "UseHybridModels": true,
    "EnableFallback": true,
    "DefaultHorizonDays": 30
  }
}
```

### 3. Register Services

In your `Program.cs` or `Startup.cs`:

```csharp
// Replace your existing IDemandForecaster registration with:
services.AddHybridForecasting(configuration);

// Or if you want to keep your existing C# forecaster as fallback:
services.AddHybridForecastingWithFallback<ImprovedDemandForecaster>(configuration);
```

### 4. Use the Service

Your existing code remains unchanged:

```csharp
public class YourService
{
    private readonly IDemandForecaster _forecaster;

    public YourService(IDemandForecaster forecaster)
    {
        _forecaster = forecaster;
    }

    public async Task<ModelOutput> GetForecast(IEnumerable<SalesData> data)
    {
        // This now uses the hybrid Python system automatically
        return _forecaster.Forecast(data, horizon: 30);
    }
}
```

## How It Works

### Decision Flow

1. **Input Validation** - Checks data quality and sufficiency
2. **Python Model Attempt** - Tries hybrid prediction system first
3. **Fallback Logic** - Falls back to C# SSA model if Python fails
4. **Statistical Fallback** - Uses simple statistics as last resort

### Python Integration

The system executes Python scripts as separate processes:

1. **Data Conversion** - C# data → JSON → Python pandas DataFrame
2. **Model Execution** - Runs your hybrid prediction system
3. **Result Parsing** - Python JSON → C# ModelOutput
4. **Error Handling** - Graceful fallback on any failures

### Approach Selection

The Python system intelligently selects:
- **Individual Models** - For SKUs with sufficient historical data
- **Category Models** - For new products or insufficient data
- **Hybrid Blending** - For optimal accuracy combinations

## Features

### Enhanced Output

The `ModelOutput` now includes:
- `ApproachUsed` - Which prediction method was selected
- `DecisionReason` - Why that approach was chosen
- `HistoricalMAPE` - Model accuracy metrics
- `ConfidenceLevel` - Prediction confidence

### Robust Error Handling

- Python environment validation
- Process timeout management
- Automatic fallback chains
- Comprehensive logging

### Performance Optimizations

- Temporary file cleanup
- Process reuse where possible
- Configurable timeouts
- Memory-efficient data transfer

## Testing

Run the example program:

```bash
cd CSharpIntegration/Examples
dotnet run
```

This will:
1. Generate sample historical data
2. Test various forecast horizons
3. Demonstrate error handling
4. Show domain model conversions

## Deployment Considerations

### Python Environment

Ensure your deployment environment has:
- Python 3.8+ installed
- Required Python packages (pandas, scikit-learn, etc.)
- Access to your trained models and data files

### File Paths

Update configuration paths for your deployment:
- `PythonScriptPath` - Absolute path to bridge script
- `DataDirectory` - Path to your data and model files

### Security

- Validate all file paths to prevent directory traversal
- Consider running Python processes in sandboxed environment
- Implement proper logging and monitoring

## Troubleshooting

### Common Issues

1. **Python Not Found**
   - Verify Python is in system PATH
   - Update `PythonExecutable` in configuration

2. **Import Errors**
   - Ensure all Python dependencies are installed
   - Check Python path configuration in bridge script

3. **Model Loading Failures**
   - Verify data directory contains required model files
   - Check file permissions and accessibility

4. **Performance Issues**
   - Increase `PythonTimeoutSeconds` for large datasets
   - Consider caching frequently used predictions

### Logging

Enable debug logging for detailed troubleshooting:

```json
{
  "Logging": {
    "LogLevel": {
      "SalesForecasting": "Debug"
    }
  }
}
```

## Migration Path

### Phase 1: Parallel Testing
- Deploy hybrid system alongside existing forecaster
- Compare results and performance
- Validate accuracy improvements

### Phase 2: Gradual Rollout
- Use hybrid system for specific SKU categories
- Monitor system stability and performance
- Collect feedback and metrics

### Phase 3: Full Migration
- Replace existing forecaster completely
- Remove legacy code and dependencies
- Optimize configuration and performance

This integration maintains full compatibility with your existing C# codebase while providing access to the advanced Python-based hybrid prediction capabilities.
