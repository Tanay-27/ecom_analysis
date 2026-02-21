using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using SalesForecasting.Configuration;
using SalesForecasting.MLModels.Training;
using SalesForecasting.Services;

namespace SalesForecasting.Extensions
{
    /// <summary>
    /// Extension methods for registering forecasting services in DI container
    /// </summary>
    public static class ServiceCollectionExtensions
    {
        /// <summary>
        /// Registers the hybrid forecasting system with dependency injection
        /// </summary>
        /// <param name="services">Service collection</param>
        /// <param name="configuration">Application configuration</param>
        /// <returns>Service collection for chaining</returns>
        public static IServiceCollection AddHybridForecasting(
            this IServiceCollection services, 
            IConfiguration configuration)
        {
            // Register configuration
            services.Configure<ForecastingConfiguration>(
                configuration.GetSection(ForecastingConfiguration.SectionName));

            // Register Python adapter
            services.AddSingleton<PythonModelAdapter>(serviceProvider =>
            {
                var config = configuration.GetSection(ForecastingConfiguration.SectionName)
                    .Get<ForecastingConfiguration>() ?? new ForecastingConfiguration();
                
                var logger = serviceProvider.GetService<ILogger<PythonModelAdapter>>();
                
                return new PythonModelAdapter(
                    config.PythonScriptPath,
                    config.DataDirectory,
                    logger);
            });

            // Register hybrid forecaster as the primary IDemandForecaster
            services.AddSingleton<IDemandForecaster, HybridDemandForecaster>(serviceProvider =>
            {
                var pythonAdapter = serviceProvider.GetRequiredService<PythonModelAdapter>();
                var logger = serviceProvider.GetRequiredService<ILogger<HybridDemandForecaster>>();
                
                // Note: fallbackForecaster can be null if you don't have an existing C# implementation
                // If you have an existing ImprovedDemandForecaster, register it and inject it here
                IDemandForecaster? fallbackForecaster = null;
                
                return new HybridDemandForecaster(pythonAdapter, fallbackForecaster, logger);
            });

            return services;
        }

        /// <summary>
        /// Registers the hybrid forecasting system with an existing C# fallback forecaster
        /// </summary>
        /// <param name="services">Service collection</param>
        /// <param name="configuration">Application configuration</param>
        /// <returns>Service collection for chaining</returns>
        public static IServiceCollection AddHybridForecastingWithFallback<TFallback>(
            this IServiceCollection services, 
            IConfiguration configuration)
            where TFallback : class, IDemandForecaster
        {
            // Register the fallback forecaster
            services.AddSingleton<TFallback>();

            // Register configuration
            services.Configure<ForecastingConfiguration>(
                configuration.GetSection(ForecastingConfiguration.SectionName));

            // Register Python adapter
            services.AddSingleton<PythonModelAdapter>(serviceProvider =>
            {
                var config = configuration.GetSection(ForecastingConfiguration.SectionName)
                    .Get<ForecastingConfiguration>() ?? new ForecastingConfiguration();
                
                var logger = serviceProvider.GetService<ILogger<PythonModelAdapter>>();
                
                return new PythonModelAdapter(
                    config.PythonScriptPath,
                    config.DataDirectory,
                    logger);
            });

            // Register hybrid forecaster with fallback
            services.AddSingleton<IDemandForecaster, HybridDemandForecaster>(serviceProvider =>
            {
                var pythonAdapter = serviceProvider.GetRequiredService<PythonModelAdapter>();
                var fallbackForecaster = serviceProvider.GetRequiredService<TFallback>();
                var logger = serviceProvider.GetRequiredService<ILogger<HybridDemandForecaster>>();
                
                return new HybridDemandForecaster(pythonAdapter, fallbackForecaster, logger);
            });

            return services;
        }
    }
}
