# System Cleanup and Modularization Plan

## Files to Remove (Unnecessary/Experimental)
- `comprehensive_ordering_analysis.py` - Experimental analysis
- `run_analysis.py` - Old analysis script
- `update_predictions.py` - Redundant with main API
- `core/data_exploration.py` - Experimental exploration
- `core/db_model_tester.py` - Testing file
- `core/sales_predictor.py` - Old predictor (replaced by sales_prediction_engine.py)
- `core/ordering_optimizer.py` - Not part of core prediction system
- `core/business_dashboard.py` - Dashboard logic (separate concern)
- `api_server.py` - Large monolithic server (needs refactoring)
- `start_dashboard.py` - Dashboard starter
- `auth.py` - Authentication (separate concern)
- `database.py` - Simple DB wrapper (separate concern)
- `models.py` - Simple models (separate concern)

## Core Production Files to Keep
- `core/data_coordinator.py` - Data unification
- `core/sales_prediction_engine.py` - Individual SKU predictions
- `core/category_prediction_engine.py` - Category-based predictions
- `core/hybrid_prediction_engine.py` - Hybrid approach
- `core/multi_dimensional_predictor.py` - Channel/State/Godown predictions
- `core/data_export_manager.py` - Export and database management
- `core/prediction_api.py` - Production API
- `run_predictions.py` - Main execution script

## Import Issues to Fix
- Fix relative imports in hybrid_prediction_engine.py
- Fix relative imports in prediction_api.py
- Create proper __init__.py files
- Standardize import patterns

## Structure Improvements
- Create clean main.py entry point
- Add proper error handling
- Standardize logging
- Remove duplicate code
- Add type hints consistently
- Create configuration management
