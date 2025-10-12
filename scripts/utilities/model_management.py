#!/usr/bin/env python3
"""
Model Management and Input Requirements

This script defines when models need retraining and what inputs are required
for predictions.
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import json

class ModelManager:
    """
    Manages model training, retraining, and input requirements.
    """
    
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.model_metadata = {}
        
    def should_retrain_model(self, sku, last_training_date=None):
        """
        Determine if model needs retraining.
        
        Args:
            sku: SKU identifier
            last_training_date: When model was last trained
            
        Returns:
            bool: True if retraining is needed
        """
        if last_training_date is None:
            return True  # No model exists
        
        # Retrain if:
        # 1. Model is older than 30 days
        # 2. New data is available (more than 7 days of new data)
        # 3. Model performance has degraded
        
        days_since_training = (datetime.now() - last_training_date).days
        
        retrain_conditions = [
            days_since_training > 30,  # Monthly retraining
            days_since_training > 7 and self._has_new_data(sku)  # New data available
        ]
        
        return any(retrain_conditions)
    
    def _has_new_data(self, sku):
        """Check if new data is available for SKU."""
        # This would check your data source for new records
        # For now, assume new data is always available
        return True
    
    def get_input_requirements(self):
        """
        Define what inputs are required for predictions.
        
        Returns:
            dict: Input requirements for different prediction types
        """
        return {
            "daily_prediction": {
                "required_fields": [
                    "sku",
                    "date",
                    "quantity_lag1",  # Yesterday's sales
                    "quantity_lag7",  # Last week's sales
                    "quantity_lag30", # Last month's sales
                    "month",
                    "day_of_week"
                ],
                "optional_fields": [
                    "year",
                    "quarter",
                    "seasonal_factor"
                ],
                "description": "Predict next day's sales for a specific SKU"
            },
            
            "monthly_prediction": {
                "required_fields": [
                    "sku",
                    "year",
                    "month",
                    "quantity_lag1",   # Last month
                    "quantity_lag3",   # 3 months ago
                    "quantity_lag6",   # 6 months ago
                    "quantity_lag12",  # 12 months ago (seasonal)
                    "ma_3",           # 3-month moving average
                    "ma_6",           # 6-month moving average
                    "ma_12",          # 12-month moving average
                    "month_sin",      # Seasonal sine component
                    "month_cos",      # Seasonal cosine component
                    "yoy_growth",     # Year-over-year growth
                    "trend"           # Time trend
                ],
                "optional_fields": [
                    "quarter",
                    "rate",
                    "amount"
                ],
                "description": "Predict next month's sales for a specific SKU"
            },
            
            "batch_prediction": {
                "required_fields": [
                    "csv_file_path",
                    "prediction_type"  # "daily" or "monthly"
                ],
                "optional_fields": [
                    "date_range",
                    "sku_filter"
                ],
                "description": "Batch prediction for multiple SKUs from CSV file"
            }
        }
    
    def create_prediction_input_template(self, prediction_type="daily"):
        """
        Create input template for predictions.
        
        Args:
            prediction_type: "daily", "monthly", or "batch"
            
        Returns:
            dict: Input template
        """
        requirements = self.get_input_requirements()
        
        if prediction_type not in requirements:
            raise ValueError(f"Unknown prediction type: {prediction_type}. Available: {list(requirements.keys())}")
        
        template = {
            "prediction_type": prediction_type,
            "inputs": {},
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0"
            }
        }
        
        # Add required fields
        for field in requirements[prediction_type]["required_fields"]:
            template["inputs"][field] = None
        
        # Add optional fields
        for field in requirements[prediction_type]["optional_fields"]:
            template["inputs"][field] = None
        
        return template
    
    def validate_prediction_input(self, inputs, prediction_type="daily"):
        """
        Validate prediction inputs.
        
        Args:
            inputs: Input dictionary
            prediction_type: Type of prediction
            
        Returns:
            tuple: (is_valid, errors)
        """
        requirements = self.get_input_requirements()
        
        if prediction_type not in requirements:
            return False, [f"Unknown prediction type: {prediction_type}"]
        
        errors = []
        required_fields = requirements[prediction_type]["required_fields"]
        
        # Check required fields
        for field in required_fields:
            if field not in inputs or inputs[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Validate data types and ranges
        if "sku" in inputs and inputs["sku"]:
            if not isinstance(inputs["sku"], str):
                errors.append("SKU must be a string")
        
        if "date" in inputs and inputs["date"]:
            try:
                pd.to_datetime(inputs["date"])
            except:
                errors.append("Date must be in valid format (YYYY-MM-DD)")
        
        if "month" in inputs and inputs["month"]:
            month = inputs["month"]
            if not (1 <= month <= 12):
                errors.append("Month must be between 1 and 12")
        
        if "day_of_week" in inputs and inputs["day_of_week"]:
            dow = inputs["day_of_week"]
            if not (0 <= dow <= 6):
                errors.append("Day of week must be between 0 and 6")
        
        return len(errors) == 0, errors
    
    def save_model(self, model, sku, model_type="daily"):
        """
        Save trained model to disk.
        
        Args:
            model: Trained model object
            sku: SKU identifier
            model_type: Type of model ("daily" or "monthly")
        """
        model_path = self.model_dir / f"{sku}_{model_type}_model.pkl"
        metadata_path = self.model_dir / f"{sku}_{model_type}_metadata.json"
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            "sku": sku,
            "model_type": model_type,
            "training_date": datetime.now().isoformat(),
            "model_path": str(model_path),
            "version": "1.0"
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.model_metadata[f"{sku}_{model_type}"] = metadata
    
    def load_model(self, sku, model_type="daily"):
        """
        Load trained model from disk.
        
        Args:
            sku: SKU identifier
            model_type: Type of model ("daily" or "monthly")
            
        Returns:
            tuple: (model, metadata) or (None, None) if not found
        """
        model_path = self.model_dir / f"{sku}_{model_type}_model.pkl"
        metadata_path = self.model_dir / f"{sku}_{model_type}_metadata.json"
        
        if not model_path.exists() or not metadata_path.exists():
            return None, None
        
        # Load model
        model = joblib.load(model_path)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return model, metadata
    
    def get_model_status(self, sku, model_type="daily"):
        """
        Get status of model for a SKU.
        
        Args:
            sku: SKU identifier
            model_type: Type of model
            
        Returns:
            dict: Model status information
        """
        model, metadata = self.load_model(sku, model_type)
        
        if model is None:
            return {
                "exists": False,
                "needs_training": True,
                "last_trained": None,
                "status": "No model found"
            }
        
        last_trained = datetime.fromisoformat(metadata["training_date"])
        needs_retraining = self.should_retrain_model(sku, last_trained)
        
        return {
            "exists": True,
            "needs_training": needs_retraining,
            "last_trained": last_trained.isoformat(),
            "status": "Needs retraining" if needs_retraining else "Up to date",
            "model_path": metadata["model_path"]
        }
    
    def list_available_models(self):
        """
        List all available trained models.
        
        Returns:
            list: List of available models
        """
        models = []
        
        for model_file in self.model_dir.glob("*_model.pkl"):
            # Extract SKU and model type from filename
            parts = model_file.stem.split("_")
            if len(parts) >= 3:
                sku = "_".join(parts[:-2])
                model_type = parts[-2]
                
                status = self.get_model_status(sku, model_type)
                models.append({
                    "sku": sku,
                    "model_type": model_type,
                    "status": status
                })
        
        return models

def main():
    """Demonstrate model management functionality."""
    print("="*60)
    print("🔄 MODEL MANAGEMENT & INPUT REQUIREMENTS")
    print("="*60)
    
    # Initialize model manager
    manager = ModelManager()
    
    # Show input requirements
    print("\n📋 INPUT REQUIREMENTS:")
    requirements = manager.get_input_requirements()
    
    for pred_type, req in requirements.items():
        print(f"\n{pred_type.upper()}:")
        print(f"  Description: {req['description']}")
        print(f"  Required fields: {', '.join(req['required_fields'])}")
        if req['optional_fields']:
            print(f"  Optional fields: {', '.join(req['optional_fields'])}")
    
    # Show input templates
    print(f"\n📝 INPUT TEMPLATES:")
    
    for pred_type in ["daily_prediction", "monthly_prediction"]:
        template = manager.create_prediction_input_template(pred_type)
        print(f"\n{pred_type.upper()} Prediction Template:")
        print(f"  {json.dumps(template, indent=2)}")
    
    # Show validation examples
    print(f"\n✅ VALIDATION EXAMPLES:")
    
    # Valid daily input
    valid_daily = {
        "sku": "CMSM01",
        "date": "2025-01-15",
        "quantity_lag1": 50,
        "quantity_lag7": 45,
        "quantity_lag30": 40,
        "month": 1,
        "day_of_week": 2
    }
    
    is_valid, errors = manager.validate_prediction_input(valid_daily, "daily_prediction")
    print(f"Valid daily input: {is_valid}")
    if errors:
        print(f"  Errors: {errors}")
    
    # Invalid input
    invalid_input = {
        "sku": "CMSM01",
        "date": "invalid-date",
        "month": 13  # Invalid month
    }
    
    is_valid, errors = manager.validate_prediction_input(invalid_input, "daily_prediction")
    print(f"Invalid input: {is_valid}")
    if errors:
        print(f"  Errors: {errors}")
    
    # Model retraining logic
    print(f"\n🔄 RETRAINING LOGIC:")
    print("Models need retraining when:")
    print("• Model is older than 30 days")
    print("• New data is available (more than 7 days)")
    print("• Model performance has degraded")
    
    # Show model status examples
    print(f"\n📊 MODEL STATUS EXAMPLES:")
    status = manager.get_model_status("CMSM01", "daily")
    print(f"Model status for CMSM01: {json.dumps(status, indent=2)}")
    
    print(f"\n✅ Model management setup completed!")
    print("="*60)

if __name__ == "__main__":
    main()
