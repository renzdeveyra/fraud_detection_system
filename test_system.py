#!/usr/bin/env python
"""
Comprehensive test script for the fraud detection system with the new directory structure.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from infrastructure.config import load_paths, get_project_root, load_params
from infrastructure.utils.data_loader import (
    load_raw_data, 
    load_processed_data, 
    process_raw_data,
    split_data,
    load_transaction_batch,
    save_results
)
from infrastructure.utils.model_ops import (
    load_model,
    save_model,
    load_rules
)
from pipelines.training_pipeline import (
    train_classifier,
    train_anomaly_detector
)
from pipelines.inference_pipeline import (
    load_experts,
    process_single_transaction
)

def test_training_pipeline():
    """Test the training pipeline with the new directory structure"""
    print("\n=== Testing Training Pipeline ===")
    
    # Create a small sample dataset for training
    try:
        # Load processed data
        df = load_processed_data()
        
        # Take a small sample for quick testing
        sample_df = df.sample(n=min(1000, len(df)), random_state=42)
        
        # Ensure we have a target column (fraud)
        if 'fraud' not in sample_df.columns and 'Class' in sample_df.columns:
            sample_df = sample_df.rename(columns={'Class': 'fraud'})
        
        # Save to examples directory
        paths = load_paths()
        examples_dir = os.path.join(get_project_root(), paths['data']['examples'])
        os.makedirs(examples_dir, exist_ok=True)
        
        sample_path = os.path.join(examples_dir, 'sample_data.parquet')
        sample_df.to_parquet(sample_path, index=False)
        print(f"✅ Created sample dataset with {len(sample_df)} records at {sample_path}")
        
        # Train classifier
        print("\nTraining classifier...")
        classifier, metrics = train_classifier(sample_path, test_size=0.3, save=True)
        print(f"✅ Successfully trained classifier with metrics: {metrics}")
        
        # Train anomaly detector
        print("\nTraining anomaly detector...")
        detector, metrics = train_anomaly_detector(sample_path, contamination=0.01, save=True)
        print(f"✅ Successfully trained anomaly detector with metrics: {metrics}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to test training pipeline: {str(e)}")
        return False

def test_inference_pipeline():
    """Test the inference pipeline with the new directory structure"""
    print("\n=== Testing Inference Pipeline ===")
    
    try:
        # Load experts
        print("Loading expert systems...")
        mediator = load_experts()
        print("✅ Successfully loaded expert systems")
        
        # Create a test transaction
        test_transaction = {
            "distance_from_home": 57.87,
            "distance_from_last_transaction": 0.31,
            "ratio_to_median_purchase_price": 1.95,
            "repeat_retailer": 1,
            "used_chip": 1,
            "used_pin_number": 1,
            "online_order": 0,
            "transaction_id": "test-123"
        }
        
        # Process the transaction
        print("\nProcessing test transaction...")
        result = process_single_transaction(mediator, test_transaction)
        print(f"✅ Successfully processed test transaction")
        print(f"Decision: {result['decision']}")
        print(f"Explanation: {result['explanation']}")
        
        # Save the result to the inference directory
        paths = load_paths()
        inference_dir = os.path.join(get_project_root(), paths['data']['inference'])
        os.makedirs(inference_dir, exist_ok=True)
        
        result_path = os.path.join(inference_dir, 'test_result.json')
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"✅ Saved inference result to {result_path}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to test inference pipeline: {str(e)}")
        return False

def test_rules_loading():
    """Test loading rules with the new directory structure"""
    print("\n=== Testing Rules Loading ===")
    
    try:
        # Load static rules
        static_rules = load_rules('static')
        print(f"✅ Successfully loaded static rules: {list(static_rules.keys())}")
        
        # Load anomaly thresholds
        thresholds = load_rules('thresholds')
        print(f"✅ Successfully loaded anomaly thresholds: {thresholds}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load rules: {str(e)}")
        return False

def main():
    """Run all system tests"""
    print("Testing fraud detection system with new directory structure...")
    
    # Test rules loading
    test_rules_loading()
    
    # Test training pipeline
    test_training_pipeline()
    
    # Test inference pipeline
    test_inference_pipeline()
    
    print("\nAll system tests completed!")

if __name__ == "__main__":
    main()
