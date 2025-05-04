#!/usr/bin/env python
"""
Test script to verify data loading with the new directory structure.
"""

import os
import sys
import pandas as pd
from infrastructure.config import load_paths, get_project_root
from infrastructure.utils.data_loader import (
    load_raw_data,
    load_processed_data,
    process_raw_data,
    split_data,
    load_transaction_batch
)

def test_load_raw_data():
    """Test loading raw data from the source directory"""
    print("\n=== Testing Raw Data Loading ===")
    try:
        df = load_raw_data()
        print(f"✅ Successfully loaded raw data with shape: {df.shape}")
        print(f"Sample data:\n{df.head(3)}")
        return df
    except Exception as e:
        print(f"❌ Failed to load raw data: {str(e)}")
        return None

def test_load_processed_data():
    """Test loading processed data from the prepared directory"""
    print("\n=== Testing Processed Data Loading ===")
    try:
        df = load_processed_data()
        print(f"✅ Successfully loaded processed data with shape: {df.shape}")
        print(f"Sample data:\n{df.head(3)}")
        return df
    except Exception as e:
        print(f"❌ Failed to load processed data: {str(e)}")
        return None

def test_process_raw_data():
    """Test processing raw data and saving to processed directory"""
    print("\n=== Testing Data Processing ===")
    try:
        df = process_raw_data(save=True)
        print(f"✅ Successfully processed raw data with shape: {df.shape}")

        # Verify the file was saved
        paths = load_paths()
        processed_path = os.path.join(get_project_root(), paths['data']['processed'])
        if os.path.exists(processed_path):
            print(f"✅ Processed data saved to: {processed_path}")
        else:
            print(f"❌ Failed to save processed data to: {processed_path}")

        return df
    except Exception as e:
        print(f"❌ Failed to process raw data: {str(e)}")
        return None

def test_data_splitting():
    """Test splitting data into training and testing sets"""
    print("\n=== Testing Data Splitting ===")
    try:
        # Load processed data
        df = load_processed_data()

        # Ensure we have a target column (fraud)
        if 'fraud' not in df.columns and 'Class' in df.columns:
            df = df.rename(columns={'Class': 'fraud'})

        # Split the data
        X_train, X_test, y_train, y_test = split_data(df, target_col='fraud', test_size=0.2)

        print(f"✅ Successfully split data:")
        print(f"  - Training features: {X_train.shape}")
        print(f"  - Testing features: {X_test.shape}")
        print(f"  - Training labels: {y_train.shape}")
        print(f"  - Testing labels: {y_test.shape}")

        # Save splits to their respective directories
        paths = load_paths()
        train_dir = os.path.join(get_project_root(), paths['data']['training'])
        test_dir = os.path.join(get_project_root(), paths['data']['testing'])

        # Ensure directories exist
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Save training data
        train_df = pd.concat([X_train, y_train], axis=1)
        train_path = os.path.join(train_dir, 'train_data.parquet')
        train_df.to_parquet(train_path, index=False)
        print(f"✅ Saved training data to: {train_path}")

        # Save testing data
        test_df = pd.concat([X_test, y_test], axis=1)
        test_path = os.path.join(test_dir, 'test_data.parquet')
        test_df.to_parquet(test_path, index=False)
        print(f"✅ Saved testing data to: {test_path}")

        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"❌ Failed to split data: {str(e)}")
        return None, None, None, None

def test_transaction_batch_loading():
    """Test loading a transaction batch from different directories"""
    print("\n=== Testing Transaction Batch Loading ===")

    # Get paths
    paths = load_paths()

    # Test loading from source
    try:
        source_path = paths['data']['source']
        df = load_transaction_batch(source_path)
        print(f"✅ Successfully loaded transaction batch from source: {df.shape}")
    except Exception as e:
        print(f"❌ Failed to load transaction batch from source: {str(e)}")

    # Test loading from processed
    try:
        processed_path = paths['data']['processed']
        df = load_transaction_batch(processed_path)
        print(f"✅ Successfully loaded transaction batch from processed: {df.shape}")
    except Exception as e:
        print(f"❌ Failed to load transaction batch from processed: {str(e)}")

    # Test loading from training (if available)
    train_path = os.path.join(paths['data']['training'], 'train_data.parquet')
    if os.path.exists(os.path.join(get_project_root(), train_path)):
        try:
            df = load_transaction_batch(train_path)
            print(f"✅ Successfully loaded transaction batch from training: {df.shape}")
        except Exception as e:
            print(f"❌ Failed to load transaction batch from training: {str(e)}")

def main():
    """Run all tests"""
    print("Testing data loading with new directory structure...")

    # Test loading raw data
    raw_df = test_load_raw_data()

    # Test loading processed data
    processed_df = test_load_processed_data()

    # Test processing raw data
    processed_df = test_process_raw_data()

    # Test data splitting
    test_data_splitting()

    # Test transaction batch loading
    test_transaction_batch_loading()

    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
