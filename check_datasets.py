#!/usr/bin/env python
"""
Script to check the differences between datasets.
"""

import pandas as pd
import os

def check_dataset(file_path, name):
    """Check a dataset and print its properties"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            print(f"Unsupported file format for {file_path}")
            return

        print(f"\n=== {name} Dataset ===")
        print(f"Path: {file_path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Sample data:\n{df.head(3)}")

        # Check for missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"Missing values:\n{missing[missing > 0]}")
        else:
            print("No missing values")

        # Check data types
        print(f"Data types:\n{df.dtypes}")

    except Exception as e:
        print(f"Error checking {name} dataset: {str(e)}")

def main():
    """Check all datasets"""
    # Check source dataset
    check_dataset('data/source/creditcard.csv', 'Source')

    # Check processed dataset
    check_dataset('data/processed/transactions_processed.parquet', 'Processed')

    # Check old prepared dataset if it exists
    if os.path.exists('data/prepared/transactions_processed.parquet'):
        check_dataset('data/prepared/transactions_processed.parquet', 'Old Prepared')

if __name__ == "__main__":
    main()
