import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from infrastructure.config import load_paths, get_project_root

def load_raw_data() -> pd.DataFrame:
    """Load the original credit card fraud dataset"""
    paths = load_paths()
    data_path = os.path.join(get_project_root(), paths['data']['raw'])

    return pd.read_csv(data_path)

def load_processed_data() -> pd.DataFrame:
    """Load the processed transactions data"""
    paths = load_paths()
    data_path = os.path.join(get_project_root(), paths['data']['processed'])

    return pd.read_parquet(data_path)

def process_raw_data(save: bool = True) -> pd.DataFrame:
    """Process raw data and optionally save to processed directory"""
    df = load_raw_data()

    # Basic processing
    processed_df = df.copy()

    # Handle missing values
    processed_df = processed_df.fillna(0)

    # Add timestamp features if available
    if 'Time' in processed_df.columns:
        processed_df['hour_of_day'] = processed_df['Time'] % 24
        processed_df['day_of_week'] = (processed_df['Time'] // 24) % 7

    # Normalize amounts
    if 'Amount' in processed_df.columns:
        processed_df['norm_amount'] = processed_df['Amount'] / processed_df['Amount'].max()

    # Process new features
    # Convert boolean features to integers if they're not already
    bool_features = ['repeat_retailer', 'used_chip', 'used_pin_number', 'online_order']
    for feature in bool_features:
        if feature in processed_df.columns:
            processed_df[feature] = processed_df[feature].astype(int)

    # Normalize distance features if they exist
    distance_features = ['distance_from_home', 'distance_from_last_transaction']
    for feature in distance_features:
        if feature in processed_df.columns:
            # Apply log transformation to handle skewed distributions
            processed_df[f'log_{feature}'] = np.log1p(processed_df[feature])

    # Process ratio_to_median_purchase_price if it exists
    if 'ratio_to_median_purchase_price' in processed_df.columns:
        # Cap extreme values
        processed_df['ratio_to_median_purchase_price'] = processed_df['ratio_to_median_purchase_price'].clip(0, 10)

    # Ensure fraud label is properly formatted
    if 'fraud' in processed_df.columns:
        processed_df['fraud'] = processed_df['fraud'].astype(int)

    if save:
        paths = load_paths()
        output_path = os.path.join(get_project_root(), paths['data']['processed'])

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        processed_df.to_parquet(output_path, index=False)

    return processed_df

def split_data(df: pd.DataFrame,
               target_col: str = 'fraud',
               test_size: float = 0.2,
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and testing sets"""
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test

def load_transaction_batch(batch_file: str) -> pd.DataFrame:
    """Load a batch of transactions for processing"""
    batch_path = os.path.join(get_project_root(), batch_file)

    if batch_path.endswith('.csv'):
        return pd.read_csv(batch_path)
    elif batch_path.endswith('.parquet'):
        return pd.read_parquet(batch_path)
    elif batch_path.endswith('.json'):
        return pd.read_json(batch_path)
    else:
        raise ValueError(f"Unsupported file format for {batch_file}")

def save_results(results: Dict[str, Any], output_file: str) -> None:
    """Save detection results to file"""
    output_path = os.path.join(get_project_root(), output_file)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert to DataFrame if it's a dictionary
    if isinstance(results, dict):
        results_df = pd.DataFrame([results])
    else:
        results_df = pd.DataFrame(results)

    # Save based on file extension
    if output_file.endswith('.csv'):
        results_df.to_csv(output_path, index=False)
    elif output_file.endswith('.parquet'):
        results_df.to_parquet(output_path, index=False)
    elif output_file.endswith('.json'):
        results_df.to_json(output_path, orient='records')
    else:
        raise ValueError(f"Unsupported file format for {output_file}")