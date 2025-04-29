"""
Common utilities for fraud detection experts.
Contains shared functions used across different expert systems.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging

from infrastructure.utils import logger


def extract_model_features(model, transaction: Dict[str, Any], default_value=0) -> np.ndarray:
    """
    Extract features from transaction based on model's expected features.

    Args:
        model: Trained ML model with feature_names_in_ attribute
        transaction: Transaction dictionary
        default_value: Default value for missing features

    Returns:
        numpy array of features in the correct order for the model
    """
    features = []

    if hasattr(model, 'feature_names_in_'):
        for feature in model.feature_names_in_:
            features.append(transaction.get(feature, default_value))
    else:
        # If model doesn't have feature names, extract all numeric values
        features = [v for k, v in transaction.items()
                   if isinstance(v, (int, float)) and k not in ['id', 'user_id', 'transaction_id']]

    return np.array(features).reshape(1, -1)


def check_model_compatibility(model, available_features: List[str]) -> Tuple[bool, List[str]]:
    """
    Check if model is compatible with available features.

    Args:
        model: Trained ML model with feature_names_in_ attribute
        available_features: List of available feature names

    Returns:
        Tuple of (is_compatible, missing_features)
    """
    if not hasattr(model, 'feature_names_in_'):
        return True, []  # Can't check compatibility

    required_features = set(model.feature_names_in_)
    available_features = set(available_features)
    missing_features = list(required_features - available_features)

    is_compatible = len(missing_features) == 0

    if not is_compatible:
        logger.warning(f"Model missing features: {missing_features}")

    return is_compatible, missing_features


def calculate_heuristic_fraud_score(transaction: Dict[str, Any]) -> float:
    """
    Calculate a heuristic fraud score based on transaction features.
    Used as a fallback when ML models can't be applied.

    Focuses only on the core dataset features:
    - distance_from_home
    - distance_from_last_transaction
    - ratio_to_median_purchase_price
    - repeat_retailer
    - used_chip
    - used_pin_number
    - online_order

    Args:
        transaction: Transaction dictionary

    Returns:
        Fraud score between 0 and 1
    """
    fraud_factors = []

    # Check for high distance from home
    if transaction.get('distance_from_home', 0) > 100:
        fraud_factors.append(0.25)

    # Check for high distance from last transaction
    if transaction.get('distance_from_last_transaction', 0) > 50:
        fraud_factors.append(0.2)

    # Check for unusual ratio to median purchase price
    if transaction.get('ratio_to_median_purchase_price', 1.0) > 3.0:
        fraud_factors.append(0.25)

    # Check for non-repeat retailer (new merchant)
    if transaction.get('repeat_retailer', 1) == 0:
        fraud_factors.append(0.1)

    # Check for unusual payment methods
    if transaction.get('used_chip', 1) == 0:
        fraud_factors.append(0.2)

    if transaction.get('used_pin_number', 1) == 0:
        fraud_factors.append(0.2)

    # Check for online order
    if transaction.get('online_order', 0) == 1:
        fraud_factors.append(0.2)

    # Calculate combined score
    if not fraud_factors:
        return 0.0  # No fraud indicators detected

    # Combine factors (max possible is around 1.4, so we scale it)
    combined_score = min(1.0, sum(fraud_factors) / 1.4)

    return combined_score


def calculate_heuristic_anomaly_score(transaction: Dict[str, Any]) -> float:
    """
    Calculate a heuristic anomaly score based on transaction features.
    Used as a fallback when ML models can't be applied.

    Focuses only on the core dataset features:
    - distance_from_home
    - distance_from_last_transaction
    - ratio_to_median_purchase_price
    - repeat_retailer
    - used_chip
    - used_pin_number
    - online_order

    Args:
        transaction: Transaction dictionary

    Returns:
        Anomaly score between -1 and 1 (negative is more anomalous)
    """
    anomaly_factors = []

    # Check for high distance from home
    if transaction.get('distance_from_home', 0) > 100:
        anomaly_factors.append(0.3)

    # Check for high distance from last transaction
    if transaction.get('distance_from_last_transaction', 0) > 50:
        anomaly_factors.append(0.25)

    # Check for unusual ratio to median purchase price
    if transaction.get('ratio_to_median_purchase_price', 1.0) > 3.0:
        anomaly_factors.append(0.25)

    # Check for non-repeat retailer (new merchant)
    if transaction.get('repeat_retailer', 1) == 0:
        anomaly_factors.append(0.1)

    # Check for unusual payment methods
    if transaction.get('used_chip', 1) == 0:
        anomaly_factors.append(0.15)

    if transaction.get('used_pin_number', 1) == 0:
        anomaly_factors.append(0.15)

    # Check for online order
    if transaction.get('online_order', 0) == 1:
        anomaly_factors.append(0.15)

    # Calculate combined score
    if not anomaly_factors:
        return 0.0  # No anomalies detected

    # Combine factors (max possible is around 1.35, so we scale it)
    combined_score = sum(anomaly_factors) / 1.5

    # Convert to anomaly score range (-1 to 1, where negative is more anomalous)
    return -min(1.0, combined_score)


def safe_predict(model, features: np.ndarray, fallback_value=0.5) -> float:
    """
    Safely make predictions with a model, handling exceptions.

    Args:
        model: Trained ML model
        features: Feature array
        fallback_value: Default value to return if prediction fails

    Returns:
        Prediction score
    """
    try:
        if hasattr(model, 'predict_proba'):
            # For classifiers that output probabilities
            return model.predict_proba(features)[0][1]
        elif hasattr(model, 'decision_function'):
            # For anomaly detectors and some classifiers
            return model.decision_function(features)[0]
        else:
            # For other models, use predict and hope for the best
            return float(model.predict(features)[0])
    except Exception as e:
        logger.warning(f"Error in model prediction: {str(e)}")
        return fallback_value


def get_model_version(model) -> str:
    """
    Get the version of a model.

    Args:
        model: Trained ML model

    Returns:
        Model version string
    """
    if hasattr(model, 'version'):
        return model.version

    # Try to infer version from model attributes
    model_type = type(model).__name__

    if hasattr(model, '_sklearn_version'):
        return f"{model_type}-{model._sklearn_version}"

    # Default version
    return f"{model_type}-unknown"


def add_model_version(model, version: str) -> None:
    """
    Add version information to a model.

    Args:
        model: Trained ML model
        version: Version string
    """
    model.version = version


class ModelCache:
    """Simple cache for model predictions to improve performance."""

    def __init__(self, max_size=1000):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of items to store in cache
        """
        self.cache = {}
        self.max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        return self.cache.get(key)

    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        # If cache is full, remove oldest item
        if len(self.cache) >= self.max_size:
            # Simple approach: clear half the cache
            keys = list(self.cache.keys())
            for old_key in keys[:len(keys)//2]:
                del self.cache[old_key]

        self.cache[key] = value

    def clear(self) -> None:
        """Clear the cache."""
        self.cache = {}


def generate_cache_key(model_version: str, features: np.ndarray) -> str:
    """
    Generate a cache key for model predictions.

    Args:
        model_version: Model version string
        features: Feature array

    Returns:
        Cache key string
    """
    # Use hash of features as part of the key
    feature_hash = hash(features.tobytes())
    return f"{model_version}_{feature_hash}"
