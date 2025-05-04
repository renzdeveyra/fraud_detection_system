"""
Unified caching utilities for the fraud detection system.
Provides standardized caching mechanisms for model predictions and other data.
"""

import hashlib
from typing import Dict, Any, Optional, Union, Callable
import numpy as np


class Cache:
    """Generic cache implementation for the fraud detection system."""

    def __init__(self, max_size: int = 1000, name: str = "generic"):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of items to store in cache
            name: Name of the cache for logging purposes
        """
        self.cache = {}
        self.max_size = max_size
        self.name = name
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        value = self.cache.get(key)
        if value is not None:
            self.hits += 1
        else:
            self.misses += 1
        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        # If cache is full, remove oldest items
        if len(self.cache) >= self.max_size:
            # Simple approach: clear half the cache
            keys = list(self.cache.keys())
            for old_key in keys[:len(keys)//2]:
                del self.cache[old_key]

        self.cache[key] = value

    def clear(self) -> None:
        """Clear the cache."""
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "name": self.name,
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }


class ModelCache(Cache):
    """Specialized cache for model predictions."""

    def __init__(self, max_size: int = 1000, model_name: str = "model"):
        """
        Initialize the model cache.

        Args:
            max_size: Maximum number of items to store in cache
            model_name: Name of the model for cache key generation
        """
        super().__init__(max_size, name=f"{model_name}_cache")
        self.model_name = model_name

    def generate_key(self, model_version: str, features: np.ndarray) -> str:
        """
        Generate a cache key for model predictions.

        Args:
            model_version: Model version string
            features: Feature array

        Returns:
            Cache key string
        """
        # Use hash of features as part of the key
        feature_bytes = features.tobytes()
        feature_hash = hashlib.md5(feature_bytes).hexdigest()
        return f"{self.model_name}_{model_version}_{feature_hash}"

    def cached_predict(self, predict_func: Callable, 
                      model_version: str, 
                      features: np.ndarray, 
                      fallback_value: float = 0.5) -> float:
        """
        Make a cached prediction, using the cache if available.

        Args:
            predict_func: Function to call for prediction if cache miss
            model_version: Model version string
            features: Feature array
            fallback_value: Default value to return if prediction fails

        Returns:
            Prediction result
        """
        cache_key = self.generate_key(model_version, features)
        
        # Check cache first
        cached_result = self.get(cache_key)
        if cached_result is not None:
            return cached_result
            
        # Cache miss, make prediction
        try:
            result = predict_func(features)
            self.set(cache_key, result)
            return result
        except Exception as e:
            # Return fallback value on error
            return fallback_value


class TransactionCache(Cache):
    """Specialized cache for transaction processing results."""

    def __init__(self, max_size: int = 1000):
        """
        Initialize the transaction cache.

        Args:
            max_size: Maximum number of items to store in cache
        """
        super().__init__(max_size, name="transaction_cache")

    def generate_key(self, transaction: Dict[str, Any]) -> str:
        """
        Generate a cache key for a transaction.

        Args:
            transaction: Transaction dictionary

        Returns:
            Cache key string
        """
        # Use transaction ID if available
        transaction_id = transaction.get('id', '') or transaction.get('transaction_id', '')
        
        if transaction_id:
            return f"tx_{transaction_id}"
            
        # Otherwise, create a hash of key transaction properties
        key_props = []
        for field in ['amount', 'merchant', 'timestamp', 'user_id']:
            if field in transaction:
                key_props.append(f"{field}:{transaction[field]}")
                
        key_str = "_".join(key_props)
        return f"tx_{hashlib.md5(key_str.encode()).hexdigest()}"
