"""
Utility functions for the fraud detection system.
Provides core functionality used across the system.
"""

from infrastructure.utils.logger import logger, log_execution_time, log_transaction_processing
from infrastructure.utils.data_loader import (
    load_raw_data, load_processed_data, process_raw_data,
    split_data, load_transaction_batch, save_results
)
from infrastructure.utils.model_ops import (
    save_model, load_model, load_rules, save_rules,
    save_model_metrics, get_model_info
)
from infrastructure.utils.caching import Cache, ModelCache, TransactionCache
from infrastructure.utils.error_handling import (
    handle_errors, safe_execute, FraudDetectionError,
    ConfigurationError, ModelError, DataError, ProcessingError
)

__all__ = [
    # Logging utilities
    'logger',
    'log_execution_time',
    'log_transaction_processing',

    # Data loading utilities
    'load_raw_data',
    'load_processed_data',
    'process_raw_data',
    'split_data',
    'load_transaction_batch',
    'save_results',

    # Model operations
    'save_model',
    'load_model',
    'load_rules',
    'save_rules',
    'save_model_metrics',
    'get_model_info',

    # Caching utilities
    'Cache',
    'ModelCache',
    'TransactionCache',

    # Error handling
    'handle_errors',
    'safe_execute',
    'FraudDetectionError',
    'ConfigurationError',
    'ModelError',
    'DataError',
    'ProcessingError'
]