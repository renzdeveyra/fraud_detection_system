"""
Error handling utilities for the fraud detection system.
Provides standardized error handling mechanisms.
"""

import functools
import traceback
from typing import Callable, Any, Dict, Optional, Type, Union, List

from infrastructure.utils.logger import logger


class FraudDetectionError(Exception):
    """Base exception class for fraud detection system errors."""
    
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            original_error: Original exception that caused this error
        """
        self.original_error = original_error
        self.traceback = traceback.format_exc() if original_error else None
        super().__init__(message)


class ConfigurationError(FraudDetectionError):
    """Exception raised for configuration errors."""
    pass


class ModelError(FraudDetectionError):
    """Exception raised for model-related errors."""
    pass


class DataError(FraudDetectionError):
    """Exception raised for data-related errors."""
    pass


class ProcessingError(FraudDetectionError):
    """Exception raised for transaction processing errors."""
    pass


def handle_errors(error_type: Type[FraudDetectionError] = FraudDetectionError,
                 fallback_value: Optional[Any] = None,
                 log_level: str = "error") -> Callable:
    """
    Decorator to standardize error handling.
    
    Args:
        error_type: Type of error to raise if an exception occurs
        fallback_value: Value to return if an exception occurs
        log_level: Logging level to use for the error
    
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get function name for error message
                func_name = getattr(func, "__name__", "unknown")
                
                # Create error message
                error_message = f"Error in {func_name}: {str(e)}"
                
                # Log the error
                if log_level == "debug":
                    logger.debug(error_message)
                elif log_level == "info":
                    logger.info(error_message)
                elif log_level == "warning":
                    logger.warning(error_message)
                else:
                    logger.error(error_message)
                    logger.debug(traceback.format_exc())
                
                # Raise or return fallback
                if fallback_value is None:
                    raise error_type(error_message, original_error=e)
                return fallback_value
                
        return wrapper
    return decorator


def safe_execute(func: Callable, 
                *args, 
                fallback_value: Any = None, 
                error_message: str = "Error executing function", 
                **kwargs) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Arguments to pass to the function
        fallback_value: Value to return if an exception occurs
        error_message: Message to log if an exception occurs
        **kwargs: Keyword arguments to pass to the function
    
    Returns:
        Function result or fallback value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"{error_message}: {str(e)}")
        logger.debug(traceback.format_exc())
        return fallback_value
