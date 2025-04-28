import logging
import functools
import time
import traceback
from typing import Callable, Any

# Get logger
logger = logging.getLogger('fraudDetection')

def log_execution_time(func: Callable) -> Callable:
    """Decorator to log function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        logger.debug(f"Starting execution of {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"Executed {func.__name__} in {execution_time:.4f} seconds")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Failed executing {func.__name__} after {execution_time:.4f} seconds: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
            
    return wrapper

def log_transaction_processing(transaction_id: str) -> Callable:
    """Decorator to log transaction processing details"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger.info(f"Processing transaction {transaction_id}")
            
            try:
                result = func(*args, **kwargs)
                decision = result.get('decision', 'UNKNOWN')
                logger.info(f"Transaction {transaction_id} decision: {decision}")
                return result
                
            except Exception as e:
                logger.error(f"Error processing transaction {transaction_id}: {str(e)}")
                logger.debug(traceback.format_exc())
                raise
                
        return wrapper
    return decorator

def setup_console_logger(level: str = 'INFO') -> None:
    """Set up console logger for interactive sessions"""
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level))
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.setLevel(getattr(logging, level))
    
    logger.info("Console logger configured")

# Export logger for use in other modules
__all__ = ['logger', 'log_execution_time', 'log_transaction_processing', 'setup_console_logger']