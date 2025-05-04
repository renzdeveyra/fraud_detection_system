"""
Compatibility module for legacy code using src/utils.py.
Redirects to standardized infrastructure modules.
"""

import logging
import joblib
import yaml
from pathlib import Path

from infrastructure.config import load_params, get_project_root
from infrastructure.utils import logger as infra_logger
from infrastructure.utils.model_ops import save_model as infra_save_model, load_model as infra_load_model


def load_config():
    """
    Load configuration from params.yaml.
    Compatibility function that redirects to infrastructure.config.
    
    Returns:
        Dictionary containing configuration parameters
    """
    # First try to use infrastructure module
    try:
        return load_params()
    except Exception:
        # Fall back to original implementation
        with open(Path(__file__).parent.parent/"config"/"params.yaml") as f:
            return yaml.safe_load(f)


def setup_logger(name):
    """
    Set up a logger with the given name.
    Compatibility function that uses infrastructure logger if possible.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger
    """
    # Try to use infrastructure logger first
    try:
        # Return a child logger of the infrastructure logger
        child_logger = logging.getLogger(f"fraudDetection.{name}")
        # If no handlers, add the infrastructure logger's handlers
        if not child_logger.handlers:
            for handler in infra_logger.handlers:
                child_logger.addHandler(handler)
        return child_logger
    except Exception:
        # Fall back to original implementation
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        return logger


def save_model(model, path):
    """
    Save a model to the given path.
    Compatibility function that redirects to infrastructure.utils.model_ops.
    
    Args:
        model: Model to save
        path: Path to save the model to
    """
    try:
        # Try to determine model type for infrastructure save_model
        if "classifier" in str(path).lower():
            model_name = "classifier"
        elif "anomaly" in str(path).lower() or "isolation" in str(path).lower():
            model_name = "anomaly"
        else:
            # Fall back to original implementation
            joblib.dump(model, path)
            print(f"Model saved to {path}")
            return
            
        # Use infrastructure save_model
        infra_save_model(model, model_name)
        print(f"Model saved using infrastructure utilities")
    except Exception:
        # Fall back to original implementation
        joblib.dump(model, path)
        print(f"Model saved to {path}")


def load_model(path):
    """
    Load a model from the given path.
    Compatibility function that redirects to infrastructure.utils.model_ops.
    
    Args:
        path: Path to load the model from
        
    Returns:
        Loaded model
    """
    try:
        # Try to determine model type for infrastructure load_model
        if "classifier" in str(path).lower():
            return infra_load_model("classifier")
        elif "anomaly" in str(path).lower() or "isolation" in str(path).lower():
            return infra_load_model("anomaly")
    except Exception:
        pass
        
    # Fall back to original implementation
    return joblib.load(path)
