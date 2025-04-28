import os
import yaml
import logging.config
from typing import Dict, Any

# Base directory
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_project_root() -> str:
    """Return the project root directory path"""
    return _BASE_DIR

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    full_path = os.path.join(_BASE_DIR, config_path)
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)

def load_paths() -> Dict[str, Any]:
    """Load file paths configuration"""
    return load_config('infrastructure/config/paths.yaml')

def load_params() -> Dict[str, Any]:
    """Load model parameters configuration"""
    return load_config('infrastructure/config/model_params.yaml')

def setup_logging() -> None:
    """Configure logging based on config file"""
    log_config_path = os.path.join(_BASE_DIR, 'infrastructure/config/logging.conf')
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(_BASE_DIR, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    logging.config.fileConfig(log_config_path, disable_existing_loggers=False)

# Setup logging on import
setup_logging()