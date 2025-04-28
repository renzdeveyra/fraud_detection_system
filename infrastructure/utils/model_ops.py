import os
import pickle
import json
import yaml
from typing import Any, Dict, Optional
import datetime
from infrastructure.config import load_paths, get_project_root
from infrastructure.utils.logger import logger, log_execution_time

@log_execution_time
def save_model(model: Any, model_name: str, version: Optional[str] = None) -> str:
    """Save a trained model to disk"""
    paths = load_paths()
    
    if model_name == 'classifier':
        model_path = paths['models']['classifier']['path']
    elif model_name == 'anomaly':
        model_path = paths['models']['anomaly']['path']
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Add version suffix if provided
    if version:
        base, ext = os.path.splitext(model_path)
        model_path = f"{base}_{version}{ext}"
    
    # Ensure directory exists
    full_path = os.path.join(get_project_root(), model_path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    
    # Save the model
    with open(full_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Saved {model_name} model to {model_path}")
    return model_path

@log_execution_time
def load_model(model_name: str, version: Optional[str] = None) -> Any:
    """Load a trained model from disk"""
    paths = load_paths()
    
    if model_name == 'classifier':
        model_path = paths['models']['classifier']['path']
    elif model_name == 'anomaly':
        model_path = paths['models']['anomaly']['path']
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Add version suffix if provided
    if version:
        base, ext = os.path.splitext(model_path)
        model_path = f"{base}_{version}{ext}"
    
    full_path = os.path.join(get_project_root(), model_path)
    
    # Check if model exists
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Model file not found: {full_path}")
    
    # Load the model
    with open(full_path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Loaded {model_name} model from {model_path}")
    return model

def load_rules(rule_type: str) -> Dict[str, Any]:
    """Load rules from JSON or YAML file"""
    paths = load_paths()
    
    if rule_type == 'static':
        rule_path = paths['models']['classifier']['rules']
        full_path = os.path.join(get_project_root(), rule_path)
        
        with open(full_path, 'r') as f:
            rules = json.load(f)
            
    elif rule_type == 'thresholds':
        threshold_path = paths['models']['anomaly']['thresholds']
        full_path = os.path.join(get_project_root(), threshold_path)
        
        with open(full_path, 'r') as f:
            rules = yaml.safe_load(f)
            
    else:
        raise ValueError(f"Unknown rule type: {rule_type}")
    
    logger.debug(f"Loaded {rule_type} rules from {full_path}")
    return rules

def save_rules(rules: Dict[str, Any], rule_type: str) -> None:
    """Save rules to JSON or YAML file"""
    paths = load_paths()
    
    if rule_type == 'static':
        rule_path = paths['models']['classifier']['rules']
        full_path = os.path.join(get_project_root(), rule_path)
        
        with open(full_path, 'w') as f:
            json.dump(rules, f, indent=2)
            
    elif rule_type == 'thresholds':
        threshold_path = paths['models']['anomaly']['thresholds']
        full_path = os.path.join(get_project_root(), threshold_path)
        
        with open(full_path, 'w') as f:
            yaml.dump(rules, f)
            
    else:
        raise ValueError(f"Unknown rule type: {rule_type}")
    
    logger.info(f"Saved {rule_type} rules to {full_path}")

def save_model_metrics(metrics: Dict[str, Any], model_name: str) -> None:
    """Save model evaluation metrics"""
    metrics_dir = os.path.join(get_project_root(), f"experts/{model_name}/evaluation")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Add timestamp
    metrics['timestamp'] = datetime.datetime.now().isoformat()
    
    metrics_file = os.path.join(metrics_dir, f"{model_name}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved {model_name} metrics to {metrics_file}")

def get_model_info(model_name: str, version: Optional[str] = None) -> Dict[str, Any]:
    """Get information about a model file"""
    paths = load_paths()
    
    if model_name == 'classifier':
        model_path = paths['models']['classifier']['path']
    elif model_name == 'anomaly':
        model_path = paths['models']['anomaly']['path']
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Add version suffix if provided
    if version:
        base, ext = os.path.splitext(model_path)
        model_path = f"{base}_{version}{ext}"
    
    full_path = os.path.join(get_project_root(), model_path)
    
    # Check if model exists
    if not os.path.exists(full_path):
        return {
            "exists": False,
            "path": model_path,
            "size": 0,
            "modified": None
        }
    
    # Get file stats
    stats = os.stat(full_path)
    
    return {
        "exists": True,
        "path": model_path,
        "size": stats.st_size,
        "modified": datetime.datetime.fromtimestamp(stats.st_mtime).isoformat()
    }