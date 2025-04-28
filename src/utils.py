import yaml
import joblib
import logging
import numpy as np
from pathlib import Path

def load_config():
    with open(Path(__file__).parent.parent/"config"/"params.yaml") as f:
        return yaml.safe_load(f)

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger

def save_model(model, path):
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path):
    return joblib.load(path)
