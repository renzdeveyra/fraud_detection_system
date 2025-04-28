"""
Dynamic threshold adjustment for anomaly detection.
Adjusts thresholds based on recent transaction patterns and feedback.
"""

import os
import yaml
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

from infrastructure.utils import logger
from infrastructure.config import load_paths, get_project_root


class ThresholdAdjuster:
    """Dynamically adjusts anomaly detection thresholds"""
    
    def __init__(self, base_thresholds_path=None):
        """Initialize with base thresholds"""
        paths = load_paths()
        self.base_path = base_thresholds_path or os.path.join(
            get_project_root(),
            paths['models']['anomaly']['thresholds']
        )
        
        # Load base thresholds
        self.thresholds = self._load_base_thresholds()
        
        # Recent scores for adaptation
        self.recent_scores = []
        self.max_scores = 1000  # Keep last 1000 scores
        
        logger.info(f"Initialized threshold adjuster with base thresholds: {self.thresholds}")
    
    def _load_base_thresholds(self) -> Dict[str, float]:
        """Load base thresholds from YAML file"""
        try:
            with open(self.base_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get('thresholds', {
                    'critical': -0.7,
                    'high': -0.4,
                    'medium': -0.2,
                    'low': 0.0
                })
        except Exception as e:
            logger.warning(f"Could not load base thresholds: {str(e)}. Using defaults.")
            return {
                'critical': -0.7,
                'high': -0.4,
                'medium': -0.2,
                'low': 0.0
            }
    
    def add_score(self, score: float, is_fraud: bool = False) -> None:
        """Add a new anomaly score to recent history"""
        self.recent_scores.append({
            'score': score,
            'is_fraud': is_fraud,
            'timestamp': datetime.now().isoformat()
        })
        
        # Trim if needed
        if len(self.recent_scores) > self.max_scores:
            self.recent_scores = self.recent_scores[-self.max_scores:]
    
    def adjust_thresholds(self) -> Dict[str, float]:
        """Adjust thresholds based on recent scores"""
        if len(self.recent_scores) < 100:
            logger.info("Not enough data to adjust thresholds")
            return self.thresholds
        
        # Extract scores
        scores = [s['score'] for s in self.recent_scores]
        fraud_scores = [s['score'] for s in self.recent_scores if s['is_fraud']]
        
        # If we have confirmed fraud cases, use them to adjust
        if fraud_scores:
            # Set critical threshold to capture 90% of fraud
            new_critical = np.percentile(fraud_scores, 10) if len(fraud_scores) >= 10 else self.thresholds['critical']
            
            # Adjust other thresholds proportionally
            spread = abs(new_critical)
            new_thresholds = {
                'critical': new_critical,
                'high': new_critical * 0.7,
                'medium': new_critical * 0.4,
                'low': new_critical * 0.2
            }
        else:
            # Without fraud cases, use overall distribution
            new_thresholds = {
                'critical': np.percentile(scores, 0.1),
                'high': np.percentile(scores, 1),
                'medium': np.percentile(scores, 5),
                'low': np.percentile(scores, 10)
            }
        
        # Apply smoothing to avoid drastic changes
        for key in self.thresholds:
            self.thresholds[key] = 0.8 * self.thresholds[key] + 0.2 * new_thresholds[key]
        
        logger.info(f"Adjusted thresholds: {self.thresholds}")
        return self.thresholds
    
    def save_thresholds(self, path=None) -> None:
        """Save current thresholds to file"""
        save_path = path or self.base_path
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save to YAML
            with open(save_path, 'w') as f:
                yaml.dump({'thresholds': self.thresholds}, f)
                
            logger.info(f"Saved adjusted thresholds to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save thresholds: {str(e)}")