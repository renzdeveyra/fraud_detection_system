"""
Anomaly detector expert for fraud detection system.
Detects unusual transactions using isolation forest model.
"""

import os
import numpy as np
from typing import Dict, Any, List, Optional

from infrastructure.config import load_params, load_paths, get_project_root
from infrastructure.utils import logger
from experts.anomaly_detector.thresholds.dynamic_adjustments import ThresholdAdjuster


class AnomalyDetectorExpert:
    """Expert system for anomaly detection"""

    def __init__(self, model, context):
        """
        Initialize the anomaly detector expert.

        Args:
            model: Trained anomaly detection model
            context: Shared context buffer
        """
        self.model = model
        self.context = context
        self.params = load_params()['anomaly']

        # Initialize threshold adjuster
        self.threshold_adjuster = ThresholdAdjuster()

        # Load thresholds
        self.thresholds = self.threshold_adjuster.thresholds

        logger.info(f"Initialized anomaly detector expert with thresholds: {self.thresholds}")

    def calculate_severity(self, score: float) -> str:
        """
        Convert raw score to risk categories.

        Args:
            score: Anomaly score from model

        Returns:
            str: Severity level (CRITICAL, HIGH, MEDIUM, LOW, NORMAL)
        """
        if score < self.thresholds['critical']:
            return 'CRITICAL'
        elif score < self.thresholds['high']:
            return 'HIGH'
        elif score < self.thresholds['medium']:
            return 'MEDIUM'
        elif score < self.thresholds['low']:
            return 'LOW'
        else:
            return 'NORMAL'

    def analyze(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Full anomaly analysis of a transaction.

        Args:
            transaction: Transaction data

        Returns:
            Dict containing analysis results
        """
        # Extract numeric features for the model
        features = []
        if hasattr(self.model, 'feature_names_in_'):
            for feature in self.model.feature_names_in_:
                if feature in transaction:
                    features.append(transaction[feature])
                else:
                    features.append(0)  # Default value for missing features
        else:
            # If model doesn't have feature names, extract all numeric values
            features = [v for k, v in transaction.items()
                       if isinstance(v, (int, float)) and k not in ['id', 'user_id']]

        # Get raw anomaly score
        try:
            raw_score = self.model.decision_function([features])[0]
        except Exception as e:
            logger.warning(f"Error in anomaly detection: {str(e)}")
            raw_score = 0.0  # Default score

        # Determine severity
        severity = self.calculate_severity(raw_score)

        # Compare to cluster centroids (if available)
        cluster_distance = 0
        try:
            if hasattr(self.context, 'get_cluster_distance'):
                cluster_distance = self.context.get_cluster_distance(transaction)
        except Exception as e:
            logger.warning(f"Error getting cluster distance: {str(e)}")
            cluster_distance = 0

        # Calculate final anomaly score
        anomaly_score = np.clip(raw_score + 0.5 * cluster_distance, -1, 1)

        # Add score to threshold adjuster for future adjustments
        self.threshold_adjuster.add_score(
            score=raw_score,
            is_fraud=transaction.get('is_fraud', False)
        )

        # Return analysis results
        return {
            'raw_score': float(raw_score),
            'severity': severity,
            'cluster_deviation': float(cluster_distance),
            'anomaly_score': float(anomaly_score)
        }

    def update_thresholds(self, auto_adjust: bool = True) -> Dict[str, float]:
        """
        Update anomaly detection thresholds.

        Args:
            auto_adjust: Whether to automatically adjust thresholds based on recent data

        Returns:
            Dict containing updated thresholds
        """
        if auto_adjust:
            # Automatically adjust thresholds based on recent data
            self.thresholds = self.threshold_adjuster.adjust_thresholds()

        # Save updated thresholds
        self.threshold_adjuster.save_thresholds()

        logger.info(f"Updated anomaly thresholds: {self.thresholds}")
        return self.thresholds

    def set_threshold(self, level: str, value: float) -> None:
        """
        Set a specific threshold value.

        Args:
            level: Threshold level (critical, high, medium, low)
            value: New threshold value
        """
        if level not in self.thresholds:
            logger.warning(f"Unknown threshold level: {level}")
            return

        self.thresholds[level] = value
        self.threshold_adjuster.thresholds[level] = value
        self.threshold_adjuster.save_thresholds()

        logger.info(f"Set {level} threshold to {value}")
