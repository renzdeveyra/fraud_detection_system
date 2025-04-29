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
from experts.common.utils import (
    extract_model_features,
    check_model_compatibility,
    calculate_heuristic_anomaly_score,
    safe_predict,
    get_model_version,
    add_model_version,
    ModelCache,
    generate_cache_key
)


class AnomalyDetectorExpert:
    """
    Expert system for anomaly detection.

    Uses unsupervised learning to identify unusual transactions
    that deviate from normal patterns, even if they don't match
    known fraud patterns.
    """

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

        # Add version if not present
        if not hasattr(self.model, 'version'):
            add_model_version(self.model, f"anomaly_detector_v1")

        # Initialize threshold adjuster
        self.threshold_adjuster = ThresholdAdjuster()

        # Load thresholds
        self.thresholds = self.threshold_adjuster.thresholds

        # Initialize prediction cache
        self.prediction_cache = ModelCache(max_size=1000)

        logger.info(f"Initialized anomaly detector expert with model version: {get_model_version(self.model)}")
        logger.info(f"Current thresholds: {self.thresholds}")

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

        Applies unsupervised anomaly detection to identify unusual
        transactions that deviate from normal patterns.

        Args:
            transaction: Transaction data dictionary

        Returns:
            Dictionary containing analysis results
        """
        # Check if we have this transaction in cache
        transaction_id = transaction.get('id', '') or transaction.get('transaction_id', '')
        cache_key = f"{transaction_id}_{get_model_version(self.model)}"
        cached_result = self.prediction_cache.get(cache_key)

        if cached_result:
            raw_score = cached_result
            logger.debug(f"Using cached anomaly score for transaction {transaction_id}")
        else:
            # Calculate a heuristic anomaly score as a fallback
            heuristic_score = calculate_heuristic_anomaly_score(transaction)

            # Try to use the ML model if possible
            try:
                # Extract features for the model
                features = extract_model_features(self.model, transaction)

                # Check model compatibility with available features
                available_features = [k for k in transaction.keys()
                                     if isinstance(transaction[k], (int, float))]
                is_compatible, _ = check_model_compatibility(self.model, available_features)

                if is_compatible:
                    # Get raw anomaly score
                    raw_score = safe_predict(
                        self.model,
                        features,
                        fallback_value=heuristic_score
                    )
                else:
                    # Use heuristic score if model is not compatible
                    raw_score = heuristic_score

                # Cache the result
                if transaction_id:
                    self.prediction_cache.set(cache_key, raw_score)

            except Exception as e:
                logger.warning(f"Error in anomaly detection: {str(e)}")
                # Use the heuristic score as a fallback
                raw_score = heuristic_score

        # Determine severity based on thresholds
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

        # Return comprehensive analysis results
        return {
            'raw_score': float(raw_score),
            'severity': severity,
            'cluster_deviation': float(cluster_distance),
            'anomaly_score': float(anomaly_score),
            'model_version': get_model_version(self.model),
            'thresholds': {k: float(v) for k, v in self.thresholds.items()}
        }

    # The _calculate_heuristic_score method has been moved to common/utils.py
    # and is now imported as calculate_heuristic_anomaly_score

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
