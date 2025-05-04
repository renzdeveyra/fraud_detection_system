"""
Fraud Classifier Expert for fraud detection system.
Implements supervised learning for known fraud pattern detection.
"""

import json
import numpy as np
from sklearn.base import BaseEstimator
from typing import Dict, List, Any, Optional

from infrastructure.utils import logger
from experts.common.utils import (
    extract_model_features,
    check_model_compatibility,
    calculate_heuristic_fraud_score,
    safe_predict,
    get_model_version,
    add_model_version,
    ModelCache,
    generate_cache_key
)

class FraudClassifierExpert:
    """
    Expert system for supervised fraud classification.

    Combines machine learning predictions with rule-based evaluation
    to identify fraudulent transactions based on known patterns.
    """

    def __init__(self, model: BaseEstimator, rules_path: str):
        """
        Initialize the fraud classifier expert.

        Args:
            model: Trained classifier model
            rules_path: Path to static rules JSON file
        """
        self.model = model

        # Add version if not present
        if not hasattr(self.model, 'version'):
            add_model_version(self.model, f"fraud_classifier_v1")

        # Load rules
        with open(rules_path) as f:
            self.rules = json.load(f)

        # Initialize empty context (will be set by mediator)
        self.context = None

        # Initialize prediction cache
        self.prediction_cache = ModelCache(max_size=1000)

        logger.info(f"Initialized fraud classifier expert with model version: {get_model_version(self.model)}")

    def apply_rules(self, transaction: Dict) -> int:
        """
        Apply domain-specific fraud rules based on core dataset features:
        - distance_from_home
        - distance_from_last_transaction
        - ratio_to_median_purchase_price
        - repeat_retailer
        - used_chip
        - used_pin_number
        - online_order
        """
        violations = 0

        # Rule 1: High distance from home
        distance_rules = self.rules.get('distance_rules', {})
        if transaction.get('distance_from_home', 0) > distance_rules.get('high_distance_from_home', 100):
            violations += 1

        # Rule 2: High distance from last transaction
        if transaction.get('distance_from_last_transaction', 0) > distance_rules.get('high_distance_from_last_transaction', 50):
            violations += 1

        # Rule 3: Unusual ratio to median purchase price
        transaction_pattern_rules = self.rules.get('transaction_pattern_rules', {})
        if transaction.get('ratio_to_median_purchase_price', 1.0) > transaction_pattern_rules.get('high_ratio_to_median_threshold', 3.0):
            violations += 1

        # Rule 4: Non-repeat retailer (new merchant)
        if transaction_pattern_rules.get('new_retailer_flag', True) and transaction.get('repeat_retailer', 1) == 0:
            violations += 1

        # Rule 5: Suspicious payment methods
        payment_method_rules = self.rules.get('payment_method_rules', {})

        # Check for no chip usage when expected
        if payment_method_rules.get('no_chip', False) and transaction.get('used_chip', 1) == 0:
            violations += 1

        # Check for no PIN usage when expected
        if payment_method_rules.get('no_pin', False) and transaction.get('used_pin_number', 1) == 0:
            violations += 1

        # Check for online order (potentially higher risk)
        if payment_method_rules.get('online_order', False) and transaction.get('online_order', 0) == 1:
            violations += 1

        return violations

    def evaluate(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combined ML + rules evaluation of a transaction.

        Applies both machine learning prediction and rule-based evaluation
        to determine the fraud risk of a transaction.

        Args:
            transaction: Transaction data dictionary

        Returns:
            Dictionary containing evaluation results
        """
        # Apply rules first (this is fast and reliable)
        rule_violations = self.apply_rules(transaction)

        # Check if we have this transaction in cache
        transaction_id = transaction.get('id', '') or transaction.get('transaction_id', '')
        cache_key = f"{transaction_id}_{get_model_version(self.model)}"
        cached_result = self.prediction_cache.get(cache_key)

        if cached_result:
            ml_score = cached_result
            logger.debug(f"Using cached prediction for transaction {transaction_id}")
        else:
            # Try ML prediction if possible
            try:
                # Extract features for the model
                features = extract_model_features(self.model, transaction)

                # Make prediction
                ml_score = safe_predict(self.model, features, fallback_value=0.5)

                # Cache the result
                if transaction_id:
                    self.prediction_cache.set(cache_key, ml_score)

            except Exception as e:
                # If prediction fails, use a rule-based score
                logger.warning(f"Error in ML prediction: {str(e)}")
                # Use rule violations and heuristic score to estimate fraud probability
                heuristic_score = calculate_heuristic_fraud_score(transaction)
                ml_score = min(0.9, (heuristic_score + 0.15 * rule_violations) / 2)

        # Calculate confidence and risk level
        confidence = min(1.0, ml_score + 0.2 * rule_violations)

        # Determine risk level
        if confidence > 0.8:
            risk = 'HIGH'
        elif confidence > 0.5:
            risk = 'MEDIUM'
        else:
            risk = 'LOW'

        # Get feature importance if available
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_'):
            for i, feature in enumerate(self.model.feature_names_in_):
                feature_importance[feature] = float(self.model.feature_importances_[i])
        elif hasattr(self.model, 'coef_') and hasattr(self.model, 'feature_names_in_'):
            for i, feature in enumerate(self.model.feature_names_in_):
                feature_importance[feature] = float(self.model.coef_[0][i])

        # Return comprehensive evaluation results
        return {
            'ml_score': float(ml_score),
            'rule_violations': rule_violations,
            'confidence': float(confidence),
            'risk': risk,
            'score': float(confidence),  # For compatibility with mediator
            'model_version': get_model_version(self.model),
            'feature_importance': feature_importance if feature_importance else None
        }