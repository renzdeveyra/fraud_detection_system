"""
Context manager for fraud detection system.
Handles transaction context and fraud patterns based on core dataset features:
- distance_from_home
- distance_from_last_transaction
- ratio_to_median_purchase_price
- repeat_retailer
- used_chip
- used_pin_number
- online_order
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime

from infrastructure.utils import logger
from infrastructure.config import load_paths, get_project_root


def get_recent_frauds(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get recent fraud patterns from the file.

    Args:
        limit: Maximum number of fraud patterns to return

    Returns:
        List of recent fraud patterns
    """
    paths = load_paths()
    file_path = os.path.join(
        get_project_root(),
        paths.get('shared', {}).get('fraud_history', 'experts/coordination/shared_context/recent_frauds.json')
    )

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Return only the specified number of recent frauds
        recent_frauds = data.get("recent_frauds", [])[:limit]
        logger.info(f"Loaded {len(recent_frauds)} recent frauds from {file_path}")
        return recent_frauds
    except Exception as e:
        logger.warning(f"Error loading recent frauds: {str(e)}")
        return []


def add_fraud(transaction: Dict[str, Any], classifier_score: float,
              anomaly_score: float, rule_violations: int) -> bool:
    """
    Add a fraud transaction to the recent frauds list.

    Args:
        transaction: Transaction dictionary
        classifier_score: Fraud classifier score
        anomaly_score: Anomaly detector score
        rule_violations: Number of rule violations

    Returns:
        bool: True if successful, False otherwise
    """
    paths = load_paths()
    file_path = os.path.join(
        get_project_root(),
        paths.get('shared', {}).get('fraud_history', 'experts/coordination/shared_context/recent_frauds.json')
    )

    try:
        # Load existing data
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except:
            data = {
                "recent_frauds": [],
                "last_updated": datetime.now().isoformat()
            }

        # Create fraud entry with only core features
        fraud_entry = {
            "transaction_id": transaction.get('id', '') or transaction.get('transaction_id', ''),
            "timestamp": datetime.now().isoformat(),
            "transaction_features": {
                "distance_from_home": transaction.get('distance_from_home', 0),
                "distance_from_last_transaction": transaction.get('distance_from_last_transaction', 0),
                "ratio_to_median_purchase_price": transaction.get('ratio_to_median_purchase_price', 1.0),
                "repeat_retailer": transaction.get('repeat_retailer', 0),
                "used_chip": transaction.get('used_chip', 1),
                "used_pin_number": transaction.get('used_pin_number', 1),
                "online_order": transaction.get('online_order', 0)
            },
            "classifier_score": classifier_score,
            "anomaly_score": anomaly_score,
            "rule_violations": rule_violations
        }

        # Add to recent frauds
        data["recent_frauds"].insert(0, fraud_entry)

        # Limit to 100 recent frauds
        data["recent_frauds"] = data["recent_frauds"][:100]

        # Update timestamp
        data["last_updated"] = datetime.now().isoformat()

        # Save data
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Added fraud transaction {transaction.get('id', '') or transaction.get('transaction_id', '')} to recent frauds")
        return True
    except Exception as e:
        logger.error(f"Error adding fraud: {str(e)}")
        return False


def check_transaction_context(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check transaction context based on core dataset features:
    - distance_from_home
    - distance_from_last_transaction
    - ratio_to_median_purchase_price
    - repeat_retailer
    - used_chip
    - used_pin_number
    - online_order

    Args:
        transaction: Transaction dictionary

    Returns:
        Dict containing context check results
    """
    # Default context with focus on core features
    context = {
        'distance_from_home_unusual': False,
        'distance_from_last_transaction_unusual': False,
        'purchase_price_ratio_unusual': False,
        'payment_method_unusual': False,
        'is_new_retailer': False,
        'matches_recent_fraud': False
    }

    # Load rules to get thresholds
    import json
    import os
    from infrastructure.config import get_project_root, load_paths

    paths = load_paths()
    rules_path = os.path.join(
        get_project_root(),
        paths.get('models', {}).get('classifier', {}).get('rules', 'experts/fraud_classifier/rules/static_rules.json')
    )

    try:
        with open(rules_path, 'r') as f:
            rules = json.load(f)
    except Exception as e:
        logger.warning(f"Error loading rules: {str(e)}")
        rules = {}

    # Get thresholds from rules
    distance_rules = rules.get('distance_rules', {})
    transaction_pattern_rules = rules.get('transaction_pattern_rules', {})

    # Check unusual distances
    if 'distance_from_home' in transaction:
        context['distance_from_home_unusual'] = transaction['distance_from_home'] > distance_rules.get('high_distance_from_home', 100)

    if 'distance_from_last_transaction' in transaction:
        context['distance_from_last_transaction_unusual'] = transaction['distance_from_last_transaction'] > distance_rules.get('high_distance_from_last_transaction', 50)

    # Check unusual purchase price ratio
    if 'ratio_to_median_purchase_price' in transaction:
        context['purchase_price_ratio_unusual'] = transaction['ratio_to_median_purchase_price'] > transaction_pattern_rules.get('high_ratio_to_median_threshold', 3.0)

    # Check if this is a new retailer
    if 'repeat_retailer' in transaction:
        context['is_new_retailer'] = transaction['repeat_retailer'] == 0

    # Check unusual payment methods
    payment_method_unusual = False
    payment_method_rules = rules.get('payment_method_rules', {})

    if payment_method_rules.get('no_chip', True) and 'used_chip' in transaction and transaction['used_chip'] == 0:
        payment_method_unusual = True

    if payment_method_rules.get('no_pin', True) and 'used_pin_number' in transaction and transaction['used_pin_number'] == 0:
        payment_method_unusual = True

    if payment_method_rules.get('online_order', True) and 'online_order' in transaction and transaction['online_order'] == 1:
        payment_method_unusual = True

    context['payment_method_unusual'] = payment_method_unusual

    # Check if transaction matches recent fraud patterns
    recent_frauds = get_recent_frauds(limit=10)
    for fraud in recent_frauds:
        fraud_features = fraud.get('transaction_features', {})
        matches = 0
        total = 0

        # Compare core features
        for key in ['distance_from_home', 'distance_from_last_transaction',
                   'ratio_to_median_purchase_price', 'repeat_retailer',
                   'used_chip', 'used_pin_number', 'online_order']:
            if key in fraud_features and key in transaction:
                total += 1
                # For numeric features, check if within reasonable range
                if key in ['distance_from_home', 'distance_from_last_transaction', 'ratio_to_median_purchase_price']:
                    if abs(fraud_features[key] - transaction[key]) / max(1, fraud_features[key]) < 0.3:
                        matches += 1
                else:
                    # For binary features, check exact match
                    if fraud_features[key] == transaction[key]:
                        matches += 1

        # If more than 70% of features match, consider it a match
        if total > 0 and matches / total > 0.7:
            context['matches_recent_fraud'] = True
            break

    return context
