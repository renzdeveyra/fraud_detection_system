import json
from sklearn.base import BaseEstimator
from typing import Dict

class FraudClassifierExpert:
    def __init__(self, model: BaseEstimator, rules_path: str):
        self.model = model
        with open(rules_path) as f:
            self.rules = json.load(f)
        # Initialize empty context (will be set by mediator)
        self.context = None

    def apply_rules(self, transaction: Dict) -> int:
        """Apply domain-specific fraud rules"""
        violations = 0

        # Rule 1: Large amount (simplified since we don't have user history yet)
        if transaction.get('amount', 0) > self.rules.get('amount_rules', {}).get('high_value_threshold', 5000):
            violations += 1

        # Rule 2: High-velocity transactions (if count_1h is available)
        if transaction.get('count_1h', 0) > self.rules.get('velocity_rules', {}).get('max_transactions_per_hour', 5):
            violations += 1

        # Rule 3: Suspicious country
        if transaction.get('country') in self.rules.get('location_rules', {}).get('suspicious_countries', []):
            violations += 1

        # Rule 4: Unusual hour
        hour = transaction.get('hour_of_day', -1)
        if (hour >= self.rules.get('time_rules', {}).get('unusual_hour_start', 1) and
            hour <= self.rules.get('time_rules', {}).get('unusual_hour_end', 5)):
            violations += 1

        # Rule 5: High distance from home
        if transaction.get('distance_from_home', 0) > self.rules.get('location_rules', {}).get('high_distance_from_home', 100):
            violations += 1

        # Rule 6: High distance from last transaction
        if transaction.get('distance_from_last_transaction', 0) > self.rules.get('location_rules', {}).get('high_distance_from_last_transaction', 50):
            violations += 1

        # Rule 7: Unusual ratio to median purchase price
        if transaction.get('ratio_to_median_purchase_price', 1.0) > self.rules.get('transaction_rules', {}).get('high_ratio_to_median_threshold', 3.0):
            violations += 1

        # Rule 8: Suspicious payment methods
        suspicious_methods = self.rules.get('transaction_rules', {}).get('suspicious_payment_methods', {})

        # Check for no chip usage when expected
        if suspicious_methods.get('no_chip', False) and transaction.get('used_chip', 1) == 0:
            violations += 1

        # Check for no PIN usage when expected
        if suspicious_methods.get('no_pin', False) and transaction.get('used_pin_number', 1) == 0:
            violations += 1

        # Check for online order (potentially higher risk)
        if suspicious_methods.get('online_order', False) and transaction.get('online_order', 0) == 1:
            violations += 1

        return violations

    def evaluate(self, transaction: Dict) -> Dict:
        """Combined ML + rules evaluation"""
        # Extract numeric features for the model
        features = []
        for feature in self.model.feature_names_in_:
            if feature in transaction:
                features.append(transaction[feature])
            else:
                features.append(0)  # Default value for missing features

        # Make prediction
        try:
            ml_score = self.model.predict_proba([features])[0][1]
        except Exception as e:
            # If prediction fails, use a default score
            print(f"Error in ML prediction: {str(e)}")
            ml_score = 0.5

        rule_violations = self.apply_rules(transaction)

        # Calculate confidence and risk level
        confidence = min(1.0, ml_score + 0.2 * rule_violations)

        if confidence > 0.8:
            risk = 'HIGH'
        elif confidence > 0.5:
            risk = 'MEDIUM'
        else:
            risk = 'LOW'

        return {
            'ml_score': float(ml_score),
            'rule_violations': rule_violations,
            'confidence': float(confidence),
            'risk': risk,
            'score': float(confidence)  # For compatibility with mediator
        }