import json
import numpy as np
from collections import deque
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

from infrastructure.config import load_paths, load_params, get_project_root
from infrastructure.utils.logger import logger


class ContextBuffer:
    """Shared memory between expert systems to store transaction context"""

    def __init__(self, max_size: Optional[int] = None):
        params = load_params()
        paths = load_paths()

        # Initialize buffer size
        self.max_size = max_size or params['memory']['context_buffer_size']
        self.recent_transactions = deque(maxlen=self.max_size)

        # Initialize fraud patterns storage
        fraud_patterns_path = os.path.join(get_project_root(), paths['shared']['fraud_history'])
        os.makedirs(os.path.dirname(fraud_patterns_path), exist_ok=True)

        if os.path.exists(fraud_patterns_path):
            with open(fraud_patterns_path, 'r') as f:
                self.fraud_patterns = json.load(f)
        else:
            self.fraud_patterns = {
                'recent_frauds': [],
                'clusters': []
            }
            self._save_fraud_patterns()

        # Current clustering centroids
        self.centroids = np.array([])

        logger.info("Context buffer initialized")

    def update(self, transaction: Dict[str, Any],
              classifier_result: Dict[str, Any],
              anomaly_result: Dict[str, Any]) -> None:
        """Update context with new transaction data and expert results"""
        # Add to recent transactions
        context_entry = {
            'transaction': transaction,
            'classifier_result': classifier_result,
            'anomaly_result': anomaly_result,
            'timestamp': datetime.now().isoformat()
        }
        self.recent_transactions.append(context_entry)

        # Update fraud patterns if flagged as fraud
        if (classifier_result.get('risk') == 'HIGH' or
            anomaly_result.get('severity') == 'CRITICAL'):
            self._update_fraud_patterns(transaction, classifier_result, anomaly_result)

    def _update_fraud_patterns(self, transaction: Dict[str, Any],
                              classifier_result: Dict[str, Any],
                              anomaly_result: Dict[str, Any]) -> None:
        """Update recorded fraud patterns"""
        fraud_entry = {
            'transaction_features': {k: v for k, v in transaction.items()
                                     if k not in ['id', 'transaction_id']},
            'classifier_insights': classifier_result.get('rule_violations', []),
            'anomaly_insights': {
                'score': anomaly_result.get('raw_score', 0),
                'cluster_deviation': anomaly_result.get('cluster_deviation', 0)
            },
            'timestamp': datetime.now().isoformat()
        }

        # Add to recent frauds list
        self.fraud_patterns['recent_frauds'].append(fraud_entry)

        # Keep only last 100 fraud entries
        self.fraud_patterns['recent_frauds'] = self.fraud_patterns['recent_frauds'][-100:]

        # Save updated patterns
        self._save_fraud_patterns()

    def _save_fraud_patterns(self) -> None:
        """Save fraud patterns to disk"""
        paths = load_paths()
        fraud_patterns_path = os.path.join(get_project_root(), paths['shared']['fraud_history'])

        with open(fraud_patterns_path, 'w') as f:
            json.dump(self.fraud_patterns, f, indent=2)



    def get_recent_frauds(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent fraud entries"""
        return self.fraud_patterns['recent_frauds'][-limit:]

    def get_cluster_distance(self, transaction: Dict[str, Any]) -> float:
        """Calculate distance from transaction to known fraud clusters"""
        if len(self.centroids) == 0:
            return 0.0

        # Extract numeric features
        features = np.array([v for k, v in transaction.items()
                            if isinstance(v, (int, float)) and k not in ['id', 'transaction_id']])

        if len(features) == 0:
            return 0.0

        # Calculate minimum distance to any centroid
        distances = np.linalg.norm(self.centroids - features, axis=1)
        return float(np.min(distances)) if len(distances) > 0 else 0.0

    def update_clusters(self, centroids: np.ndarray) -> None:
        """Update fraud cluster centroids"""
        self.centroids = centroids

        # Store in fraud patterns
        self.fraud_patterns['clusters'] = centroids.tolist() if len(centroids) > 0 else []
        self._save_fraud_patterns()

    def get_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of current context for decision-making"""
        return {
            'recent_transactions_count': len(self.recent_transactions),
            'fraud_patterns_count': len(self.fraud_patterns['recent_frauds']),
            'cluster_count': len(self.centroids)
        }

    def check_transaction_context(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check transaction against context using only core dataset features:
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

        # Check unusual distances
        if 'distance_from_home' in transaction:
            context['distance_from_home_unusual'] = transaction['distance_from_home'] > 100

        if 'distance_from_last_transaction' in transaction:
            context['distance_from_last_transaction_unusual'] = transaction['distance_from_last_transaction'] > 50

        # Check unusual purchase price ratio
        if 'ratio_to_median_purchase_price' in transaction:
            context['purchase_price_ratio_unusual'] = transaction['ratio_to_median_purchase_price'] > 3.0

        # Check if this is a new retailer
        if 'repeat_retailer' in transaction:
            context['is_new_retailer'] = transaction['repeat_retailer'] == 0

        # Check unusual payment methods
        payment_method_unusual = False

        if 'used_chip' in transaction and transaction['used_chip'] == 0:
            payment_method_unusual = True

        if 'used_pin_number' in transaction and transaction['used_pin_number'] == 0:
            payment_method_unusual = True

        if 'online_order' in transaction and transaction['online_order'] == 1:
            payment_method_unusual = True

        context['payment_method_unusual'] = payment_method_unusual

        # Check if transaction matches recent fraud patterns
        recent_frauds = self.get_recent_frauds(limit=10)
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

    def add_fraud(self, transaction: Dict[str, Any],
                 classifier_score: float,
                 anomaly_score: float,
                 rule_violations: int) -> None:
        """
        Add a fraud transaction to the recent frauds list.

        Args:
            transaction: Transaction dictionary
            classifier_score: Fraud classifier score
            anomaly_score: Anomaly detector score
            rule_violations: Number of rule violations
        """
        # Create fraud entry
        fraud_entry = {
            'transaction_features': {k: v for k, v in transaction.items()
                                    if k not in ['id', 'transaction_id']},
            'classifier_score': classifier_score,
            'anomaly_score': anomaly_score,
            'rule_violations': rule_violations,
            'timestamp': datetime.now().isoformat()
        }

        # Add to recent frauds list
        self.fraud_patterns['recent_frauds'].append(fraud_entry)

        # Keep only last 100 fraud entries
        self.fraud_patterns['recent_frauds'] = self.fraud_patterns['recent_frauds'][-100:]

        # Save updated patterns
        self._save_fraud_patterns()

    def __del__(self):
        """Clean up resources"""
        pass