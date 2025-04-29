"""
Unit tests for fraud detection expert systems.
"""

import os
import sys
import unittest
import json
import numpy as np
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experts.common.utils import (
    extract_model_features,
    check_model_compatibility,
    calculate_heuristic_fraud_score,
    calculate_heuristic_anomaly_score,
    safe_predict,
    ModelCache
)
from experts.fraud_classifier.predict import FraudClassifierExpert
from experts.anomaly_detector.detect import AnomalyDetectorExpert
from experts.coordination.mediator import ExpertMediator


class TestCommonUtils(unittest.TestCase):
    """Test common utility functions"""

    def test_extract_model_features(self):
        """Test feature extraction from transaction"""
        # Create mock model
        model = MagicMock()
        model.feature_names_in_ = ['amount', 'hour_of_day', 'distance_from_home']

        # Create test transaction
        transaction = {
            'id': 'test123',
            'amount': 1000,
            'hour_of_day': 14,
            'distance_from_home': 50,
            'extra_field': 'value'
        }

        # Extract features
        features = extract_model_features(model, transaction)

        # Check results
        self.assertEqual(features.shape, (1, 3))
        self.assertEqual(features[0, 0], 1000)
        self.assertEqual(features[0, 1], 14)
        self.assertEqual(features[0, 2], 50)

    def test_extract_model_features_missing(self):
        """Test feature extraction with missing features"""
        # Create mock model
        model = MagicMock()
        model.feature_names_in_ = ['amount', 'hour_of_day', 'missing_feature']

        # Create test transaction
        transaction = {
            'id': 'test123',
            'amount': 1000,
            'hour_of_day': 14
        }

        # Extract features
        features = extract_model_features(model, transaction)

        # Check results
        self.assertEqual(features.shape, (1, 3))
        self.assertEqual(features[0, 0], 1000)
        self.assertEqual(features[0, 1], 14)
        self.assertEqual(features[0, 2], 0)  # Default value for missing feature

    def test_check_model_compatibility(self):
        """Test model compatibility check"""
        # Create mock model
        model = MagicMock()
        model.feature_names_in_ = ['amount', 'hour_of_day', 'distance_from_home']

        # Check with all features available
        is_compatible, missing = check_model_compatibility(
            model, ['amount', 'hour_of_day', 'distance_from_home', 'extra']
        )
        self.assertTrue(is_compatible)
        self.assertEqual(len(missing), 0)

        # Check with missing features
        is_compatible, missing = check_model_compatibility(
            model, ['amount', 'extra']
        )
        self.assertFalse(is_compatible)
        self.assertEqual(len(missing), 2)
        self.assertIn('hour_of_day', missing)
        self.assertIn('distance_from_home', missing)

    def test_calculate_heuristic_fraud_score(self):
        """Test heuristic fraud score calculation"""
        # Test transaction with multiple fraud indicators
        transaction = {
            'amount': 10000,  # High amount
            'hour_of_day': 3,  # Unusual hour
            'country': 'XX',  # Suspicious country
            'distance_from_home': 200,  # High distance
            'used_chip': 0  # No chip
        }

        score = calculate_heuristic_fraud_score(transaction)
        self.assertGreater(score, 0.5)  # Should be high risk

        # Test normal transaction
        normal_transaction = {
            'amount': 100,
            'hour_of_day': 14,
            'country': 'US',
            'distance_from_home': 5,
            'used_chip': 1
        }

        score = calculate_heuristic_fraud_score(normal_transaction)
        self.assertEqual(score, 0.0)  # Should be no risk

    def test_calculate_heuristic_anomaly_score(self):
        """Test heuristic anomaly score calculation"""
        # Test transaction with multiple anomaly indicators
        transaction = {
            'amount': 10000,  # High amount
            'hour_of_day': 3,  # Unusual hour
            'country': 'XX',  # Suspicious country
            'distance_from_home': 200,  # High distance
            'used_chip': 0  # No chip
        }

        score = calculate_heuristic_anomaly_score(transaction)
        self.assertLess(score, 0)  # Should be anomalous (negative score)

        # Test normal transaction
        normal_transaction = {
            'amount': 100,
            'hour_of_day': 14,
            'country': 'US',
            'distance_from_home': 5,
            'used_chip': 1
        }

        score = calculate_heuristic_anomaly_score(normal_transaction)
        self.assertEqual(score, 0.0)  # Should be normal

    def test_safe_predict(self):
        """Test safe prediction with error handling"""
        # Create mock model that works
        model = MagicMock()
        model.predict_proba.return_value = np.array([[0.2, 0.8]])

        # Test successful prediction
        score = safe_predict(model, np.array([[1, 2, 3]]))
        self.assertEqual(score, 0.8)

        # Create mock model that raises exception
        error_model = MagicMock()
        error_model.predict_proba.side_effect = Exception("Test error")

        # Test prediction with error
        score = safe_predict(error_model, np.array([[1, 2, 3]]), fallback_value=0.42)
        self.assertEqual(score, 0.42)  # Should return fallback value

    def test_model_cache(self):
        """Test model prediction cache"""
        cache = ModelCache(max_size=2)

        # Add items to cache
        cache.set('key1', 0.8)
        cache.set('key2', 0.5)

        # Check cache retrieval
        self.assertEqual(cache.get('key1'), 0.8)
        self.assertEqual(cache.get('key2'), 0.5)
        self.assertIsNone(cache.get('nonexistent'))

        # Test cache size limit
        cache.set('key3', 0.3)  # This should evict key1
        self.assertIsNone(cache.get('key1'))
        self.assertEqual(cache.get('key2'), 0.5)
        self.assertEqual(cache.get('key3'), 0.3)

        # Test cache clear
        cache.clear()
        self.assertIsNone(cache.get('key2'))
        self.assertIsNone(cache.get('key3'))


class TestFraudClassifierExpert(unittest.TestCase):
    """Test fraud classifier expert"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock model
        self.model = MagicMock()
        self.model.feature_names_in_ = ['amount', 'hour_of_day', 'distance_from_home']
        self.model.predict_proba.return_value = np.array([[0.2, 0.8]])

        # Create temporary rules file
        self.rules_file = 'test_rules.json'
        rules = {
            'amount_rules': {'high_value_threshold': 5000},
            'velocity_rules': {'max_transactions_per_hour': 5},
            'location_rules': {
                'suspicious_countries': ['XX', 'YY', 'ZZ'],
                'high_distance_from_home': 100,
                'high_distance_from_last_transaction': 50
            },
            'time_rules': {'unusual_hour_start': 1, 'unusual_hour_end': 5},
            'transaction_rules': {
                'high_ratio_to_median_threshold': 3.0,
                'suspicious_payment_methods': {
                    'no_chip': True,
                    'no_pin': True,
                    'online_order': True
                }
            }
        }
        with open(self.rules_file, 'w') as f:
            json.dump(rules, f)

        # Create expert
        self.expert = FraudClassifierExpert(self.model, self.rules_file)

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.rules_file):
            os.remove(self.rules_file)

    def test_apply_rules(self):
        """Test rule application"""
        # Test transaction with multiple rule violations
        transaction = {
            'amount': 10000,  # High amount
            'hour_of_day': 3,  # Unusual hour
            'country': 'XX',  # Suspicious country
            'distance_from_home': 200,  # High distance
            'distance_from_last_transaction': 100,  # High distance
            'ratio_to_median_purchase_price': 5.0,  # High ratio
            'used_chip': 0,  # No chip
            'used_pin_number': 0,  # No PIN
            'online_order': 1  # Online order
        }

        violations = self.expert.apply_rules(transaction)
        self.assertEqual(violations, 9)  # All rules violated

        # Test normal transaction
        normal_transaction = {
            'amount': 100,
            'hour_of_day': 14,
            'country': 'US',
            'distance_from_home': 5,
            'distance_from_last_transaction': 10,
            'ratio_to_median_purchase_price': 1.0,
            'used_chip': 1,
            'used_pin_number': 1,
            'online_order': 0
        }

        violations = self.expert.apply_rules(normal_transaction)
        self.assertEqual(violations, 0)  # No rules violated

    def test_evaluate(self):
        """Test transaction evaluation"""
        # Test high-risk transaction
        transaction = {
            'id': 'test123',
            'amount': 10000,
            'hour_of_day': 3,
            'country': 'XX',
            'distance_from_home': 200
        }

        result = self.expert.evaluate(transaction)

        # Check result structure
        self.assertIn('ml_score', result)
        self.assertIn('rule_violations', result)
        self.assertIn('confidence', result)
        self.assertIn('risk', result)
        self.assertIn('score', result)
        self.assertIn('model_version', result)

        # Check values
        self.assertEqual(result['ml_score'], 0.8)  # From mock model
        self.assertGreater(result['rule_violations'], 0)
        self.assertEqual(result['risk'], 'HIGH')

        # Test caching
        self.model.predict_proba.reset_mock()
        result2 = self.expert.evaluate(transaction)
        self.assertEqual(result2['ml_score'], 0.8)
        self.model.predict_proba.assert_not_called()  # Should use cached result


class TestAnomalyDetectorExpert(unittest.TestCase):
    """Test anomaly detector expert"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock model
        self.model = MagicMock()
        self.model.feature_names_in_ = [
            'distance_from_home',
            'distance_from_last_transaction',
            'ratio_to_median_purchase_price',
            'repeat_retailer',
            'used_chip',
            'used_pin_number',
            'online_order',
            'log_distance_from_home',
            'log_distance_from_last_transaction'
        ]
        self.model.decision_function.return_value = np.array([-0.5])  # Anomalous score

        # Create mock context
        self.context = MagicMock()
        self.context.get_cluster_distance.return_value = 0.1

        # Create expert
        self.expert = AnomalyDetectorExpert(self.model, self.context)

        # Override thresholds for testing - use actual float values, not MagicMock
        self.expert.thresholds = {
            'critical': -0.7,
            'high': -0.4,
            'medium': -0.2,
            'low': 0.0
        }

    def test_calculate_severity(self):
        """Test severity calculation"""
        self.assertEqual(self.expert.calculate_severity(-0.8), 'CRITICAL')
        self.assertEqual(self.expert.calculate_severity(-0.5), 'HIGH')
        self.assertEqual(self.expert.calculate_severity(-0.3), 'MEDIUM')
        self.assertEqual(self.expert.calculate_severity(-0.1), 'LOW')
        self.assertEqual(self.expert.calculate_severity(0.1), 'NORMAL')

    def test_analyze(self):
        """Test transaction analysis"""
        # Test anomalous transaction
        transaction = {
            'id': 'test123',
            'amount': 10000,
            'hour_of_day': 3,
            'country': 'XX',
            'distance_from_home': 200
        }

        result = self.expert.analyze(transaction)

        # Check result structure
        self.assertIn('raw_score', result)
        self.assertIn('severity', result)
        self.assertIn('cluster_deviation', result)
        self.assertIn('anomaly_score', result)
        self.assertIn('model_version', result)
        self.assertIn('thresholds', result)

        # Check values
        self.assertEqual(result['raw_score'], -0.5)  # From mock model
        self.assertEqual(result['severity'], 'HIGH')
        self.assertEqual(result['cluster_deviation'], 0.1)  # From mock context

        # Test caching
        self.model.decision_function.reset_mock()
        result2 = self.expert.analyze(transaction)
        self.assertEqual(result2['raw_score'], -0.5)
        self.model.decision_function.assert_not_called()  # Should use cached result

    def test_analyze_with_heuristic_fallback(self):
        """Test analysis with fallback to heuristic scoring"""
        # Create a model that raises an exception
        error_model = MagicMock()
        error_model.feature_names_in_ = [
            'distance_from_home',
            'distance_from_last_transaction',
            'ratio_to_median_purchase_price',
            'repeat_retailer',
            'used_chip',
            'used_pin_number',
            'online_order'
        ]
        error_model.decision_function.side_effect = Exception("Test error")

        # Create expert with error model
        expert = AnomalyDetectorExpert(error_model, self.context)
        # Use actual float values for thresholds
        expert.thresholds = {
            'critical': -0.7,
            'high': -0.4,
            'medium': -0.2,
            'low': 0.0
        }

        # Test transaction with core features
        transaction = {
            'id': 'test123',
            'distance_from_home': 200,
            'distance_from_last_transaction': 50,
            'ratio_to_median_purchase_price': 3.0,
            'repeat_retailer': 0,
            'used_chip': 0,
            'used_pin_number': 0,
            'online_order': 1
        }

        result = expert.analyze(transaction)

        # Should use heuristic score
        self.assertIn('raw_score', result)
        self.assertLess(result['raw_score'], 0)  # Should be negative (anomalous)
        self.assertIn('severity', result)
        self.assertNotEqual(result['severity'], 'NORMAL')  # Should detect anomaly


class TestExpertMediator(unittest.TestCase):
    """Test expert mediator"""

    def setUp(self):
        """Set up test fixtures"""
        # Create mock experts
        self.classifier = MagicMock()
        self.detector = MagicMock()

        # Create mediator
        self.mediator = ExpertMediator(self.classifier, self.detector)

        # Set up mock fusion rules
        self.mediator.fusion_rules = {
            'weighted_rules': {
                'default': {
                    'classifier_weight': 0.7,
                    'anomaly_weight': 0.3,
                    'threshold_review': 0.5,
                    'threshold_block': 0.8
                }
            },
            'priority_rules': [
                {
                    'name': 'high_confidence_classifier',
                    'condition': 'classifier.confidence > 0.9',
                    'action': 'BLOCK',
                    'description': 'Block when classifier has high confidence'
                },
                {
                    'name': 'critical_anomaly',
                    'condition': 'anomaly.severity == "CRITICAL"',
                    'action': 'BLOCK',
                    'description': 'Block when anomaly detector finds critical anomaly'
                }
            ],
            'agreement_rules': [
                {
                    'name': 'both_high_risk',
                    'condition': 'classifier.risk == "HIGH" and anomaly.severity in ["HIGH", "CRITICAL"]',
                    'action': 'BLOCK',
                    'description': 'Block when both experts indicate high risk'
                }
            ]
        }

    def test_resolve_conflict_priority_rules(self):
        """Test conflict resolution with priority rules"""
        # Test high confidence classifier
        classifier_result = {'confidence': 0.95, 'risk': 'HIGH', 'score': 0.95}
        detector_result = {'severity': 'MEDIUM', 'anomaly_score': 0.3}

        decision, explanation = self.mediator.resolve_conflict(classifier_result, detector_result)
        self.assertEqual(decision, 'BLOCK')
        self.assertIn('classifier has high confidence', explanation.lower())

        # Test critical anomaly
        classifier_result = {'confidence': 0.7, 'risk': 'MEDIUM', 'score': 0.7}
        detector_result = {'severity': 'CRITICAL', 'anomaly_score': 0.8}

        decision, explanation = self.mediator.resolve_conflict(classifier_result, detector_result)
        self.assertEqual(decision, 'BLOCK')
        self.assertIn('critical', explanation.lower())

    def test_resolve_conflict_agreement_rules(self):
        """Test conflict resolution with agreement rules"""
        # Test both experts agree on high risk
        classifier_result = {'confidence': 0.8, 'risk': 'HIGH', 'score': 0.8}
        detector_result = {'severity': 'HIGH', 'anomaly_score': 0.6}

        decision, explanation = self.mediator.resolve_conflict(classifier_result, detector_result)
        self.assertEqual(decision, 'BLOCK')
        self.assertIn('both experts', explanation.lower())

    def test_resolve_conflict_weighted_scoring(self):
        """Test conflict resolution with weighted scoring"""
        # Test medium risk (review)
        classifier_result = {'confidence': 0.6, 'risk': 'MEDIUM', 'score': 0.6}
        detector_result = {'severity': 'LOW', 'anomaly_score': 0.2}

        decision, explanation = self.mediator.resolve_conflict(classifier_result, detector_result)
        self.assertEqual(decision, 'REVIEW')
        self.assertIn('combined risk score', explanation.lower())

        # Test low risk (allow)
        classifier_result = {'confidence': 0.3, 'risk': 'LOW', 'score': 0.3}
        detector_result = {'severity': 'NORMAL', 'anomaly_score': 0.1}

        decision, explanation = self.mediator.resolve_conflict(classifier_result, detector_result)
        self.assertEqual(decision, 'ALLOW')
        self.assertIn('below review threshold', explanation.lower())

    def test_process_transaction(self):
        """Test transaction processing workflow"""
        # Set up mock expert responses
        self.classifier.evaluate.return_value = {
            'ml_score': 0.8,
            'rule_violations': 3,
            'confidence': 0.9,
            'risk': 'HIGH',
            'score': 0.9
        }

        self.detector.analyze.return_value = {
            'raw_score': -0.5,
            'severity': 'HIGH',
            'cluster_deviation': 0.1,
            'anomaly_score': 0.6
        }

        # Mock context
        self.mediator.context.check_transaction_context.return_value = {
            'user_known': True,
            'transaction_velocity': 2
        }

        # Test transaction
        transaction = {
            'id': 'test123',
            'amount': 5000,
            'user_id': 'user456'
        }

        result = self.mediator.process_transaction(transaction)

        # Check result structure
        self.assertIn('transaction_id', result)
        self.assertIn('classifier', result)
        self.assertIn('anomaly', result)
        self.assertIn('decision', result)
        self.assertIn('explanation', result)

        # Check values
        self.assertEqual(result['transaction_id'], 'test123')
        self.assertEqual(result['decision'], 'BLOCK')

        # Check caching
        self.classifier.evaluate.reset_mock()
        self.detector.analyze.reset_mock()

        result2 = self.mediator.process_transaction(transaction)
        self.assertEqual(result2['decision'], 'BLOCK')
        self.classifier.evaluate.assert_not_called()  # Should use cached result
        self.detector.analyze.assert_not_called()  # Should use cached result


if __name__ == '__main__':
    unittest.main()
