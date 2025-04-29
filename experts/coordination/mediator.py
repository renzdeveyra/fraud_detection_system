"""
Expert Mediator for fraud detection system.
Coordinates decisions between multiple expert systems.
"""

import json
import yaml
import os
from typing import Dict, List, Any, Optional, Tuple

from infrastructure.utils import logger
from infrastructure.memory.context_buffer import ContextBuffer
from infrastructure.config import load_paths, get_project_root

class ExpertMediator:
    """
    Coordinates decisions between multiple expert systems.

    The Expert Mediator is responsible for:
    1. Collecting opinions from different expert systems
    2. Resolving conflicts between expert opinions
    3. Making final decisions about transactions
    4. Maintaining shared context between experts
    5. Explaining decision rationale

    This implementation uses a weighted scoring approach with configurable
    fusion rules to combine the outputs of a supervised classifier and
    an unsupervised anomaly detector.
    """

    def __init__(self, classifier, detector):
        """
        Initialize the expert mediator.

        Args:
            classifier: Fraud classifier expert
            detector: Anomaly detector expert
        """
        self.classifier = classifier
        self.detector = detector

        # Load fusion rules if available
        self.fusion_rules = self._load_fusion_rules()

        # Create a new context if not available
        from infrastructure.memory import ContextBuffer
        self.context = ContextBuffer()

        # Share context with experts
        self.classifier.context = self.context
        self.detector.context = self.context

        # Initialize decision cache
        self.decision_cache = {}
        self.max_cache_size = 1000

        logger.info("Initialized expert mediator with fusion rules")

    def _load_fusion_rules(self) -> Dict[str, Any]:
        """
        Load fusion rules from YAML configuration file.

        Returns:
            Dictionary containing fusion rules
        """
        try:
            paths = load_paths()
            rules_path = os.path.join(
                get_project_root(),
                paths.get('shared', {}).get('fusion_rules', 'experts/coordination/fusion_rules.yaml')
            )

            if os.path.exists(rules_path):
                with open(rules_path, 'r') as f:
                    rules = yaml.safe_load(f)
                logger.info(f"Loaded fusion rules from {rules_path}")
                return rules
            else:
                logger.warning(f"Fusion rules file not found at {rules_path}, using defaults")
        except Exception as e:
            logger.warning(f"Error loading fusion rules: {str(e)}, using defaults")

        # Default rules
        return {
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

    def resolve_conflict(self,
                        classifier_result: Dict[str, Any],
                        detector_result: Dict[str, Any],
                        transaction_context: Dict[str, Any] = None) -> Tuple[str, str]:
        """
        Apply fusion rules to resolve conflicts between expert opinions.

        This method implements a sophisticated decision-making process:
        1. First applies priority rules that can override other decisions
        2. Then checks for agreement between experts
        3. Applies context-based rules if context is available
        4. Falls back to weighted scoring for final decision

        Args:
            classifier_result: Results from fraud classifier expert
            detector_result: Results from anomaly detector expert
            transaction_context: Optional context information about the transaction

        Returns:
            Tuple of (decision, explanation)
        """
        if transaction_context is None:
            transaction_context = {}

        explanation = []
        decision = None

        # Determine which weighted rules to use based on core dataset features
        weighted_rule_set = 'default'
        if transaction_context.get('distance_from_home_unusual', False) or \
           transaction_context.get('distance_from_last_transaction_unusual', False):
            weighted_rule_set = 'unusual_distance'
        elif transaction_context.get('payment_method_unusual', False):
            weighted_rule_set = 'unusual_payment'

        # Get weights and thresholds
        weighted_rules = self.fusion_rules.get('weighted_rules', {}).get(weighted_rule_set, {
            'classifier_weight': 0.7,
            'anomaly_weight': 0.3,
            'threshold_review': 0.5,
            'threshold_block': 0.8
        })

        # Step 1: Apply priority rules first
        for rule in self.fusion_rules.get('priority_rules', []):
            # Simple rule evaluation (in a real system, this would use a proper rule engine)
            if rule['condition'] == 'classifier.confidence > 0.9':
                if classifier_result.get('confidence', 0) > 0.9:
                    decision = rule['action']
                    explanation.append(rule['description'])
                    explanation.append(f"Classifier confidence: {classifier_result.get('confidence', 0):.2f}")
                    break

            elif rule['condition'] == 'anomaly.severity == "CRITICAL"':
                if detector_result.get('severity') == 'CRITICAL':
                    decision = rule['action']
                    explanation.append(rule['description'])
                    break

            elif rule['condition'] == 'context.matches_recent_fraud == true':
                if transaction_context.get('matches_recent_fraud', False):
                    decision = rule['action']
                    explanation.append(rule['description'])
                    break

        # Step 2: If no priority rule matched, check agreement rules
        if decision is None:
            for rule in self.fusion_rules.get('agreement_rules', []):
                if rule['condition'] == 'classifier.risk == "HIGH" and anomaly.severity in ["HIGH", "CRITICAL"]':
                    if (classifier_result.get('risk') == 'HIGH' and
                        detector_result.get('severity') in ['HIGH', 'CRITICAL']):
                        decision = rule['action']
                        explanation.append(rule['description'])
                        break

                elif rule['condition'] == 'classifier.risk == "MEDIUM" and anomaly.severity == "MEDIUM"':
                    if (classifier_result.get('risk') == 'MEDIUM' and
                        detector_result.get('severity') == 'MEDIUM'):
                        decision = rule['action']
                        explanation.append(rule['description'])
                        break

                elif rule['condition'] == 'classifier.risk == "LOW" and anomaly.severity in ["LOW", "NORMAL"]':
                    if (classifier_result.get('risk') == 'LOW' and
                        detector_result.get('severity') in ['LOW', 'NORMAL']):
                        decision = rule['action']
                        explanation.append(rule['description'])
                        break

                elif rule['condition'] == 'classifier.risk == "HIGH" and anomaly.severity == "NORMAL" or classifier.risk == "LOW" and anomaly.severity == "CRITICAL"':
                    if ((classifier_result.get('risk') == 'HIGH' and detector_result.get('severity') == 'NORMAL') or
                        (classifier_result.get('risk') == 'LOW' and detector_result.get('severity') == 'CRITICAL')):
                        decision = rule['action']
                        explanation.append(rule['description'])
                        break

        # Step 3: If still no decision, check context rules based on core dataset features
        if decision is None and transaction_context:
            for rule in self.fusion_rules.get('context_rules', []):
                if rule['condition'] == 'context.distance_from_home_unusual == true':
                    if transaction_context.get('distance_from_home_unusual', False):
                        decision = rule['action']
                        explanation.append(rule['description'])
                        break

                elif rule['condition'] == 'context.distance_from_last_transaction_unusual == true':
                    if transaction_context.get('distance_from_last_transaction_unusual', False):
                        decision = rule['action']
                        explanation.append(rule['description'])
                        break

                elif rule['condition'] == 'context.purchase_price_ratio_unusual == true':
                    if transaction_context.get('purchase_price_ratio_unusual', False):
                        decision = rule['action']
                        explanation.append(rule['description'])
                        break

                elif rule['condition'] == 'context.payment_method_unusual == true':
                    if transaction_context.get('payment_method_unusual', False):
                        decision = rule['action']
                        explanation.append(rule['description'])
                        break

                elif rule['condition'] == 'context.is_new_retailer == true':
                    if transaction_context.get('is_new_retailer', False):
                        decision = rule['action']
                        explanation.append(rule['description'])
                        break

                elif rule['condition'] == 'context.matches_recent_fraud == true':
                    if transaction_context.get('matches_recent_fraud', False):
                        decision = rule['action']
                        explanation.append(rule['description'])
                        break

        # Step 4: If still no decision, use weighted scoring
        if decision is None:
            # Calculate combined score
            combined_score = (
                weighted_rules['classifier_weight'] * classifier_result.get('score', 0.5) +
                weighted_rules['anomaly_weight'] * detector_result.get('anomaly_score', 0)
            )

            # Determine decision based on thresholds
            if combined_score >= weighted_rules['threshold_block']:
                decision = 'BLOCK'
                explanation.append(f"Combined risk score ({combined_score:.2f}) exceeds block threshold ({weighted_rules['threshold_block']})")
            elif combined_score >= weighted_rules['threshold_review']:
                decision = 'REVIEW'
                explanation.append(f"Combined risk score ({combined_score:.2f}) exceeds review threshold ({weighted_rules['threshold_review']})")
            else:
                decision = 'ALLOW'
                explanation.append(f"Combined risk score ({combined_score:.2f}) below review threshold ({weighted_rules['threshold_review']})")

            # Add weights used
            explanation.append(f"Using weights: classifier={weighted_rules['classifier_weight']}, anomaly={weighted_rules['anomaly_weight']}")

        explanation_text = "; ".join(explanation)
        return decision, explanation_text

    def process_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a transaction through the full expert collaboration workflow.

        This method:
        1. Gets opinions from each expert system
        2. Updates the shared context with transaction data
        3. Resolves any conflicts between expert opinions
        4. Makes a final decision about the transaction
        5. Returns comprehensive results with explanations

        Args:
            transaction: Transaction data dictionary

        Returns:
            Dictionary containing processing results and decision
        """
        # Check cache for previously processed transaction
        transaction_id = transaction.get('id', '') or transaction.get('transaction_id', '')
        if transaction_id in self.decision_cache:
            logger.info(f"Using cached decision for transaction {transaction_id}")
            return self.decision_cache[transaction_id]

        # Get expert opinions
        class_report = self.classifier.evaluate(transaction)
        anom_report = self.detector.analyze(transaction)

        # Update shared context
        self.context.update(transaction, class_report, anom_report)

        # Get transaction context
        transaction_context = self.context.check_transaction_context(transaction)

        # Make mediated decision with explanation
        decision, explanation = self.resolve_conflict(
            class_report,
            anom_report,
            transaction_context
        )

        # Log decision
        logger.info(f"Experts collab decision: {decision} - {explanation}")

        # Create comprehensive result
        result = {
            'transaction_id': transaction_id,
            'classifier': class_report,
            'anomaly': anom_report,
            'decision': decision,
            'explanation': explanation,
            'context': self.context.get_snapshot(),
            'timestamp': transaction.get('timestamp', None)
        }

        # Cache the result
        if transaction_id and len(self.decision_cache) < self.max_cache_size:
            self.decision_cache[transaction_id] = result

        # If fraud is detected, add to fraud history
        if decision == 'BLOCK':
            self.context.add_fraud(
                transaction,
                class_report.get('ml_score', 0),
                anom_report.get('anomaly_score', 0),
                class_report.get('rule_violations', 0)
            )

        return result