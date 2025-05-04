"""
Expert Mediator for fraud detection system.
Coordinates decisions between multiple expert systems.
"""

import json
import os
import yaml
from typing import Dict, List, Any, Optional, Tuple, Union

from infrastructure.config import load_paths, get_project_root
from infrastructure.utils import (
    logger, TransactionCache, handle_errors,
    ProcessingError, ConfigurationError
)
from infrastructure.memory.context_buffer import ContextBuffer

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
        self.context = ContextBuffer()

        # Share context with experts
        self.classifier.context = self.context
        self.detector.context = self.context

        # Initialize transaction cache
        self.transaction_cache = TransactionCache(max_size=1000)

        logger.info("Initialized expert mediator with fusion rules")

    @handle_errors(error_type=ConfigurationError, fallback_value=None, log_level="warning")
    def _load_fusion_rules_from_file(self) -> Optional[Dict[str, Any]]:
        """
        Load fusion rules from YAML configuration file.

        Returns:
            Dictionary containing fusion rules or None if file not found/invalid
        """
        paths = load_paths()
        rules_path = os.path.join(
            get_project_root(),
            paths.get('shared', {}).get('fusion_rules', 'experts/coordination/fusion_rules.yaml')
        )

        if not os.path.exists(rules_path):
            logger.warning(f"Fusion rules file not found at {rules_path}")
            return None

        with open(rules_path, 'r') as f:
            rules = yaml.safe_load(f)

        logger.info(f"Loaded fusion rules from {rules_path}")
        return rules

    def _get_default_fusion_rules(self) -> Dict[str, Any]:
        """
        Get default fusion rules when no configuration file is available.

        Returns:
            Dictionary containing default fusion rules
        """
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

    def _load_fusion_rules(self) -> Dict[str, Any]:
        """
        Load fusion rules from file or use defaults if not available.

        Returns:
            Dictionary containing fusion rules
        """
        rules = self._load_fusion_rules_from_file()

        if rules is None:
            logger.warning("Using default fusion rules")
            return self._get_default_fusion_rules()

        return rules

    def _get_weighted_rules(self, transaction_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine which weighted rules to use based on transaction context.

        Args:
            transaction_context: Context information about the transaction

        Returns:
            Dictionary of weighted rules to apply
        """
        # Determine which weighted rules to use based on core dataset features
        weighted_rule_set = 'default'
        if transaction_context.get('distance_from_home_unusual', False) or \
           transaction_context.get('distance_from_last_transaction_unusual', False):
            weighted_rule_set = 'unusual_distance'
        elif transaction_context.get('payment_method_unusual', False):
            weighted_rule_set = 'unusual_payment'

        # Get weights and thresholds
        return self.fusion_rules.get('weighted_rules', {}).get(weighted_rule_set, {
            'classifier_weight': 0.7,
            'anomaly_weight': 0.3,
            'threshold_review': 0.5,
            'threshold_block': 0.8
        })

    def _apply_priority_rules(self,
                             classifier_result: Dict[str, Any],
                             detector_result: Dict[str, Any],
                             transaction_context: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
        """
        Apply priority rules that can override other decisions.

        Args:
            classifier_result: Results from fraud classifier expert
            detector_result: Results from anomaly detector expert
            transaction_context: Context information about the transaction

        Returns:
            Tuple of (decision, explanation_list) or (None, []) if no rule matched
        """
        explanation = []

        for rule in self.fusion_rules.get('priority_rules', []):
            # Simple rule evaluation (in a real system, this would use a proper rule engine)
            if rule['condition'] == 'classifier.confidence > 0.9':
                if classifier_result.get('confidence', 0) > 0.9:
                    explanation.append(rule['description'])
                    explanation.append(f"Classifier confidence: {classifier_result.get('confidence', 0):.2f}")
                    return rule['action'], explanation

            elif rule['condition'] == 'anomaly.severity == "CRITICAL"':
                if detector_result.get('severity') == 'CRITICAL':
                    explanation.append(rule['description'])
                    return rule['action'], explanation

            elif rule['condition'] == 'context.matches_recent_fraud == true':
                if transaction_context.get('matches_recent_fraud', False):
                    explanation.append(rule['description'])
                    return rule['action'], explanation

        return None, []

    def _apply_agreement_rules(self,
                              classifier_result: Dict[str, Any],
                              detector_result: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
        """
        Apply rules based on agreement between experts.

        Args:
            classifier_result: Results from fraud classifier expert
            detector_result: Results from anomaly detector expert

        Returns:
            Tuple of (decision, explanation_list) or (None, []) if no rule matched
        """
        explanation = []

        for rule in self.fusion_rules.get('agreement_rules', []):
            if rule['condition'] == 'classifier.risk == "HIGH" and anomaly.severity in ["HIGH", "CRITICAL"]':
                if (classifier_result.get('risk') == 'HIGH' and
                    detector_result.get('severity') in ['HIGH', 'CRITICAL']):
                    explanation.append(rule['description'])
                    return rule['action'], explanation

            elif rule['condition'] == 'classifier.risk == "MEDIUM" and anomaly.severity == "MEDIUM"':
                if (classifier_result.get('risk') == 'MEDIUM' and
                    detector_result.get('severity') == 'MEDIUM'):
                    explanation.append(rule['description'])
                    return rule['action'], explanation

            elif rule['condition'] == 'classifier.risk == "LOW" and anomaly.severity in ["LOW", "NORMAL"]':
                if (classifier_result.get('risk') == 'LOW' and
                    detector_result.get('severity') in ['LOW', 'NORMAL']):
                    explanation.append(rule['description'])
                    return rule['action'], explanation

            elif rule['condition'] == 'classifier.risk == "HIGH" and anomaly.severity == "NORMAL" or classifier.risk == "LOW" and anomaly.severity == "CRITICAL"':
                if ((classifier_result.get('risk') == 'HIGH' and detector_result.get('severity') == 'NORMAL') or
                    (classifier_result.get('risk') == 'LOW' and detector_result.get('severity') == 'CRITICAL')):
                    explanation.append(rule['description'])
                    return rule['action'], explanation

        return None, []

    def _apply_context_rules(self, transaction_context: Dict[str, Any]) -> Tuple[Optional[str], List[str]]:
        """
        Apply rules based on transaction context.

        Args:
            transaction_context: Context information about the transaction

        Returns:
            Tuple of (decision, explanation_list) or (None, []) if no rule matched
        """
        if not transaction_context:
            return None, []

        explanation = []

        for rule in self.fusion_rules.get('context_rules', []):
            if rule['condition'] == 'context.distance_from_home_unusual == true':
                if transaction_context.get('distance_from_home_unusual', False):
                    explanation.append(rule['description'])
                    return rule['action'], explanation

            elif rule['condition'] == 'context.distance_from_last_transaction_unusual == true':
                if transaction_context.get('distance_from_last_transaction_unusual', False):
                    explanation.append(rule['description'])
                    return rule['action'], explanation

            elif rule['condition'] == 'context.purchase_price_ratio_unusual == true':
                if transaction_context.get('purchase_price_ratio_unusual', False):
                    explanation.append(rule['description'])
                    return rule['action'], explanation

            elif rule['condition'] == 'context.payment_method_unusual == true':
                if transaction_context.get('payment_method_unusual', False):
                    explanation.append(rule['description'])
                    return rule['action'], explanation

            elif rule['condition'] == 'context.is_new_retailer == true':
                if transaction_context.get('is_new_retailer', False):
                    explanation.append(rule['description'])
                    return rule['action'], explanation

            elif rule['condition'] == 'context.matches_recent_fraud == true':
                if transaction_context.get('matches_recent_fraud', False):
                    explanation.append(rule['description'])
                    return rule['action'], explanation

        return None, []

    def _apply_weighted_scoring(self,
                               classifier_result: Dict[str, Any],
                               detector_result: Dict[str, Any],
                               weighted_rules: Dict[str, Any]) -> Tuple[str, List[str]]:
        """
        Apply weighted scoring to make a decision.

        Args:
            classifier_result: Results from fraud classifier expert
            detector_result: Results from anomaly detector expert
            weighted_rules: Weighted rules to apply

        Returns:
            Tuple of (decision, explanation_list)
        """
        explanation = []

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

        return decision, explanation

    @handle_errors(error_type=ProcessingError, log_level="error")
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

        all_explanations = []

        # Get weighted rules based on context
        weighted_rules = self._get_weighted_rules(transaction_context)

        # Step 1: Apply priority rules first
        decision, explanations = self._apply_priority_rules(
            classifier_result, detector_result, transaction_context
        )
        all_explanations.extend(explanations)

        # Step 2: If no priority rule matched, check agreement rules
        if decision is None:
            decision, explanations = self._apply_agreement_rules(
                classifier_result, detector_result
            )
            all_explanations.extend(explanations)

        # Step 3: If still no decision, check context rules
        if decision is None:
            decision, explanations = self._apply_context_rules(transaction_context)
            all_explanations.extend(explanations)

        # Step 4: If still no decision, use weighted scoring
        if decision is None:
            decision, explanations = self._apply_weighted_scoring(
                classifier_result, detector_result, weighted_rules
            )
            all_explanations.extend(explanations)

        explanation_text = "; ".join(all_explanations)
        return decision, explanation_text

    def _get_expert_opinions(self, transaction: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get opinions from each expert system.

        Args:
            transaction: Transaction data dictionary

        Returns:
            Tuple of (classifier_report, anomaly_report)
        """
        class_report = self.classifier.evaluate(transaction)
        anom_report = self.detector.analyze(transaction)
        return class_report, anom_report

    def _create_result_dict(self,
                           transaction: Dict[str, Any],
                           class_report: Dict[str, Any],
                           anom_report: Dict[str, Any],
                           decision: str,
                           explanation: str) -> Dict[str, Any]:
        """
        Create a comprehensive result dictionary.

        Args:
            transaction: Transaction data dictionary
            class_report: Results from fraud classifier expert
            anom_report: Results from anomaly detector expert
            decision: Final decision (BLOCK, REVIEW, ALLOW)
            explanation: Explanation of the decision

        Returns:
            Dictionary containing processing results and decision
        """
        transaction_id = transaction.get('id', '') or transaction.get('transaction_id', '')

        return {
            'transaction_id': transaction_id,
            'classifier': class_report,
            'anomaly': anom_report,
            'decision': decision,
            'explanation': explanation,
            'context': self.context.get_snapshot(),
            'timestamp': transaction.get('timestamp', None)
        }

    def _handle_fraud_detection(self,
                               transaction: Dict[str, Any],
                               class_report: Dict[str, Any],
                               anom_report: Dict[str, Any],
                               decision: str) -> None:
        """
        Handle actions when fraud is detected.

        Args:
            transaction: Transaction data dictionary
            class_report: Results from fraud classifier expert
            anom_report: Results from anomaly detector expert
            decision: Final decision (BLOCK, REVIEW, ALLOW)
        """
        if decision == 'BLOCK':
            self.context.add_fraud(
                transaction,
                class_report.get('ml_score', 0),
                anom_report.get('anomaly_score', 0),
                class_report.get('rule_violations', 0)
            )

    @handle_errors(error_type=ProcessingError, log_level="error")
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
        cache_key = self.transaction_cache.generate_key(transaction)
        cached_result = self.transaction_cache.get(cache_key)

        if cached_result:
            logger.info(f"Using cached decision for transaction {transaction.get('id', 'unknown')}")
            return cached_result

        # Get expert opinions
        class_report, anom_report = self._get_expert_opinions(transaction)

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
        result = self._create_result_dict(
            transaction, class_report, anom_report, decision, explanation
        )

        # Cache the result
        self.transaction_cache.set(cache_key, result)

        # If fraud is detected, add to fraud history
        self._handle_fraud_detection(transaction, class_report, anom_report, decision)

        return result