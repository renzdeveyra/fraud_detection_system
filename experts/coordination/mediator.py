from typing import Dict
from infrastructure.utils import logger
from infrastructure.memory.context_buffer import ContextBuffer

class ExpertMediator:
    def __init__(self, classifier, detector):
        self.classifier = classifier
        self.detector = detector

        # Create a new context if not available
        from infrastructure.memory import ContextBuffer
        self.context = ContextBuffer()

        # Share context with experts
        self.classifier.context = self.context
        self.detector.context = self.context

    def resolve_conflict(self,
                        classifier_result: Dict,
                        detector_result: Dict) -> str:
        """Apply fusion rules from YAML config"""
        # Rule 1: Either expert has high confidence
        if classifier_result['confidence'] > 0.9 or detector_result['severity'] == 'CRITICAL':
            return 'BLOCK'

        # Rule 2: Both experts agree on risk level
        if (classifier_result['risk'] == 'HIGH' and
            detector_result['severity'] in ['HIGH', 'MEDIUM']):
            return 'REVIEW'

        # Default: Use weighted score
        combined_score = (
            0.7 * classifier_result['score'] +
            0.3 * detector_result['anomaly_score']
        )
        return 'ALLOW' if combined_score < 0.5 else 'REVIEW'

    def process_transaction(self, transaction: Dict) -> Dict:
        """Full collaboration workflow"""
        # Get expert opinions
        class_report = self.classifier.evaluate(transaction)
        anom_report = self.detector.analyze(transaction)

        # Update shared context
        self.context.update(transaction, class_report, anom_report)

        # Make mediated decision
        decision = self.resolve_conflict(class_report, anom_report)

        logger.info(f"Experts collab decision: {decision}")
        return {
            'transaction_id': transaction['id'],
            'classifier': class_report,
            'anomaly': anom_report,
            'decision': decision,
            'context': self.context.get_snapshot()
        }