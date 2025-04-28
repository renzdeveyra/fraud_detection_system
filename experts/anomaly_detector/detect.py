import numpy as np
from infrastructure.config import load_params
from typing import Dict

class AnomalyDetectorExpert:
    def __init__(self, model, context):
        self.model = model
        self.context = context
        self.params = load_params()['anomaly']
        
    def calculate_severity(self, score: float) -> str:
        """Convert raw score to risk categories"""
        if score < self.params['critical_threshold']:
            return 'CRITICAL'
        elif score < self.params['high_threshold']:
            return 'HIGH'
        else:
            return 'NORMAL'
    
    def analyze(self, transaction: Dict) -> Dict:
        """Full anomaly analysis"""
        raw_score = self.model.decision_function([transaction])[0]
        severity = self.calculate_severity(raw_score)
        
        # Compare to cluster centroids
        cluster_distance = self.context.get_cluster_distance(transaction)
        
        return {
            'raw_score': raw_score,
            'severity': severity,
            'cluster_deviation': cluster_distance,
            'anomaly_score': np.clip(raw_score + 0.5 * cluster_distance, -1, 1)
        }
