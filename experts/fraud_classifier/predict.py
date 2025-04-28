import json
from sklearn.base import BaseEstimator
from typing import Dict

class FraudClassifierExpert:
    def __init__(self, model: BaseEstimator, rules_path: str):
        self.model = model
        with open(rules_path) as f:
            self.rules = json.load(f)
            
    def apply_rules(self, transaction: Dict) -> int:
        """Apply domain-specific fraud rules"""
        violations = 0
        # Rule 1: Large amount deviation from user history
        if (transaction['amount'] > 
            3 * self.context.get_user_avg(transaction['user_id'])):
            violations +=1
            
        # Rule 2: High-velocity transactions
        if transaction['count_1h'] > self.rules['max_hourly_transactions']:
            violations +=1
            
        return violations
    
    def evaluate(self, transaction: Dict) -> Dict:
        """Combined ML + rules evaluation"""
        ml_score = self.model.predict_proba([transaction])[0][1]
        rule_violations = self.apply_rules(transaction)
        
        return {
            'ml_score': ml_score,
            'rule_violations': rule_violations,
            'confidence': min(1.0, ml_score + 0.2 * rule_violations)
        } 