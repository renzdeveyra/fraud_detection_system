"""
Rule manager for fraud detection system.
Handles loading, saving, and managing static and dynamic rules.
"""

import os
import json
import sqlite3
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from infrastructure.utils import logger
from infrastructure.config import load_paths, get_project_root


def get_static_rules() -> Dict[str, Any]:
    """
    Load static rules from JSON file.
    
    Returns:
        Dict containing static rules
    """
    paths = load_paths()
    rules_path = os.path.join(
        get_project_root(),
        paths['models']['classifier']['rules']
    )
    
    try:
        with open(rules_path, 'r') as f:
            rules = json.load(f)
        logger.info(f"Loaded static rules from {rules_path}")
        return rules
    except Exception as e:
        logger.error(f"Error loading static rules: {str(e)}")
        return {}


def save_static_rules(rules: Dict[str, Any]) -> bool:
    """
    Save static rules to JSON file.
    
    Args:
        rules: Dictionary of rules to save
        
    Returns:
        bool: True if successful, False otherwise
    """
    paths = load_paths()
    rules_path = os.path.join(
        get_project_root(),
        paths['models']['classifier']['rules']
    )
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(rules_path), exist_ok=True)
        
        # Save rules
        with open(rules_path, 'w') as f:
            json.dump(rules, f, indent=4)
            
        logger.info(f"Saved static rules to {rules_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving static rules: {str(e)}")
        return False


def get_dynamic_rules(active_only: bool = True) -> List[Dict[str, Any]]:
    """
    Load dynamic rules from database.
    
    Args:
        active_only: If True, only return active rules
        
    Returns:
        List of rule dictionaries
    """
    paths = load_paths()
    db_path = os.path.join(
        get_project_root(),
        paths['models']['classifier']['dynamic']
    )
    
    if not os.path.exists(db_path):
        logger.warning(f"Dynamic rules database not found at {db_path}")
        return []
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query rules
        if active_only:
            cursor.execute('SELECT * FROM rules WHERE is_active = 1')
        else:
            cursor.execute('SELECT * FROM rules')
            
        # Get column names
        columns = [description[0] for description in cursor.description]
        
        # Convert to list of dictionaries
        rules = []
        for row in cursor.fetchall():
            rule = {columns[i]: row[i] for i in range(len(columns))}
            rules.append(rule)
            
        conn.close()
        
        logger.info(f"Loaded {len(rules)} dynamic rules from database")
        return rules
    except Exception as e:
        logger.error(f"Error loading dynamic rules: {str(e)}")
        return []


def add_rule(name: str, rule_type: str, feature: str, operator: str, 
             threshold: float, confidence: float = 0.8) -> bool:
    """
    Add a new dynamic rule to the database.
    
    Args:
        name: Rule name
        rule_type: Type of rule (e.g., 'amount', 'velocity')
        feature: Feature the rule applies to
        operator: Comparison operator (e.g., '>', '<', '==')
        threshold: Threshold value
        confidence: Confidence score (0-1)
        
    Returns:
        bool: True if successful, False otherwise
    """
    paths = load_paths()
    db_path = os.path.join(
        get_project_root(),
        paths['models']['classifier']['dynamic']
    )
    
    if not os.path.exists(db_path):
        logger.warning(f"Dynamic rules database not found at {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Insert rule
        cursor.execute('''
        INSERT INTO rules (
            rule_name, rule_type, feature, operator, 
            threshold, confidence, created_at, updated_at, is_active
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            name, rule_type, feature, operator, 
            threshold, confidence, 
            datetime.now().isoformat(), 
            datetime.now().isoformat(),
            1
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added new rule '{name}' to database")
        return True
    except Exception as e:
        logger.error(f"Error adding rule: {str(e)}")
        return False


def update_rule(rule_id: Union[str, int], updates: Dict[str, Any]) -> bool:
    """
    Update an existing dynamic rule.
    
    Args:
        rule_id: ID of the rule to update
        updates: Dictionary of fields to update
        
    Returns:
        bool: True if successful, False otherwise
    """
    paths = load_paths()
    db_path = os.path.join(
        get_project_root(),
        paths['models']['classifier']['dynamic']
    )
    
    if not os.path.exists(db_path):
        logger.warning(f"Dynamic rules database not found at {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Build update query
        set_clauses = []
        params = []
        
        for key, value in updates.items():
            set_clauses.append(f"{key} = ?")
            params.append(value)
            
        # Add updated_at timestamp
        set_clauses.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        
        # Add rule_id
        params.append(rule_id)
        
        # Execute update
        query = f"UPDATE rules SET {', '.join(set_clauses)} WHERE id = ?"
        cursor.execute(query, params)
        
        if cursor.rowcount == 0:
            logger.warning(f"Rule with ID {rule_id} not found")
            conn.close()
            return False
            
        conn.commit()
        conn.close()
        
        logger.info(f"Updated rule {rule_id}")
        return True
    except Exception as e:
        logger.error(f"Error updating rule: {str(e)}")
        return False


def delete_rule(rule_id: Union[str, int]) -> bool:
    """
    Delete a dynamic rule.
    
    Args:
        rule_id: ID of the rule to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    paths = load_paths()
    db_path = os.path.join(
        get_project_root(),
        paths['models']['classifier']['dynamic']
    )
    
    if not os.path.exists(db_path):
        logger.warning(f"Dynamic rules database not found at {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Delete rule
        cursor.execute('DELETE FROM rules WHERE id = ?', (rule_id,))
        
        if cursor.rowcount == 0:
            logger.warning(f"Rule with ID {rule_id} not found")
            conn.close()
            return False
            
        conn.commit()
        conn.close()
        
        logger.info(f"Deleted rule {rule_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting rule: {str(e)}")
        return False


def evaluate_rule(rule: Dict[str, Any], transaction: Dict[str, Any]) -> bool:
    """
    Evaluate a rule against a transaction.
    
    Args:
        rule: Rule dictionary
        transaction: Transaction dictionary
        
    Returns:
        bool: True if rule is violated, False otherwise
    """
    feature = rule['feature']
    operator = rule['operator']
    threshold = rule['threshold']
    
    # Check if feature exists in transaction
    if feature not in transaction:
        logger.warning(f"Feature '{feature}' not found in transaction")
        return False
    
    # Get feature value
    value = transaction[feature]
    
    # Evaluate based on operator
    if operator == '>':
        return value > threshold
    elif operator == '>=':
        return value >= threshold
    elif operator == '<':
        return value < threshold
    elif operator == '<=':
        return value <= threshold
    elif operator == '==':
        return value == threshold
    elif operator == '!=':
        return value != threshold
    else:
        logger.warning(f"Unsupported operator: {operator}")
        return False


def evaluate_transaction(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a transaction against all rules.
    
    Args:
        transaction: Transaction dictionary
        
    Returns:
        Dict containing evaluation results
    """
    # Get static rules
    static_rules = get_static_rules()
    
    # Get dynamic rules
    dynamic_rules = get_dynamic_rules()
    
    # Evaluate static rules
    static_violations = []
    
    # Amount rules
    if 'amount_rules' in static_rules:
        if 'amount' in transaction:
            if transaction['amount'] > static_rules['amount_rules'].get('high_value_threshold', 5000):
                static_violations.append('high_value_threshold')
                
    # Velocity rules
    if 'velocity_rules' in static_rules:
        if 'count_1h' in transaction:
            if transaction['count_1h'] > static_rules['velocity_rules'].get('max_transactions_per_hour', 5):
                static_violations.append('max_hourly_transactions')
                
    # Location rules
    if 'location_rules' in static_rules:
        if 'country' in transaction:
            if transaction['country'] in static_rules['location_rules'].get('suspicious_countries', []):
                static_violations.append('suspicious_country')
    
    # Evaluate dynamic rules
    dynamic_violations = []
    for rule in dynamic_rules:
        if evaluate_rule(rule, transaction):
            dynamic_violations.append({
                'rule_id': rule['id'],
                'rule_name': rule['rule_name'],
                'feature': rule['feature'],
                'threshold': rule['threshold']
            })
    
    return {
        'static_violations': static_violations,
        'dynamic_violations': dynamic_violations,
        'total_violations': len(static_violations) + len(dynamic_violations)
    }


def generate_rules_from_model(model, feature_names, threshold=0.5) -> List[Dict[str, Any]]:
    """
    Generate rules from a trained model.
    
    Args:
        model: Trained classifier model
        feature_names: List of feature names
        threshold: Coefficient magnitude threshold for rule generation
        
    Returns:
        List of generated rules
    """
    if not hasattr(model, 'coef_'):
        logger.warning("Model does not have coefficients, cannot generate rules")
        return []
    
    # Get coefficients
    coefficients = model.coef_[0]
    
    # Generate rules for features with significant coefficients
    rules = []
    for i, coef in enumerate(coefficients):
        if abs(coef) > threshold:
            feature = feature_names[i]
            
            # Determine operator and threshold based on coefficient sign
            if coef > 0:
                # Positive coefficient: higher values increase fraud risk
                operator = '>'
                # Set threshold at 95th percentile (placeholder)
                rule_threshold = 0.95
            else:
                # Negative coefficient: lower values increase fraud risk
                operator = '<'
                # Set threshold at 5th percentile (placeholder)
                rule_threshold = 0.05
                
            # Create rule
            rule = {
                'rule_name': f"model_generated_{feature}",
                'rule_type': 'model_generated',
                'feature': feature,
                'operator': operator,
                'threshold': rule_threshold,
                'confidence': min(1.0, abs(coef) * 2),  # Scale coefficient to confidence
                'is_active': True
            }
            
            rules.append(rule)
    
    return rules
