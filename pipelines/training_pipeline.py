#!/usr/bin/env python
"""
Training pipeline for credit card fraud detection system.
Trains both expert systems (Fraud Classifier and Anomaly Detector).
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from infrastructure.config import load_paths, load_params, get_project_root
from infrastructure.utils import (
    logger, log_execution_time, load_raw_data, process_raw_data,
    split_data, save_model, save_model_metrics, load_rules, save_rules
)
from experts.fraud_classifier.train import FraudClassifierTrainer
from experts.anomaly_detector.train import AnomalyDetectorTrainer

@log_execution_time
def train_classifier(data_file=None, test_size=0.2, save=True):
    """Train the fraud classifier expert system"""
    logger.info("Starting fraud classifier training")

    # Load and process data
    if data_file:
        df = pd.read_csv(data_file)
        logger.info(f"Loaded custom data from {data_file}")
    else:
        df = process_raw_data(save=True)
        logger.info("Loaded and processed default data")

    # Split data
    # Use 'fraud' as the target column instead of 'Class'
    X_train, X_test, y_train, y_test = split_data(df, target_col='fraud', test_size=test_size)

    # Initialize trainer
    trainer = FraudClassifierTrainer()

    # Train model
    classifier = trainer.train(X_train, y_train)

    # Evaluate model
    # We need to preprocess X_test the same way as X_train
    X_test_processed = trainer.preprocess(X_test)
    y_pred = classifier.predict(X_test_processed)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    logger.info(f"Classifier metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Save model and metrics
    if save:
        save_model(classifier, 'classifier')
        save_model_metrics(metrics, 'fraud_classifier')

    return classifier, metrics

@log_execution_time
def train_anomaly_detector(data_file=None, contamination=0.01, save=True):
    """Train the anomaly detector expert system"""
    logger.info("Starting anomaly detector training")

    # Load and process data
    if data_file:
        df = pd.read_csv(data_file)
        logger.info(f"Loaded custom data from {data_file}")
    else:
        df = process_raw_data(save=True)
        logger.info("Loaded and processed default data")

    # Get only legitimate transactions for training
    # Use 'fraud' as the target column instead of 'Class'
    legitimate_data = df[df['fraud'] == 0]
    fraud_data = df[df['fraud'] == 1]

    # Initialize trainer
    trainer = AnomalyDetectorTrainer(contamination=contamination)

    # Train model
    detector = trainer.train(legitimate_data)

    # Evaluate model
    # Use both legitimate and fraud data for evaluation
    X_eval = pd.concat([legitimate_data, fraud_data])
    y_true = np.concatenate([
        np.zeros(len(legitimate_data)),
        np.ones(len(fraud_data))
    ])

    # Preprocess evaluation data
    X_eval_processed = trainer.preprocess(X_eval)

    # Predict anomalies
    y_pred = trainer.predict(detector, X_eval_processed)

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    metrics = {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }

    logger.info(f"Anomaly detector metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Save model and metrics
    if save:
        save_model(detector, 'anomaly')
        save_model_metrics(metrics, 'anomaly_detector')

    return detector, metrics

@log_execution_time
def update_rules():
    """Update static and dynamic rules based on model insights"""
    logger.info("Updating rules based on model insights")

    # Load current static rules
    current_rules = load_rules('static')

    # Load thresholds
    thresholds = load_rules('thresholds')

    # Save updated static rules
    save_rules(current_rules, 'static')
    save_rules(thresholds, 'thresholds')

    # Generate and add dynamic rules from the trained model
    try:
        # Import necessary modules
        import pickle
        import os
        from infrastructure.config import get_project_root, load_paths
        from experts.fraud_classifier.rules.rule_manager import generate_rules_from_model, add_rule
        from infrastructure.utils.model_ops import load_model
        from experts.fraud_classifier.train import FraudClassifierTrainer

        # Load the trained classifier model
        classifier = load_model('classifier')
        logger.info("Loaded classifier model for rule generation")

        # Initialize trainer to get feature names
        trainer = FraudClassifierTrainer()

        # Get feature names from the model if available
        if hasattr(classifier, 'feature_names_in_'):
            feature_names = classifier.feature_names_in_
        else:
            # Use default core features with transformations
            feature_names = (
                trainer.numeric_features +
                ['log_distance_from_home', 'log_distance_from_last_transaction'] +
                trainer.binary_features
            )
            logger.warning("Model doesn't have feature_names_in_ attribute, using default features")

        # Generate rules from the model
        generated_rules = generate_rules_from_model(classifier, feature_names, threshold=0.3)
        logger.info(f"Generated {len(generated_rules)} dynamic rules from model")

        # Add each rule to the database
        rules_added = 0
        for rule in generated_rules:
            success = add_rule(
                name=rule['rule_name'],
                rule_type=rule['rule_type'],
                feature=rule['feature'],
                operator=rule['operator'],
                threshold=rule['threshold'],
                confidence=rule['confidence']
            )
            if success:
                rules_added += 1

        # Log results
        logger.info(f"Successfully added {rules_added} dynamic rules to the database")

        # Also generate rules for core features based on domain knowledge
        core_features = {
            'distance_from_home': {'operator': '>', 'threshold': 100.0, 'type': 'distance'},
            'distance_from_last_transaction': {'operator': '>', 'threshold': 50.0, 'type': 'distance'},
            'ratio_to_median_purchase_price': {'operator': '>', 'threshold': 3.0, 'type': 'transaction_pattern'},
            'repeat_retailer': {'operator': '==', 'threshold': 0.0, 'type': 'transaction_pattern'},
            'used_chip': {'operator': '==', 'threshold': 0.0, 'type': 'payment_method'},
            'used_pin_number': {'operator': '==', 'threshold': 0.0, 'type': 'payment_method'},
            'online_order': {'operator': '==', 'threshold': 1.0, 'type': 'payment_method'}
        }

        # Add core feature rules
        core_rules_added = 0
        for feature, rule_info in core_features.items():
            rule_name = f"core_{feature}"
            success = add_rule(
                name=rule_name,
                rule_type=rule_info['type'],
                feature=feature,
                operator=rule_info['operator'],
                threshold=rule_info['threshold'],
                confidence=0.9  # High confidence for core rules
            )
            if success:
                core_rules_added += 1

        logger.info(f"Added {core_rules_added} core feature rules to the database")

    except Exception as e:
        logger.error(f"Error generating dynamic rules: {str(e)}")

    logger.info("Rules update completed successfully")

def main():
    """Main training pipeline entry point"""
    parser = argparse.ArgumentParser(description='Train fraud detection expert systems')
    parser.add_argument('--data', type=str, help='Path to data file (CSV)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test split size')
    parser.add_argument('--contamination', type=float, default=0.01, help='Anomaly contamination rate')
    parser.add_argument('--skip-classifier', action='store_true', help='Skip classifier training')
    parser.add_argument('--skip-detector', action='store_true', help='Skip anomaly detector training')
    parser.add_argument('--update-rules', action='store_true', help='Update rules after training')

    args = parser.parse_args()

    # Create a shared data file path if provided
    data_file = args.data

    # Train classifier if not skipped
    if not args.skip_classifier:
        classifier, classifier_metrics = train_classifier(data_file, args.test_size)
        logger.info("Fraud classifier training completed")

    # Train anomaly detector if not skipped
    if not args.skip_detector:
        detector, detector_metrics = train_anomaly_detector(data_file, args.contamination)
        logger.info("Anomaly detector training completed")

    # Update rules if requested
    if args.update_rules:
        update_rules()

    logger.info("Training pipeline completed successfully")

if __name__ == "__main__":
    main()