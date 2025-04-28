#!/usr/bin/env python
"""
Training module for the Fraud Classifier expert system.
Implements supervised learning for known fraud pattern detection.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

from infrastructure.utils import logger, log_execution_time
from infrastructure.config import load_params


class FraudClassifierTrainer:
    """Trainer for the supervised fraud classification model"""

    def __init__(self, params=None):
        """
        Initialize the fraud classifier trainer.

        Args:
            params (dict, optional): Model hyperparameters. If None, loads from config.
        """
        self.params = params or load_params('fraud_classifier')
        logger.info(f"Initialized fraud classifier trainer with params: {self.params}")

        # Define preprocessing steps
        self.scaler = StandardScaler()

        # Setup SMOTE for handling class imbalance
        self.smote = SMOTE(
            random_state=self.params.get('random_state', 42),
            sampling_strategy=self.params.get('sampling_strategy', 0.1)
        )

        # Initialize the model
        self._init_model()

    def _init_model(self):
        """Initialize the classifier model based on configuration"""
        model_type = self.params.get('model_type', 'logistic_regression')

        if model_type == 'logistic_regression':
            self.model = LogisticRegression(
                C=self.params.get('C', 1.0),
                penalty=self.params.get('penalty', 'l2'),
                class_weight=self.params.get('class_weight', 'balanced'),
                random_state=self.params.get('random_state', 42),
                max_iter=self.params.get('max_iter', 1000),
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_execution_time
    def preprocess(self, X, y=None):
        """
        Preprocess data for model training or inference.

        Args:
            X (DataFrame): Features
            y (Series, optional): Target variable, required for training

        Returns:
            X_processed: Processed features
            y_resampled: Resampled targets (if y is provided)
        """
        # Make a copy to avoid modifying original data
        X_copy = X.copy()

        # Extract features to use - exclude non-predictive columns
        feature_cols = self.params.get('feature_cols', X_copy.columns)
        features = X_copy[feature_cols]

        # Scale numeric features
        X_scaled = self.scaler.fit_transform(features)
        X_processed = pd.DataFrame(X_scaled, columns=feature_cols)

        # Apply SMOTE for class imbalance if we have labels
        if y is not None:
            X_resampled, y_resampled = self.smote.fit_resample(X_processed, y)
            logger.info(f"Applied SMOTE: {sum(y)} fraud → {sum(y_resampled)} fraud samples")
            return X_resampled, y_resampled

        return X_processed

    @log_execution_time
    def train(self, X, y, optimize=False):
        """
        Train the fraud classifier.

        Args:
            X (DataFrame): Features
            y (Series): Target variable (1 for fraud, 0 for legitimate)
            optimize (bool): Whether to perform hyperparameter optimization

        Returns:
            model: Trained classifier model
        """
        logger.info(f"Training fraud classifier on {len(X)} samples ({sum(y)} fraud instances)")

        # Preprocess data
        X_train, y_train = self.preprocess(X, y)

        # Optimize hyperparameters if requested
        if optimize:
            logger.info("Starting hyperparameter optimization")
            param_grid = self.params.get('param_grid', {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'class_weight': ['balanced', None]
            })

            grid_search = GridSearchCV(
                self.model, param_grid, cv=5, scoring='f1', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

            self.model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
        else:
            # Train model with current parameters
            self.model.fit(X_train, y_train)

        # Extract feature importance
        self._analyze_feature_importance(X_train.columns)

        return self.model

    def _analyze_feature_importance(self, feature_names):
        """
        Analyze and log feature importance from the trained model.
        Used for rule generation and model explainability.

        Args:
            feature_names: List of feature column names
        """
        # For logistic regression, coefficient magnitude indicates importance
        if hasattr(self.model, 'coef_'):
            coefficients = self.model.coef_[0]
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients,
                'Abs_Value': np.abs(coefficients)
            })

            # Sort by absolute importance
            importance_df = importance_df.sort_values('Abs_Value', ascending=False)

            # Log top N most important features
            top_n = min(10, len(importance_df))
            logger.info(f"Top {top_n} important features:")
            for i, row in importance_df.head(top_n).iterrows():
                logger.info(f"  {row['Feature']}: {row['Coefficient']:.4f}")

            # Store feature importance for rule generation
            self.feature_importance = importance_df

    def predict(self, X):
        """
        Make predictions using the trained model.

        Args:
            X (DataFrame): Features to predict

        Returns:
            predictions: Binary predictions (1 for fraud, 0 for legitimate)
        """
        X_processed = self.preprocess(X)
        return self.model.predict(X_processed)

    def predict_proba(self, X):
        """
        Get fraud probability scores.

        Args:
            X (DataFrame): Features to predict

        Returns:
            probabilities: Probability of fraud for each transaction
        """
        X_processed = self.preprocess(X)
        proba = self.model.predict_proba(X_processed)
        return proba[:, 1]  # Return probability of class 1 (fraud)

    def suggest_rules(self, threshold=0.5):
        """
        Generate rule suggestions based on model coefficients.

        Args:
            threshold (float): Coefficient magnitude threshold for suggesting rules

        Returns:
            rules (dict): Dictionary of suggested rules based on model insights
        """
        if not hasattr(self, 'feature_importance'):
            logger.warning("No feature importance available. Train model first.")
            return {}

        # Filter features with significant coefficients
        significant = self.feature_importance[
            self.feature_importance['Abs_Value'] > threshold
            ]

        rule_suggestions = {}
        for _, row in significant.iterrows():
            feature = row['Feature']
            coef = row['Coefficient']

            # Positive coefficients indicate fraud signals
            if coef > 0:
                rule_suggestions[feature] = {
                    'direction': 'high',
                    'coefficient': float(coef),
                    'suggested_threshold': 'upper_percentile_95'
                }
            else:
                rule_suggestions[feature] = {
                    'direction': 'low',
                    'coefficient': float(coef),
                    'suggested_threshold': 'lower_percentile_5'
                }

        return rule_suggestions


if __name__ == "__main__":
    # Simple test code
    from sklearn.datasets import make_classification

    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=10,
        n_redundant=5, n_clusters_per_class=2, weights=[0.95, 0.05],
        random_state=42
    )
    X_df = pd.DataFrame(X, columns=[f'V{i}' for i in range(20)])
    y_series = pd.Series(y)

    # Train classifier
    trainer = FraudClassifierTrainer()
    model = trainer.train(X_df, y_series)

    # Get predictions
    y_pred = trainer.predict(X_df)
    y_proba = trainer.predict_proba(X_df)

    # Print accuracy
    accuracy = (y_pred == y_series).mean()
    print(f"Test accuracy: {accuracy:.4f}")

    # Generate rule suggestions
    rules = trainer.suggest_rules(threshold=0.4)
    print(f"Generated {len(rules)} rule suggestions")