#!/usr/bin/env python
"""
Training module for the Fraud Classifier expert system.
Implements supervised learning for known fraud pattern detection.

Focuses on core dataset features:
- distance_from_home
- distance_from_last_transaction
- ratio_to_median_purchase_price
- repeat_retailer
- used_chip
- used_pin_number
- online_order
- fraud (target)
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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
        all_params = load_params()
        self.params = params or all_params.get('classifier', {})

        # Define core dataset features
        self.core_features = [
            'distance_from_home',
            'distance_from_last_transaction',
            'ratio_to_median_purchase_price',
            'repeat_retailer',
            'used_chip',
            'used_pin_number',
            'online_order'
        ]

        # Define feature types
        self.numeric_features = [
            'distance_from_home',
            'distance_from_last_transaction',
            'ratio_to_median_purchase_price'
        ]

        self.binary_features = [
            'repeat_retailer',
            'used_chip',
            'used_pin_number',
            'online_order'
        ]

        logger.info(f"Initialized fraud classifier trainer with core features: {self.core_features}")
        logger.info(f"Parameters: {self.params}")

        # Define preprocessing steps
        self.scaler = StandardScaler()

        # Setup preprocessing pipeline
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('bin', 'passthrough', self.binary_features)
            ],
            remainder='drop'  # Drop any columns not specified
        )

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

        if model_type.lower() in ['logistic_regression', 'logisticregression']:
            # Get parameters from config
            params_dict = self.params.get('params', {})

            self.model = LogisticRegression(
                C=params_dict.get('C', 1.0),
                penalty=params_dict.get('penalty', 'l2'),
                class_weight=params_dict.get('class_weight', 'balanced'),
                random_state=params_dict.get('random_state', 42),
                max_iter=params_dict.get('max_iter', 1000),
                solver=params_dict.get('solver', 'liblinear'),
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_execution_time
    def preprocess(self, X, y=None):
        """
        Preprocess data for model training or inference.

        Focuses on core dataset features and applies appropriate transformations:
        - Log transformation for distance features
        - Scaling for numeric features
        - Pass-through for binary features

        Args:
            X (DataFrame): Features
            y (Series, optional): Target variable, required for training

        Returns:
            X_processed: Processed features
            y_resampled: Resampled targets (if y is provided)
        """
        # Make a copy to avoid modifying original data
        X_copy = X.copy()

        # Ensure all core features exist, fill missing ones with defaults
        for feature in self.core_features:
            if feature not in X_copy.columns:
                if feature in self.binary_features:
                    X_copy[feature] = 0  # Default for binary features
                else:
                    X_copy[feature] = 0.0  # Default for numeric features
                logger.warning(f"Feature {feature} not found in input data, using default value")

        # Create log-transformed distance features
        if 'distance_from_home' in X_copy.columns:
            X_copy['log_distance_from_home'] = np.log1p(X_copy['distance_from_home'])

        if 'distance_from_last_transaction' in X_copy.columns:
            X_copy['log_distance_from_last_transaction'] = np.log1p(X_copy['distance_from_last_transaction'])

        # Add log-transformed features to numeric features list
        numeric_features_with_log = self.numeric_features + ['log_distance_from_home', 'log_distance_from_last_transaction']

        # Update preprocessor with log-transformed features
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features_with_log),
                ('bin', 'passthrough', self.binary_features)
            ],
            remainder='drop'  # Drop any columns not specified
        )

        # Apply preprocessing
        X_processed_array = self.preprocessor.fit_transform(X_copy)

        # Create DataFrame with appropriate column names
        processed_columns = (
            numeric_features_with_log +  # Scaled numeric features
            self.binary_features  # Passthrough binary features
        )
        X_processed = pd.DataFrame(X_processed_array, columns=processed_columns)

        # Store feature names for the model
        self.feature_names = processed_columns

        # Apply SMOTE for class imbalance if we have labels
        if y is not None:
            X_resampled, y_resampled = self.smote.fit_resample(X_processed, y)
            logger.info(f"Applied SMOTE: {sum(y)} fraud → {sum(y_resampled)} fraud samples")
            return X_resampled, y_resampled

        return X_processed

    @log_execution_time
    def train(self, X, y, optimize=False):
        """
        Train the fraud classifier on core dataset features.

        Args:
            X (DataFrame): Features
            y (Series): Target variable (1 for fraud, 0 for legitimate)
            optimize (bool): Whether to perform hyperparameter optimization

        Returns:
            model: Trained classifier model
        """
        logger.info(f"Training fraud classifier on {len(X)} samples ({sum(y)} fraud instances)")
        logger.info(f"Using core features: {self.core_features}")

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

        # Store feature names in the model for later use
        self.model.feature_names_in_ = np.array(self.feature_names)

        # Add version information
        self.model.version = "fraud_classifier_v2"

        # Extract feature importance
        self._analyze_feature_importance(X_train.columns)

        # Log model information
        logger.info(f"Model trained successfully with {len(self.model.feature_names_in_)} features")
        logger.info(f"Model version: {self.model.version}")

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

    def suggest_rules(self, threshold=0.3):
        """
        Generate rule suggestions based on model coefficients.
        Customized for core dataset features with specific thresholds.

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

        # Define default thresholds for core features
        default_thresholds = {
            'distance_from_home': 100,
            'log_distance_from_home': 4.6,  # log(100)
            'distance_from_last_transaction': 50,
            'log_distance_from_last_transaction': 3.9,  # log(50)
            'ratio_to_median_purchase_price': 3.0,
            'repeat_retailer': 0,  # Binary feature
            'used_chip': 0,  # Binary feature
            'used_pin_number': 0,  # Binary feature
            'online_order': 1  # Binary feature
        }

        for _, row in significant.iterrows():
            feature = row['Feature']
            coef = row['Coefficient']

            # Check if this is a binary feature
            is_binary = feature in self.binary_features

            # Positive coefficients indicate fraud signals
            if coef > 0:
                if is_binary:
                    # For binary features with positive coefficients
                    if feature == 'online_order':
                        # online_order=1 indicates fraud
                        suggested_value = 1
                    else:
                        # For other binary features, 0 typically indicates fraud
                        suggested_value = 0

                    rule_suggestions[feature] = {
                        'direction': 'equals',
                        'coefficient': float(coef),
                        'suggested_value': suggested_value,
                        'rule_type': 'binary'
                    }
                else:
                    # For numeric features with positive coefficients
                    # Higher values indicate fraud
                    rule_suggestions[feature] = {
                        'direction': 'high',
                        'coefficient': float(coef),
                        'suggested_threshold': default_thresholds.get(feature, 'upper_percentile_95'),
                        'rule_type': 'numeric'
                    }
            else:
                if is_binary:
                    # For binary features with negative coefficients
                    if feature == 'online_order':
                        # online_order=0 indicates legitimate
                        suggested_value = 0
                    else:
                        # For other binary features, 1 typically indicates legitimate
                        suggested_value = 1

                    rule_suggestions[feature] = {
                        'direction': 'equals',
                        'coefficient': float(coef),
                        'suggested_value': suggested_value,
                        'rule_type': 'binary'
                    }
                else:
                    # For numeric features with negative coefficients
                    # Lower values indicate fraud
                    rule_suggestions[feature] = {
                        'direction': 'low',
                        'coefficient': float(coef),
                        'suggested_threshold': default_thresholds.get(feature, 'lower_percentile_5'),
                        'rule_type': 'numeric'
                    }

        # Add specific rule combinations
        if 'used_chip' in rule_suggestions and 'used_pin_number' in rule_suggestions:
            rule_suggestions['payment_method_combination'] = {
                'description': 'Both chip and PIN not used',
                'features': ['used_chip', 'used_pin_number'],
                'condition': 'both_equal_zero',
                'rule_type': 'combination'
            }

        # Log rule suggestions
        logger.info(f"Generated {len(rule_suggestions)} rule suggestions based on model coefficients")

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