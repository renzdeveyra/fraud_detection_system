#!/usr/bin/env python
"""
Training module for the Anomaly Detector expert system.
Implements unsupervised learning for novel fraud pattern detection.
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from infrastructure.utils import logger, log_execution_time
from infrastructure.config import load_params


class AnomalyDetectorTrainer:
    """Trainer for the unsupervised anomaly detection model"""

    def __init__(self, contamination=0.01, params=None):
        """
        Initialize the anomaly detector trainer.

        Args:
            contamination (float): Expected proportion of anomalies in training data
            params (dict, optional): Model hyperparameters. If None, loads from config.
        """
        self.params = params or load_params('anomaly_detector')
        self.contamination = contamination or self.params.get('contamination', 0.01)

        logger.info(f"Initialized anomaly detector trainer with contamination: {self.contamination}")

        # Define preprocessing components
        self.scaler = RobustScaler()  # Less sensitive to outliers than StandardScaler

        # For dimensionality reduction (optional)
        self.pca = None
        if self.params.get('use_pca', False):
            self.pca = PCA(n_components=self.params.get('pca_components', 10))

        # Initialize the model
        self._init_model()

    def _init_model(self):
        """Initialize the anomaly detection model based on configuration"""
        model_type = self.params.get('model_type', 'isolation_forest')

        if model_type == 'isolation_forest':
            self.model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.params.get('n_estimators', 100),
                max_samples=self.params.get('max_samples', 'auto'),
                max_features=self.params.get('max_features', 1.0),
                random_state=self.params.get('random_state', 42),
                n_jobs=-1
            )
        elif model_type == 'dbscan':
            self.model = DBSCAN(
                eps=self.params.get('eps', 0.5),
                min_samples=self.params.get('min_samples', 5),
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    @log_execution_time
    def preprocess(self, X):
        """
        Preprocess data for model training or inference.

        Args:
            X (DataFrame): Features

        Returns:
            X_processed: Processed features
        """
        # Make a copy to avoid modifying original data
        X_copy = X.copy()

        # Drop target column if present
        if 'Class' in X_copy.columns:
            X_copy = X_copy.drop('Class', axis=1)

        # Extract features to use - exclude non-predictive columns
        exclude_cols = self.params.get('exclude_cols', [])
        feature_cols = [col for col in X_copy.columns if col not in exclude_cols]
        features = X_copy[feature_cols]

        # Scale numeric features
        X_scaled = self.scaler.fit_transform(features)
        X_processed = pd.DataFrame(X_scaled, columns=feature_cols)

        # Apply dimensionality reduction if configured
        if self.pca is not None:
            X_pca = self.pca.fit_transform(X_scaled)
            pca_cols = [f'PC{i + 1}' for i in range(X_pca.shape[1])]
            X_processed = pd.DataFrame(X_pca, columns=pca_cols)

            # Log explained variance
            explained_var = self.pca.explained_variance_ratio_.sum()
            logger.info(f"PCA with {len(pca_cols)} components explains {explained_var:.2%} of variance")

        return X_processed

    @log_execution_time
    def train(self, X):
        """
        Train the anomaly detector.

        Args:
            X (DataFrame): Training data (should contain only legitimate transactions)

        Returns:
            model: Trained anomaly detection model
        """
        model_type = self.params.get('model_type', 'isolation_forest')
        logger.info(f"Training {model_type} anomaly detector on {len(X)} samples")

        # Preprocess data
        X_train = self.preprocess(X)

        # Train model
        self.model.fit(X_train)

        # For Isolation Forest, learn decision function thresholds
        if model_type == 'isolation_forest':
            self._learn_thresholds(X_train)

        return self.model

    def _learn_thresholds(self, X_train):
        """
        Learn decision function thresholds for different severity levels.

        Args:
            X_train (DataFrame): Preprocessed training data
        """
        # Get anomaly scores (decision function)
        scores = self.model.decision_function(X_train)

        # Calculate thresholds for different severity levels
        self.thresholds = {
            'LOW': np.percentile(scores, 10),
            'MEDIUM': np.percentile(scores, 5),
            'HIGH': np.percentile(scores, 1),
            'CRITICAL': np.percentile(scores, 0.1)
        }

        # Log thresholds
        logger.info(f"Learned anomaly thresholds: {self.thresholds}")

        # Save thresholds to file
        thresholds_file = os.path.join(
            self.params.get('model_dir', 'models'),
            'anomaly_thresholds.json'
        )

        os.makedirs(os.path.dirname(thresholds_file), exist_ok=True)
        with open(thresholds_file, 'w') as f:
            json.dump({k: float(v) for k, v in self.thresholds.items()}, f, indent=2)

        logger.info(f"Saved thresholds to {thresholds_file}")

    def predict(self, model, X):
        """
        Predict anomalies using the trained model.

        Args:
            model: Trained anomaly detection model
            X (DataFrame): Data to evaluate

        Returns:
            predictions: Binary predictions (1 for anomaly/fraud, 0 for normal)
        """
        # Preprocess data
        X_processed = self.preprocess(X)

        # Get predictions (-1 for anomalies, 1 for normal)
        if hasattr(model, 'predict'):
            raw_predictions = model.predict(X_processed)
            # Convert to 0/1 (0 for normal, 1 for anomaly/fraud)
            return (raw_predictions == -1).astype(int)
        else:
            # For clustering algorithms like DBSCAN
            # Points with cluster label -1 are considered anomalies
            clusters = model.fit_predict(X_processed)
            return (clusters == -1).astype(int)

    def decision_function(self, model, X):
        """
        Get anomaly scores (decision function values).

        Args:
            model: Trained anomaly detection model
            X (DataFrame): Data to evaluate

        Returns:
            scores: Anomaly scores (lower values indicate anomalies)
        """
        # Preprocess data
        X_processed = self.preprocess(X)

        # Get scores
        if hasattr(model, 'decision_function'):
            return model.decision_function(X_processed)
        else:
            logger.warning("Model doesn't support decision_function. Using binary predictions.")
            return self.predict(model, X)

    def get_severity(self, score):
        """
        Convert anomaly score to severity level.

        Args:
            score (float): Anomaly score from decision_function

        Returns:
            severity (str): Severity level (LOW, MEDIUM, HIGH, CRITICAL)
        """
        if not hasattr(self, 'thresholds'):
            logger.warning("No thresholds available. Using default logic.")
            # Default logic
            if score < -0.2:
                return "CRITICAL"
            elif score < -0.1:
                return "HIGH"
            elif score < 0:
                return "MEDIUM"
            else:
                return "LOW"

        # Use learned thresholds
        if score <= self.thresholds['CRITICAL']:
            return "CRITICAL"
        elif score <= self.thresholds['HIGH']:
            return "HIGH"
        elif score <= self.thresholds['MEDIUM']:
            return "MEDIUM"
        elif score <= self.thresholds['LOW']:
            return "LOW"
        else:
            return "NORMAL"

    @log_execution_time
    def visualize_anomalies(self, X, y=None, save_path=None):
        """
        Visualize anomalies in 2D space.

        Args:
            X (DataFrame): Data to visualize
            y (Series, optional): True labels (1 for fraud, 0 for legitimate)
            save_path (str, optional): Path to save the visualization

        Returns:
            fig: Matplotlib figure
        """
        # Preprocess data
        X_processed = self.preprocess(X)

        # Reduce to 2D for visualization
        if X_processed.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X_processed)
        else:
            X_2d = X_processed.values

        # Get anomaly scores
        if hasattr(self.model, 'decision_function'):
            scores = self.model.decision_function(X_processed)
        else:
            # Use predictions for models without decision_function
            pred = self.model.fit_predict(X_processed)
            scores = np.where(pred == -1, -1, 1)

        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            'x': X_2d[:, 0],
            'y': X_2d[:, 1],
            'score': scores
        })

        # Add true labels if available
        if y is not None:
            plot_df['true_label'] = y.values

        # Create plot
        plt.figure(figsize=(12, 8))

        if y is not None:
            # Plot with true labels
            sns.scatterplot(
                data=plot_df, x='x', y='y', hue='true_label',
                palette={0: 'blue', 1: 'red'},
                style=plot_df['score'] < 0,  # Mark predicted anomalies
                s=100, alpha=0.7
            )
            plt.title('Anomaly Detection Results with True Labels')
            plt.legend(['Normal', 'Fraud', 'Predicted Anomaly'])
        else:
            # Plot with only anomaly scores
            sns.scatterplot(
                data=plot_df, x='x', y='y', hue='score',
                palette='viridis', s=100, alpha=0.7
            )
            plt.title('Anomaly Detection Results')
            plt.colorbar(label='Anomaly Score')

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')

        # Save plot if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Saved anomaly visualization to {save_path}")

        return plt.gcf()


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

    # Train on legitimate transactions only
    X_legitimate = X_df[y_series == 0]

    # Train anomaly detector
    detector = AnomalyDetectorTrainer(contamination=0.05)
    model = detector.train(X_legitimate)

    # Get predictions
    y_pred = detector.predict(model, X_df)

    # Calculate metrics
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_series, y_pred, average='binary'
    )

    print(f"Test metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Visualize results
    detector.visualize_anomalies(X_df, y_series, save_path="anomaly_visualization.png")