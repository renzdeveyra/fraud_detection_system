# experts/fraud_classifier/train.py

import pandas as pd
import yaml
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Import specific utilities provided
from infrastructure.utils import (
    load_processed_data,
    split_data,
    save_model,
    save_model_metrics,
    logger,  # Use the pre-configured logger instance
    log_execution_time
)

# --- Configuration ---
# Assume config files are still loaded directly here, as utils use them internally
CONFIG_DIR = os.path.join(os.path.dirname(__file__), '../../infrastructure/config')
PATHS_CONFIG_PATH = os.path.join(CONFIG_DIR, 'paths.yaml')
PARAMS_CONFIG_PATH = os.path.join(CONFIG_DIR, 'model_params.yaml')

# Load configurations
try:
    with open(PATHS_CONFIG_PATH, 'r') as f:
        paths_config = yaml.safe_load(f)
    with open(PARAMS_CONFIG_PATH, 'r') as f:
        params_config = yaml.safe_load(f)
except FileNotFoundError as e:
    logger.error(f"Configuration file not found: {e}")
    raise
except yaml.YAMLError as e:
    logger.error(f"Error parsing YAML configuration: {e}")
    raise

# --- Constants ---
# Path definitions might be less critical here if utils handle them,
# but useful for reference or if needed elsewhere.
PROCESSED_DATA_RELATIVE_PATH = paths_config['data']['processed'] # Path relative to project root
CLASSIFIER_MODEL_NAME = 'classifier' # Name used in model_ops

TARGET_COLUMN = params_config.get('data', {}).get('target_column', 'Class')
CLASSIFIER_PARAMS = params_config.get('fraud_classifier', {}).get('logistic_regression', {})
TEST_SIZE = params_config.get('training', {}).get('test_split_ratio', 0.2)
RANDOM_STATE = params_config.get('training', {}).get('random_state', 42)

# --- Main Training Logic ---

@log_execution_time # Apply the execution time logging decorator
def run_training():
    """
    Loads data, trains the supervised fraud classifier (Logistic Regression),
    evaluates it, saves the trained model and metrics using infrastructure utilities.
    """
    logger.info("Starting Fraud Classifier training pipeline...")

    # 1. Load Data using the utility
    logger.info(f"Loading processed data ('{PROCESSED_DATA_RELATIVE_PATH}') via utility...")
    try:
        # load_processed_data internally resolves the full path using project root and paths.yaml
        df = load_processed_data()
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Processed data file not found at expected location derived from '{PROCESSED_DATA_RELATIVE_PATH}'. Ensure it exists or run processing first.")
        raise
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    # Check target column exists before splitting
    if TARGET_COLUMN not in df.columns:
        logger.error(f"Target column '{TARGET_COLUMN}' not found in the loaded dataset.")
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    # Check for class imbalance (informational) - before splitting
    class_distribution = df[TARGET_COLUMN].value_counts(normalize=True)
    logger.info(f"Target class distribution in loaded data:\n{class_distribution}")
    if class_distribution.min() < 0.05: # Example threshold
         logger.warning("Significant class imbalance detected. Logistic Regression uses class_weight='balanced'.")


    # 2. Split Data using the utility
    logger.info(f"Splitting data using utility (Test size: {TEST_SIZE}, Target: '{TARGET_COLUMN}')...")
    try:
        # The split_data utility handles X, y separation and splitting
        X_train, X_test, y_train, y_test = split_data(
            df=df,
            target_col=TARGET_COLUMN,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        logger.info(f"Data split complete. Training features: {X_train.shape}, Test features: {X_test.shape}")
    except Exception as e:
        logger.error(f"Failed to split data: {e}")
        raise

    # 3. Train Model
    logger.info("Training Logistic Regression model...")
    # Ensure class_weight='balanced' is used as specified in the overview
    model_params = CLASSIFIER_PARAMS.copy() # Avoid modifying the original dict
    model_params['class_weight'] = 'balanced'
    model_params['random_state'] = RANDOM_STATE # Ensure reproducibility
    # Add solver if not specified, as default 'lbfgs' might warn with large datasets/penalty
    if 'solver' not in model_params:
        model_params['solver'] = 'liblinear' # A reasonable default

    try:
        model = LogisticRegression(**model_params)
        model.fit(X_train, y_train)
        logger.info("Model training completed.")
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise

    # 4. Evaluate Model
    logger.info("Evaluating model performance on the test set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability for the positive class

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    logger.info("--- Classification Report ---")
    for line in report.split('\n'): # Log line by line for better readability
        logger.info(line)
    logger.info("--- Confusion Matrix ---")
    logger.info(f"\n{cm}")
    logger.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    # 5. Save Metrics using the utility
    logger.info("Saving evaluation metrics...")
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(), # Convert numpy array to list for JSON serialization
        'classification_report': report,
        'model_params': model_params,
        'data_shape': {'train': X_train.shape, 'test': X_test.shape},
        'target_distribution_test': y_test.value_counts().to_dict()
    }
    try:
        # save_model_metrics handles path resolution and saving to JSON
        save_model_metrics(metrics=metrics_dict, model_name=CLASSIFIER_MODEL_NAME)
        logger.info(f"Metrics saved for model '{CLASSIFIER_MODEL_NAME}'.")
    except Exception as e:
        logger.error(f"Failed to save metrics: {e}")
        # Continue to save the model even if metrics saving fails

    # 6. Save Model using the utility
    logger.info(f"Saving trained model '{CLASSIFIER_MODEL_NAME}' using utility...")
    try:
        # save_model handles path resolution, directory creation, and saving via pickle
        saved_path = save_model(model=model, model_name=CLASSIFIER_MODEL_NAME)
        logger.info(f"Model saved successfully to relative path: {saved_path}")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        raise # Fail the pipeline if model saving fails

    logger.info("Fraud Classifier training pipeline finished successfully.")

if __name__ == "__main__":
    logger.info("Starting script execution...")
    try:
        run_training()
        logger.info("Script finished successfully.")
    except Exception as e:
        # The decorator @log_execution_time already logs the error,
        # but we add a critical log here for script exit.
        logger.critical(f"Training pipeline failed: {e}", exc_info=False) # Set exc_info=False to avoid duplicate traceback from decorator
        exit(1)

