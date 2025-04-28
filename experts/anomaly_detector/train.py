import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from ...src.utils import save_model, load_config, setup_logger

logger = setup_logger(__name__)

def train_pipeline():
    """End-to-end training pipeline"""
    config = load_config()
    
    # Load data
    logger.info("Loading data...")
    df = pd.read_csv(config['data']['path'])
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['data']['test_size'],
        stratify=y,
        random_state=config['data']['random_state']
    )
    
    # Train supervised model
    logger.info("Training supervised model...")
    clf = LogisticRegression(
        class_weight=config['models']['supervised']['class_weight'],
        max_iter=config['models']['supervised']['max_iter']
    )
    clf.fit(X_train, y_train)
    save_model(clf, "../models/supervised_model.pkl")
    
    # Train anomaly detector (only on normal data)
    logger.info("Training anomaly detector...")
    normal_data = X_train[y_train == 0]
    iso = IsolationForest(
        contamination=config['models']['anomaly']['contamination'],
        n_estimators=config['models']['anomaly']['n_estimators'],
        random_state=config['data']['random_state']
    )
    iso.fit(normal_data)
    save_model(iso, "../models/anomaly_model.pkl")

if __name__ == "__main__":
    train_pipeline()
