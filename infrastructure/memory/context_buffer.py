import json
import sqlite3
import numpy as np
from collections import deque
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import os

from infrastructure.config import load_paths, load_params, get_project_root
from infrastructure.utils.logger import logger


class ContextBuffer:
    """Shared memory between expert systems to store transaction context"""
    
    def __init__(self, max_size: Optional[int] = None):
        params = load_params()
        paths = load_paths()
        
        # Initialize buffer size
        self.max_size = max_size or params['memory']['context_buffer_size']
        self.recent_transactions = deque(maxlen=self.max_size)
        
        # Initialize user profiles database
        user_profiles_path = os.path.join(get_project_root(), paths['shared']['user_profiles'])
        os.makedirs(os.path.dirname(user_profiles_path), exist_ok=True)
        self.user_db = sqlite3.connect(user_profiles_path)
        self._init_user_profiles_db()
        
        # Initialize fraud patterns storage
        fraud_patterns_path = os.path.join(get_project_root(), paths['shared']['fraud_history'])
        os.makedirs(os.path.dirname(fraud_patterns_path), exist_ok=True)
        
        if os.path.exists(fraud_patterns_path):
            with open(fraud_patterns_path, 'r') as f:
                self.fraud_patterns = json.load(f)
        else:
            self.fraud_patterns = {
                'recent_frauds': [],
                'clusters': []
            }
            self._save_fraud_patterns()
            
        # Current clustering centroids
        self.centroids = np.array([])
        
        logger.info("Context buffer initialized")
    
    def _init_user_profiles_db(self) -> None:
        """Initialize the user profiles database schema"""
        cursor = self.user_db.cursor()
        
        # Create profiles table if not exists
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            avg_amount REAL,
            max_amount REAL,
            typical_merchants TEXT,
            last_countries TEXT,
            updated_at TIMESTAMP
        )
        ''')
        
        # Create transactions history table if not exists
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_transactions (
            transaction_id TEXT PRIMARY KEY,
            user_id TEXT,
            amount REAL,
            merchant TEXT,
            country TEXT,
            timestamp TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
        )
        ''')
        
        self.user_db.commit()
    
    def update(self, transaction: Dict[str, Any], 
              classifier_result: Dict[str, Any], 
              anomaly_result: Dict[str, Any]) -> None:
        """Update context with new transaction data and expert results"""
        # Add to recent transactions
        context_entry = {
            'transaction': transaction,
            'classifier_result': classifier_result,
            'anomaly_result': anomaly_result,
            'timestamp': datetime.now().isoformat()
        }
        self.recent_transactions.append(context_entry)
        
        # Update user profile
        if 'user_id' in transaction:
            self._update_user_profile(transaction)
        
        # Update fraud patterns if flagged as fraud
        if (classifier_result.get('risk') == 'HIGH' or 
            anomaly_result.get('severity') == 'CRITICAL'):
            self._update_fraud_patterns(transaction, classifier_result, anomaly_result)
    
    def _update_user_profile(self, transaction: Dict[str, Any]) -> None:
        """Update user profile in the database"""
        user_id = transaction.get('user_id')
        if not user_id:
            return
            
        cursor = self.user_db.cursor()
        
        # Add transaction to history
        cursor.execute('''
        INSERT OR REPLACE INTO user_transactions 
        (transaction_id, user_id, amount, merchant, country, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            transaction.get('id', ''),
            user_id,
            transaction.get('amount', 0),
            transaction.get('merchant', ''),
            transaction.get('country', ''),
            datetime.now().isoformat()
        ))
        
        # Get user's transaction history
        cursor.execute('''
        SELECT amount, merchant, country FROM user_transactions
        WHERE user_id = ? AND timestamp > ?
        ''', (
            user_id, 
            (datetime.now() - timedelta(days=30)).isoformat()
        ))
        
        transactions = cursor.fetchall()
        
        if transactions:
            # Calculate statistics
            amounts = [t[0] for t in transactions if t[0] is not None]
            merchants = [t[1] for t in transactions if t[1] is not None]
            countries = [t[2] for t in transactions if t[2] is not None]
            
            avg_amount = sum(amounts) / len(amounts) if amounts else 0
            max_amount = max(amounts) if amounts else 0
            
            # Find most common merchants
            merchant_counts = {}
            for m in merchants:
                merchant_counts[m] = merchant_counts.get(m, 0) + 1
            typical_merchants = json.dumps(
                [m for m, c in sorted(merchant_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
            )
            
            # Get recent countries
            recent_countries = json.dumps(list(set(countries))[:5])
            
            # Update user profile
            cursor.execute('''
            INSERT OR REPLACE INTO user_profiles 
            (user_id, avg_amount, max_amount, typical_merchants, last_countries, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                user_id, 
                avg_amount,
                max_amount,
                typical_merchants,
                recent_countries,
                datetime.now().isoformat()
            ))
        
        self.user_db.commit()
    
    def _update_fraud_patterns(self, transaction: Dict[str, Any],
                              classifier_result: Dict[str, Any],
                              anomaly_result: Dict[str, Any]) -> None:
        """Update recorded fraud patterns"""
        fraud_entry = {
            'transaction_features': {k: v for k, v in transaction.items() 
                                     if k not in ['id', 'user_id']},
            'classifier_insights': classifier_result.get('rule_violations', []),
            'anomaly_insights': {
                'score': anomaly_result.get('raw_score', 0),
                'cluster_deviation': anomaly_result.get('cluster_deviation', 0)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to recent frauds list
        self.fraud_patterns['recent_frauds'].append(fraud_entry)
        
        # Keep only last 100 fraud entries
        self.fraud_patterns['recent_frauds'] = self.fraud_patterns['recent_frauds'][-100:]
        
        # Save updated patterns
        self._save_fraud_patterns()
    
    def _save_fraud_patterns(self) -> None:
        """Save fraud patterns to disk"""
        paths = load_paths()
        fraud_patterns_path = os.path.join(get_project_root(), paths['shared']['fraud_history'])
        
        with open(fraud_patterns_path, 'w') as f:
            json.dump(self.fraud_patterns, f, indent=2)
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile information"""
        cursor = self.user_db.cursor()
        
        cursor.execute('''
        SELECT avg_amount, max_amount, typical_merchants, last_countries
        FROM user_profiles WHERE user_id = ?
        ''', (user_id,))
        
        profile = cursor.fetchone()
        
        if not profile:
            return {
                'user_id': user_id,
                'profile_exists': False
            }
        
        return {
            'user_id': user_id,
            'profile_exists': True,
            'avg_amount': profile[0],
            'max_amount': profile[1],
            'typical_merchants': json.loads(profile[2]) if profile[2] else [],
            'last_countries': json.loads(profile[3]) if profile[3] else []
        }
    
    def get_recent_frauds(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent fraud entries"""
        return self.fraud_patterns['recent_frauds'][-limit:]
    
    def get_cluster_distance(self, transaction: Dict[str, Any]) -> float:
        """Calculate distance from transaction to known fraud clusters"""
        if len(self.centroids) == 0:
            return 0.0
        
        # Extract numeric features
        features = np.array([v for k, v in transaction.items() 
                            if isinstance(v, (int, float)) and k not in ['id', 'user_id']])
        
        if len(features) == 0:
            return 0.0
            
        # Calculate minimum distance to any centroid
        distances = np.linalg.norm(self.centroids - features, axis=1)
        return float(np.min(distances)) if len(distances) > 0 else 0.0
    
    def update_clusters(self, centroids: np.ndarray) -> None:
        """Update fraud cluster centroids"""
        self.centroids = centroids
        
        # Store in fraud patterns
        self.fraud_patterns['clusters'] = centroids.tolist() if len(centroids) > 0 else []
        self._save_fraud_patterns()
    
    def get_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of current context for decision-making"""
        return {
            'recent_transactions_count': len(self.recent_transactions),
            'fraud_patterns_count': len(self.fraud_patterns['recent_frauds']),
            'cluster_count': len(self.centroids)
        }
    
    def __del__(self):
        """Clean up database connection"""
        if hasattr(self, 'user_db'):
            self.user_db.close()