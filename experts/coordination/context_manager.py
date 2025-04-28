"""
Context manager for fraud detection system.
Handles user profiles, transaction history, and shared context.
"""

import os
import json
import sqlite3
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta

from infrastructure.utils import logger
from infrastructure.config import load_paths, get_project_root


def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a user profile from the database.

    Args:
        user_id: User ID

    Returns:
        Dict containing user profile or None if not found
    """
    paths = load_paths()
    db_path = os.path.join(
        get_project_root(),
        paths['shared']['user_profiles']
    )

    if not os.path.exists(db_path):
        logger.warning(f"User profiles database not found at {db_path}")
        return None

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        cursor = conn.cursor()

        # Query user profile
        cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
        row = cursor.fetchone()

        if not row:
            logger.info(f"No profile found for user {user_id}")
            conn.close()
            return None

        # Convert to dictionary
        profile = dict(row)

        # Get velocity metrics
        cursor.execute('SELECT * FROM user_velocity WHERE user_id = ?', (user_id,))
        velocity_rows = cursor.fetchall()

        velocity = {}
        for v_row in velocity_rows:
            velocity[v_row['time_window']] = {
                'transaction_count': v_row['transaction_count'],
                'total_amount': v_row['total_amount'],
                'unique_merchants': v_row['unique_merchants'],
                'unique_countries': v_row['unique_countries']
            }

        profile['velocity'] = velocity

        # Get recent transactions
        cursor.execute('''
        SELECT * FROM user_transactions
        WHERE user_id = ?
        ORDER BY timestamp DESC LIMIT 10
        ''', (user_id,))

        transactions = [dict(t_row) for t_row in cursor.fetchall()]
        profile['recent_transactions'] = transactions

        conn.close()

        # Parse JSON fields
        if profile['typical_merchants']:
            profile['typical_merchants'] = json.loads(profile['typical_merchants'])

        if profile['last_countries']:
            profile['last_countries'] = json.loads(profile['last_countries'])

        logger.info(f"Loaded profile for user {user_id}")
        return profile
    except Exception as e:
        logger.error(f"Error loading user profile: {str(e)}")
        return None


def list_user_profiles(limit: int = 10, sort_by: str = 'risk') -> List[Dict[str, Any]]:
    """
    List user profiles from the database.

    Args:
        limit: Maximum number of profiles to return
        sort_by: Field to sort by ('risk', 'activity', 'recent')

    Returns:
        List of user profile dictionaries
    """
    paths = load_paths()
    db_path = os.path.join(
        get_project_root(),
        paths['shared']['user_profiles']
    )

    if not os.path.exists(db_path):
        logger.warning(f"User profiles database not found at {db_path}")
        return []

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Determine sort order
        if sort_by == 'risk':
            order_by = 'risk_score DESC'
        elif sort_by == 'activity':
            order_by = 'avg_daily_transactions DESC'
        elif sort_by == 'recent':
            order_by = 'last_transaction_date DESC'
        else:
            order_by = 'user_id'

        # Query profiles
        cursor.execute(f'''
        SELECT * FROM user_profiles
        ORDER BY {order_by}
        LIMIT ?
        ''', (limit,))

        profiles = [dict(row) for row in cursor.fetchall()]
        conn.close()

        # Parse JSON fields
        for profile in profiles:
            if profile['typical_merchants']:
                profile['typical_merchants'] = json.loads(profile['typical_merchants'])

            if profile['last_countries']:
                profile['last_countries'] = json.loads(profile['last_countries'])

        logger.info(f"Listed {len(profiles)} user profiles")
        return profiles
    except Exception as e:
        logger.error(f"Error listing user profiles: {str(e)}")
        return []


def update_user_profile(user_id: str, updates: Dict[str, Any]) -> bool:
    """
    Update a user profile in the database.

    Args:
        user_id: User ID
        updates: Dictionary of fields to update

    Returns:
        bool: True if successful, False otherwise
    """
    paths = load_paths()
    db_path = os.path.join(
        get_project_root(),
        paths['shared']['user_profiles']
    )

    if not os.path.exists(db_path):
        logger.warning(f"User profiles database not found at {db_path}")
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if profile exists
        cursor.execute('SELECT 1 FROM user_profiles WHERE user_id = ?', (user_id,))
        exists = cursor.fetchone() is not None

        if not exists:
            # Create new profile
            cursor.execute('''
            INSERT INTO user_profiles (
                user_id, created_at, updated_at
            ) VALUES (?, ?, ?)
            ''', (
                user_id,
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))

        # Build update query
        set_clauses = []
        params = []

        for key, value in updates.items():
            # Handle JSON fields
            if key in ['typical_merchants', 'last_countries'] and value is not None:
                value = json.dumps(value)

            set_clauses.append(f"{key} = ?")
            params.append(value)

        # Add updated_at timestamp
        set_clauses.append("updated_at = ?")
        params.append(datetime.now().isoformat())

        # Add user_id
        params.append(user_id)

        # Execute update
        query = f"UPDATE user_profiles SET {', '.join(set_clauses)} WHERE user_id = ?"
        cursor.execute(query, params)

        conn.commit()
        conn.close()

        logger.info(f"Updated profile for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error updating user profile: {str(e)}")
        return False


def add_transaction(transaction: Dict[str, Any]) -> bool:
    """
    Add a transaction to the database and update user profile.

    Args:
        transaction: Transaction dictionary

    Returns:
        bool: True if successful, False otherwise
    """
    if 'user_id' not in transaction or 'transaction_id' not in transaction:
        logger.error("Transaction missing required fields: user_id, transaction_id")
        return False

    paths = load_paths()
    db_path = os.path.join(
        get_project_root(),
        paths['shared']['user_profiles']
    )

    if not os.path.exists(db_path):
        logger.warning(f"User profiles database not found at {db_path}")
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Insert transaction
        cursor.execute('''
        INSERT INTO user_transactions (
            transaction_id, user_id, amount, merchant,
            merchant_category, country, timestamp, is_fraud
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            transaction['transaction_id'],
            transaction['user_id'],
            transaction.get('amount', 0),
            transaction.get('merchant', ''),
            transaction.get('merchant_category', ''),
            transaction.get('country', ''),
            transaction.get('timestamp', datetime.now().isoformat()),
            transaction.get('is_fraud', 0)
        ))

        # Update user profile
        user_id = transaction['user_id']

        # Get existing profile
        cursor.execute('SELECT * FROM user_profiles WHERE user_id = ?', (user_id,))
        profile = cursor.fetchone()

        if profile:
            # Update existing profile
            avg_amount = profile[1]  # Assuming avg_amount is the second column
            max_amount = profile[2]  # Assuming max_amount is the third column
            transaction_count = profile[3]  # Assuming avg_daily_transactions is the fourth column

            # Update average amount
            new_avg = (avg_amount * transaction_count + transaction.get('amount', 0)) / (transaction_count + 1)

            # Update max amount
            new_max = max(max_amount, transaction.get('amount', 0))

            # Update profile
            cursor.execute('''
            UPDATE user_profiles SET
                avg_amount = ?,
                max_amount = ?,
                avg_daily_transactions = avg_daily_transactions + 0.01,
                last_transaction_date = ?,
                updated_at = ?
            WHERE user_id = ?
            ''', (
                new_avg,
                new_max,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                user_id
            ))
        else:
            # Create new profile
            cursor.execute('''
            INSERT INTO user_profiles (
                user_id, avg_amount, max_amount,
                avg_daily_transactions, last_transaction_date,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                transaction.get('amount', 0),
                transaction.get('amount', 0),
                1,
                datetime.now().isoformat(),
                datetime.now().isoformat(),
                datetime.now().isoformat()
            ))

        # Update velocity metrics
        update_velocity_metrics(cursor, user_id, transaction)

        conn.commit()
        conn.close()

        logger.info(f"Added transaction {transaction['transaction_id']} for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error adding transaction: {str(e)}")
        return False


def update_velocity_metrics(cursor, user_id: str, transaction: Dict[str, Any]) -> None:
    """
    Update velocity metrics for a user.

    Args:
        cursor: Database cursor
        user_id: User ID
        transaction: Transaction dictionary
    """
    # Time windows to track
    windows = ['1h', '24h', '7d']

    # Current timestamp
    now = datetime.now()

    # Transaction timestamp
    tx_time = datetime.fromisoformat(transaction.get('timestamp', now.isoformat()))

    for window in windows:
        # Get window duration
        if window == '1h':
            duration = timedelta(hours=1)
        elif window == '24h':
            duration = timedelta(days=1)
        elif window == '7d':
            duration = timedelta(days=7)
        else:
            continue

        # Calculate window start
        window_start = (tx_time - duration).isoformat()

        # Count transactions in window
        cursor.execute('''
        SELECT COUNT(*) as count, SUM(amount) as total,
               COUNT(DISTINCT merchant) as merchants,
               COUNT(DISTINCT country) as countries
        FROM user_transactions
        WHERE user_id = ? AND timestamp > ?
        ''', (user_id, window_start))

        result = cursor.fetchone()

        if result:
            count = result[0] + 1  # Include current transaction
            total = result[1] + transaction.get('amount', 0)
            merchants = result[2]
            countries = result[3]

            # Check if merchant is new
            if transaction.get('merchant'):
                cursor.execute('''
                SELECT 1 FROM user_transactions
                WHERE user_id = ? AND merchant = ? AND timestamp > ?
                LIMIT 1
                ''', (user_id, transaction.get('merchant'), window_start))

                if not cursor.fetchone():
                    merchants += 1

            # Check if country is new
            if transaction.get('country'):
                cursor.execute('''
                SELECT 1 FROM user_transactions
                WHERE user_id = ? AND country = ? AND timestamp > ?
                LIMIT 1
                ''', (user_id, transaction.get('country'), window_start))

                if not cursor.fetchone():
                    countries += 1

            # Update or insert velocity record
            cursor.execute('''
            INSERT OR REPLACE INTO user_velocity (
                user_id, time_window, transaction_count,
                total_amount, unique_merchants, unique_countries, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id, window, count, total, merchants, countries, now.isoformat()
            ))


def get_recent_frauds() -> Dict[str, Any]:
    """
    Get recent fraud patterns from the database.

    Returns:
        Dict containing recent frauds and patterns
    """
    paths = load_paths()
    file_path = os.path.join(
        get_project_root(),
        paths['shared']['fraud_history']
    )

    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded recent frauds from {file_path}")
        return data
    except Exception as e:
        logger.warning(f"Error loading recent frauds: {str(e)}")
        return {
            "recent_frauds": [],
            "fraud_patterns": [],
            "last_updated": datetime.now().isoformat()
        }


def add_fraud(transaction: Dict[str, Any], classifier_score: float,
              anomaly_score: float, rule_violations: List[str]) -> bool:
    """
    Add a fraud transaction to the recent frauds list.

    Args:
        transaction: Transaction dictionary
        classifier_score: Fraud classifier score
        anomaly_score: Anomaly detector score
        rule_violations: List of violated rules

    Returns:
        bool: True if successful, False otherwise
    """
    paths = load_paths()
    file_path = os.path.join(
        get_project_root(),
        paths['shared']['fraud_history']
    )

    try:
        # Load existing data
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except:
            data = {
                "recent_frauds": [],
                "fraud_patterns": [],
                "last_updated": datetime.now().isoformat()
            }

        # Create fraud entry
        fraud_entry = {
            "transaction_id": transaction.get('transaction_id', ''),
            "timestamp": transaction.get('timestamp', datetime.now().isoformat()),
            "features": {
                "amount": transaction.get('amount', 0),
                "merchant_category": transaction.get('merchant_category', ''),
                "country": transaction.get('country', ''),
                "hour_of_day": transaction.get('hour_of_day', 0),
                "velocity": transaction.get('velocity', 0)
            },
            "classifier_score": classifier_score,
            "anomaly_score": anomaly_score,
            "rule_violations": rule_violations
        }

        # Add to recent frauds
        data["recent_frauds"].insert(0, fraud_entry)

        # Limit to 100 recent frauds
        data["recent_frauds"] = data["recent_frauds"][:100]

        # Update timestamp
        data["last_updated"] = datetime.now().isoformat()

        # Save data
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Added fraud transaction {transaction.get('transaction_id', '')} to recent frauds")
        return True
    except Exception as e:
        logger.error(f"Error adding fraud: {str(e)}")
        return False


def check_transaction_context(transaction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check transaction against user profile and context.

    Args:
        transaction: Transaction dictionary

    Returns:
        Dict containing context check results
    """
    if 'user_id' not in transaction:
        logger.warning("Transaction missing user_id")
        return {
            'user_known': False,
            'amount_ratio_to_average': 1.0,
            'amount_ratio_to_max': 1.0,
            'user_typical_country': True,
            'transaction_velocity': 0,
            'matches_recent_fraud': False
        }

    # Get user profile
    profile = get_user_profile(transaction['user_id'])

    # Default context
    context = {
        'user_known': False,
        'amount_ratio_to_average': 1.0,
        'amount_ratio_to_max': 1.0,
        'user_typical_country': True,
        'transaction_velocity': 0,
        'matches_recent_fraud': False
    }

    if profile:
        context['user_known'] = True

        # Check amount
        if 'amount' in transaction and profile['avg_amount'] > 0:
            context['amount_ratio_to_average'] = transaction['amount'] / profile['avg_amount']

        if 'amount' in transaction and profile['max_amount'] > 0:
            context['amount_ratio_to_max'] = transaction['amount'] / profile['max_amount']

        # Check country
        if 'country' in transaction and profile['last_countries']:
            context['user_typical_country'] = transaction['country'] in profile['last_countries']

        # Check velocity
        if 'velocity' in profile and '1h' in profile['velocity']:
            context['transaction_velocity'] = profile['velocity']['1h']['transaction_count']

    # Check against recent frauds
    recent_frauds = get_recent_frauds()

    for fraud in recent_frauds.get('recent_frauds', []):
        similarity = 0
        total_checks = 0

        # Check amount
        if 'amount' in transaction and 'amount' in fraud['features']:
            total_checks += 1
            if abs(transaction['amount'] - fraud['features']['amount']) < 100:
                similarity += 1

        # Check merchant category
        if 'merchant_category' in transaction and 'merchant_category' in fraud['features']:
            total_checks += 1
            if transaction['merchant_category'] == fraud['features']['merchant_category']:
                similarity += 1

        # Check country
        if 'country' in transaction and 'country' in fraud['features']:
            total_checks += 1
            if transaction['country'] == fraud['features']['country']:
                similarity += 1

        # Check hour of day
        if 'hour_of_day' in transaction and 'hour_of_day' in fraud['features']:
            total_checks += 1
            if abs(transaction['hour_of_day'] - fraud['features']['hour_of_day']) <= 1:
                similarity += 1

        # Calculate similarity score
        if total_checks > 0:
            similarity_score = similarity / total_checks

            if similarity_score > 0.7:
                context['matches_recent_fraud'] = True
                break

    return context
