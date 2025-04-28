#!/usr/bin/env python
"""
Script to create the user_profiles.db database with the required schema.
"""

import os
import sqlite3

def create_user_profiles_db(db_path):
    """Create the user_profiles database with the required schema."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Connect to database (will create it if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_profiles (
        user_id TEXT PRIMARY KEY,
        avg_amount REAL DEFAULT 0,
        max_amount REAL DEFAULT 0,
        avg_daily_transactions REAL DEFAULT 0,
        avg_transaction_interval REAL DEFAULT 0,
        typical_merchants TEXT,  -- JSON array of common merchants
        last_countries TEXT,     -- JSON array of recent countries
        risk_score REAL DEFAULT 0,
        account_age_days INTEGER DEFAULT 0,
        last_transaction_date TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_transactions (
        transaction_id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        amount REAL NOT NULL,
        merchant TEXT,
        merchant_category TEXT,
        country TEXT,
        timestamp TEXT NOT NULL,
        is_fraud INTEGER DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_velocity (
        user_id TEXT NOT NULL,
        time_window TEXT NOT NULL,  -- '1h', '24h', '7d', etc.
        transaction_count INTEGER DEFAULT 0,
        total_amount REAL DEFAULT 0,
        unique_merchants INTEGER DEFAULT 0,
        unique_countries INTEGER DEFAULT 0,
        last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (user_id, time_window),
        FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_patterns (
        pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        pattern_type TEXT NOT NULL,  -- 'time', 'merchant', 'amount', etc.
        pattern_value TEXT NOT NULL, -- JSON representation of the pattern
        confidence REAL DEFAULT 0,
        last_matched TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
    )
    ''')

    # Create indices
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON user_transactions(user_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON user_transactions(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_velocity_user_id ON user_velocity(user_id)')

    # Commit changes and close connection
    conn.commit()
    conn.close()

    print(f"Database created successfully at {db_path}")

if __name__ == "__main__":
    # Path to the database file
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_profiles.db")

    # Create the database
    create_user_profiles_db(db_path)
