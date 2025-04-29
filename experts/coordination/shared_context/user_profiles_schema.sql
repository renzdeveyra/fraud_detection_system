-- Schema for user_profiles.db

-- User profiles table
CREATE TABLE user_profiles (
    user_id TEXT PRIMARY KEY,
    avg_amount REAL DEFAULT 0,
    max_amount REAL DEFAULT 0,
    avg_daily_transactions REAL DEFAULT 0,
    avg_transaction_interval REAL DEFAULT 0,
    typical_merchants TEXT,  -- JSON array of common merchants
    last_countries TEXT,     -- JSON array of recent countries
    avg_distance_from_home REAL DEFAULT 0,
    avg_distance_from_last_transaction REAL DEFAULT 0,
    median_purchase_price REAL DEFAULT 0,
    pct_repeat_retailer REAL DEFAULT 0,
    pct_used_chip REAL DEFAULT 0,
    pct_used_pin REAL DEFAULT 0,
    pct_online_orders REAL DEFAULT 0,
    risk_score REAL DEFAULT 0,
    account_age_days INTEGER DEFAULT 0,
    last_transaction_date TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- User transactions history
CREATE TABLE user_transactions (
    transaction_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    amount REAL NOT NULL,
    merchant TEXT,
    merchant_category TEXT,
    country TEXT,
    timestamp TEXT NOT NULL,
    distance_from_home REAL DEFAULT 0,
    distance_from_last_transaction REAL DEFAULT 0,
    ratio_to_median_purchase_price REAL DEFAULT 1.0,
    repeat_retailer INTEGER DEFAULT 0,
    used_chip INTEGER DEFAULT 0,
    used_pin_number INTEGER DEFAULT 0,
    online_order INTEGER DEFAULT 0,
    is_fraud INTEGER DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
);

-- User velocity metrics
CREATE TABLE user_velocity (
    user_id TEXT NOT NULL,
    time_window TEXT NOT NULL,  -- '1h', '24h', '7d', etc.
    transaction_count INTEGER DEFAULT 0,
    total_amount REAL DEFAULT 0,
    unique_merchants INTEGER DEFAULT 0,
    unique_countries INTEGER DEFAULT 0,
    last_updated TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, time_window),
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
);

-- User behavioral patterns
CREATE TABLE user_patterns (
    pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    pattern_type TEXT NOT NULL,  -- 'time', 'merchant', 'amount', etc.
    pattern_value TEXT NOT NULL, -- JSON representation of the pattern
    confidence REAL DEFAULT 0,
    last_matched TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
);

-- Indices for performance
CREATE INDEX idx_transactions_user_id ON user_transactions(user_id);
CREATE INDEX idx_transactions_timestamp ON user_transactions(timestamp);
CREATE INDEX idx_velocity_user_id ON user_velocity(user_id);
