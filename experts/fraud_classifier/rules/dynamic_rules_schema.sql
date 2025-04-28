-- Schema for dynamic_rules.db

-- Rules table to store dynamically generated rules
CREATE TABLE rules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_name TEXT NOT NULL,
    rule_type TEXT NOT NULL,
    feature TEXT NOT NULL,
    operator TEXT NOT NULL,
    threshold REAL NOT NULL,
    confidence REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT 1
);

-- Rule performance tracking
CREATE TABLE rule_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rule_id INTEGER NOT NULL,
    true_positives INTEGER DEFAULT 0,
    false_positives INTEGER DEFAULT 0,
    true_negatives INTEGER DEFAULT 0,
    false_negatives INTEGER DEFAULT 0,
    precision REAL,
    recall REAL,
    f1_score REAL,
    last_evaluated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (rule_id) REFERENCES rules(id)
);

-- Feature statistics for threshold calculation
CREATE TABLE feature_statistics (
    feature TEXT PRIMARY KEY,
    min_value REAL,
    max_value REAL,
    mean REAL,
    median REAL,
    std_dev REAL,
    p5 REAL,
    p25 REAL,
    p75 REAL,
    p95 REAL,
    p99 REAL,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Rule generation history
CREATE TABLE rule_generation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_version TEXT,
    rules_added INTEGER,
    rules_removed INTEGER,
    rules_modified INTEGER,
    performance_change REAL
);