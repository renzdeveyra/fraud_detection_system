#!/usr/bin/env python
"""
Main entry point for the Fraud Detection System.

This script provides a unified command-line interface to access
all functionality of the fraud detection system, including:
- Training the expert systems
- Running inference on transactions
- Evaluating model performance
- Managing system configuration
- Setting up the system infrastructure
- Managing fraud detection rules
- Adjusting anomaly detection thresholds
- Viewing and managing user profiles

Usage:
    python main.py train [options]
    python main.py infer [options]
    python main.py evaluate [options]
    python main.py config [options]
    python main.py setup [options]
    python main.py rules [options]
    python main.py thresholds [options]
    python main.py profiles [options]
"""

import sys
import argparse
import json
from infrastructure.utils import logger

# Import pipeline modules
from pipelines.training_pipeline import train_classifier, train_anomaly_detector, update_rules
from pipelines.inference_pipeline import (
    load_experts, process_single_transaction,
    process_from_file, process_from_stream
)

# Import new modules
from experts.anomaly_detector.thresholds.dynamic_adjustments import ThresholdAdjuster
import experts.fraud_classifier.rules.rule_manager as rule_manager
import experts.coordination.context_manager as context_manager


def setup_train_parser(subparsers):
    """Setup the parser for the train command"""
    train_parser = subparsers.add_parser(
        'train',
        help='Train the fraud detection expert systems'
    )

    train_parser.add_argument('--data', type=str, help='Path to data file (CSV)')
    train_parser.add_argument('--test-size', type=float, default=0.2, help='Test split size')
    train_parser.add_argument('--contamination', type=float, default=0.01, help='Anomaly contamination rate')
    train_parser.add_argument('--skip-classifier', action='store_true', help='Skip classifier training')
    train_parser.add_argument('--skip-detector', action='store_true', help='Skip anomaly detector training')
    train_parser.add_argument('--update-rules', action='store_true', help='Update rules after training')

    return train_parser


def setup_infer_parser(subparsers):
    """Setup the parser for the infer command"""
    infer_parser = subparsers.add_parser(
        'infer',
        help='Run inference on transactions'
    )

    infer_parser.add_argument('--input', type=str, help='Input file with transactions')
    infer_parser.add_argument('--output', type=str, help='Output file for results')
    infer_parser.add_argument('--stream', type=str, help='Stream source for real-time processing')
    infer_parser.add_argument('--transaction', type=str, help='Single JSON transaction to process')

    return infer_parser


def setup_evaluate_parser(subparsers):
    """Setup the parser for the evaluate command"""
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate model performance'
    )

    eval_parser.add_argument('--data', type=str, help='Path to evaluation data file (CSV)')
    eval_parser.add_argument('--model', type=str, choices=['classifier', 'anomaly', 'both'],
                            default='both', help='Model to evaluate')

    return eval_parser


def setup_config_parser(subparsers):
    """Setup the parser for the config command"""
    config_parser = subparsers.add_parser(
        'config',
        help='Manage system configuration'
    )

    config_parser.add_argument('--show', action='store_true', help='Show current configuration')
    config_parser.add_argument('--update', type=str, help='Update configuration parameter (key=value)')

    return config_parser


def setup_setup_parser(subparsers):
    """Setup the parser for the setup command"""
    setup_parser = subparsers.add_parser(
        'setup',
        help='Initialize the fraud detection system'
    )

    setup_parser.add_argument('--force', action='store_true',
                             help='Force recreation of databases')
    setup_parser.add_argument('--skip-db', action='store_true',
                             help='Skip database creation')

    return setup_parser


def setup_rules_parser(subparsers):
    """Setup the parser for the rules command"""
    rules_parser = subparsers.add_parser(
        'rules',
        help='Manage fraud detection rules'
    )

    rules_subparsers = rules_parser.add_subparsers(dest='rules_command')

    # List rules
    list_parser = rules_subparsers.add_parser('list', help='List rules')
    list_parser.add_argument('--type', choices=['static', 'dynamic', 'all'],
                            default='all', help='Type of rules to list')

    # Add rule
    add_parser = rules_subparsers.add_parser('add', help='Add a new rule')
    add_parser.add_argument('--name', required=True, help='Rule name')
    add_parser.add_argument('--type', required=True, help='Rule type')
    add_parser.add_argument('--feature', required=True, help='Feature name')
    add_parser.add_argument('--operator', required=True, help='Comparison operator')
    add_parser.add_argument('--threshold', required=True, type=float, help='Threshold value')

    # Update rule
    update_parser = rules_subparsers.add_parser('update', help='Update a rule')
    update_parser.add_argument('--id', required=True, help='Rule ID')
    update_parser.add_argument('--active', type=bool, help='Set rule active/inactive')
    update_parser.add_argument('--threshold', type=float, help='New threshold value')

    return rules_parser


def setup_thresholds_parser(subparsers):
    """Setup the parser for the thresholds command"""
    thresholds_parser = subparsers.add_parser(
        'thresholds',
        help='Manage anomaly detection thresholds'
    )

    thresholds_subparsers = thresholds_parser.add_subparsers(dest='thresholds_command')

    # Show thresholds
    show_parser = thresholds_subparsers.add_parser('show', help='Show current thresholds')

    # Adjust thresholds
    adjust_parser = thresholds_subparsers.add_parser('adjust', help='Adjust thresholds')
    adjust_parser.add_argument('--auto', action='store_true',
                              help='Automatically adjust based on recent data')
    adjust_parser.add_argument('--critical', type=float, help='Set critical threshold')
    adjust_parser.add_argument('--high', type=float, help='Set high threshold')
    adjust_parser.add_argument('--medium', type=float, help='Set medium threshold')
    adjust_parser.add_argument('--low', type=float, help='Set low threshold')

    return thresholds_parser


def setup_profiles_parser(subparsers):
    """Setup the parser for the profiles command"""
    profiles_parser = subparsers.add_parser(
        'profiles',
        help='Manage user profiles'
    )

    profiles_subparsers = profiles_parser.add_subparsers(dest='profiles_command')

    # Show profile
    show_parser = profiles_subparsers.add_parser('show', help='Show user profile')
    show_parser.add_argument('--user-id', required=True, help='User ID')

    # List profiles
    list_parser = profiles_subparsers.add_parser('list', help='List user profiles')
    list_parser.add_argument('--limit', type=int, default=10, help='Number of profiles to show')
    list_parser.add_argument('--sort-by', choices=['risk', 'activity', 'recent'],
                            default='risk', help='Sort criteria')

    return profiles_parser


def handle_train(args):
    """Handle the train command"""
    logger.info("Starting training pipeline")

    # Create a shared data file path if provided
    data_file = args.data

    # Train classifier if not skipped
    if not args.skip_classifier:
        train_classifier(data_file, args.test_size)
        logger.info("Fraud classifier training completed")

    # Train anomaly detector if not skipped
    if not args.skip_detector:
        train_anomaly_detector(data_file, args.contamination)
        logger.info("Anomaly detector training completed")

    # Update rules if requested
    if args.update_rules:
        update_rules()

    logger.info("Training pipeline completed successfully")


def handle_infer(args):
    """Handle the infer command"""
    import json

    logger.info("Starting inference pipeline")

    # Load expert systems
    try:
        mediator = load_experts()
    except Exception as e:
        logger.error(f"Failed to initialize expert systems: {str(e)}")
        sys.exit(1)

    # Process based on input method
    try:
        if args.transaction:
            # Process single transaction from command line
            transaction = json.loads(args.transaction)
            result = process_single_transaction(mediator, transaction)
            print(json.dumps(result, indent=2))

        elif args.input:
            # Process from input file
            results = process_from_file(mediator, args.input, args.output)
            if not args.output:
                print(json.dumps(results, indent=2))

        elif args.stream:
            # Process from stream source
            process_from_stream(mediator, args.stream)

        else:
            logger.error("No input method specified")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error in inference pipeline: {str(e)}")
        sys.exit(1)

    logger.info("Inference pipeline completed successfully")


def handle_evaluate(args):
    """Handle the evaluate command"""
    logger.info(f"Model evaluation for {args.model} not implemented yet")
    # This would be implemented to evaluate model performance
    # on a test dataset


def handle_config(args):
    """Handle the config command"""
    from infrastructure.config import load_paths, load_params

    if args.show:
        # Show current configuration
        paths = load_paths()
        params = load_params()

        print("=== Paths Configuration ===")
        print(paths)
        print("\n=== Model Parameters ===")
        print(params)

    elif args.update:
        logger.info("Configuration update not implemented yet")
        # This would be implemented to update configuration parameters

    else:
        logger.error("No config action specified")
        sys.exit(1)


def handle_setup(args):
    """Handle the setup command"""
    logger.info("Initializing fraud detection system")

    if not args.skip_db:
        # Create databases
        import sqlite3
        import os
        from datetime import datetime
        from infrastructure.config import load_paths, get_project_root

        # Create user profiles database
        paths = load_paths()
        user_profiles_db_path = os.path.join(
            get_project_root(),
            paths['shared']['user_profiles']
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(user_profiles_db_path), exist_ok=True)

        # Create user profiles database
        conn = sqlite3.connect(user_profiles_db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            avg_amount REAL DEFAULT 0,
            max_amount REAL DEFAULT 0,
            avg_daily_transactions REAL DEFAULT 0,
            avg_transaction_interval REAL DEFAULT 0,
            typical_merchants TEXT,
            last_countries TEXT,
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
            time_window TEXT NOT NULL,
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
            pattern_type TEXT NOT NULL,
            pattern_value TEXT NOT NULL,
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

        logger.info(f"Created user profiles database at {user_profiles_db_path}")

        # Create dynamic rules database
        dynamic_rules_db_path = os.path.join(
            get_project_root(),
            paths['models']['classifier']['dynamic']
        )

        # Ensure directory exists
        os.makedirs(os.path.dirname(dynamic_rules_db_path), exist_ok=True)

        # Create dynamic rules database
        conn = sqlite3.connect(dynamic_rules_db_path)
        cursor = conn.cursor()

        # Create rules table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS rules (
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
        )
        ''')

        # Create rule performance tracking table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS rule_performance (
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
        )
        ''')

        # Create feature statistics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feature_statistics (
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
        )
        ''')

        # Create rule generation history table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS rule_generation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_version TEXT,
            rules_added INTEGER,
            rules_removed INTEGER,
            rules_modified INTEGER,
            performance_change REAL
        )
        ''')

        # Commit changes and close connection
        conn.commit()
        conn.close()

        logger.info(f"Created dynamic rules database at {dynamic_rules_db_path}")

        # Create recent frauds file if it doesn't exist
        recent_frauds_path = os.path.join(
            get_project_root(),
            paths['shared']['fraud_history']
        )

        if not os.path.exists(recent_frauds_path):
            # Ensure directory exists
            os.makedirs(os.path.dirname(recent_frauds_path), exist_ok=True)

            # Create empty recent frauds file
            with open(recent_frauds_path, 'w') as f:
                json.dump({
                    "recent_frauds": [],
                    "fraud_patterns": [],
                    "last_updated": datetime.now().isoformat()
                }, f, indent=2)

            logger.info(f"Created recent frauds file at {recent_frauds_path}")

    logger.info("System initialization completed")


def handle_rules(args):
    """Handle the rules command"""
    if not hasattr(args, 'rules_command') or args.rules_command is None:
        logger.error("No rules command specified")
        sys.exit(1)

    if args.rules_command == 'list':
        # List rules
        if args.type in ['static', 'all']:
            static_rules = rule_manager.get_static_rules()
            print("=== Static Rules ===")
            print(json.dumps(static_rules, indent=2))

        if args.type in ['dynamic', 'all']:
            dynamic_rules = rule_manager.get_dynamic_rules()
            print("=== Dynamic Rules ===")
            print(json.dumps(dynamic_rules, indent=2))

    elif args.rules_command == 'add':
        # Add rule
        rule_manager.add_rule(
            name=args.name,
            rule_type=args.type,
            feature=args.feature,
            operator=args.operator,
            threshold=args.threshold
        )
        logger.info(f"Rule '{args.name}' added successfully")

    elif args.rules_command == 'update':
        # Update rule
        updates = {}
        if args.active is not None:
            updates['is_active'] = args.active
        if args.threshold is not None:
            updates['threshold'] = args.threshold

        rule_manager.update_rule(args.id, updates)
        logger.info(f"Rule {args.id} updated successfully")


def handle_thresholds(args):
    """Handle the thresholds command"""
    adjuster = ThresholdAdjuster()

    if not hasattr(args, 'thresholds_command') or args.thresholds_command is None:
        logger.error("No thresholds command specified")
        sys.exit(1)

    if args.thresholds_command == 'show':
        # Show thresholds
        thresholds = adjuster.thresholds
        print("=== Current Anomaly Thresholds ===")
        print(json.dumps(thresholds, indent=2))

    elif args.thresholds_command == 'adjust':
        # Adjust thresholds
        if args.auto:
            # Auto-adjust based on recent data
            new_thresholds = adjuster.adjust_thresholds()
            logger.info(f"Thresholds automatically adjusted: {new_thresholds}")
        else:
            # Manual adjustment
            updates = {}
            if args.critical is not None:
                updates['critical'] = args.critical
            if args.high is not None:
                updates['high'] = args.high
            if args.medium is not None:
                updates['medium'] = args.medium
            if args.low is not None:
                updates['low'] = args.low

            for key, value in updates.items():
                adjuster.thresholds[key] = value

            adjuster.save_thresholds()
            logger.info(f"Thresholds manually adjusted: {adjuster.thresholds}")


def handle_profiles(args):
    """Handle the profiles command"""
    if not hasattr(args, 'profiles_command') or args.profiles_command is None:
        logger.error("No profiles command specified")
        sys.exit(1)

    if args.profiles_command == 'show':
        # Show user profile
        profile = context_manager.get_user_profile(args.user_id)
        if profile:
            print(f"=== Profile for User {args.user_id} ===")
            print(json.dumps(profile, indent=2))
        else:
            print(f"No profile found for user {args.user_id}")

    elif args.profiles_command == 'list':
        # List profiles
        profiles = context_manager.list_user_profiles(
            limit=args.limit,
            sort_by=args.sort_by
        )
        print(f"=== User Profiles (sorted by {args.sort_by}) ===")
        for profile in profiles:
            print(f"User: {profile['user_id']}, Risk: {profile['risk_score']:.2f}, "
                  f"Avg Amount: {profile['avg_amount']:.2f}")


def main():
    """Main entry point"""
    # Create the top-level parser
    parser = argparse.ArgumentParser(
        description='Fraud Detection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Create subparsers for each command
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Setup parsers for each command
    setup_train_parser(subparsers)
    setup_infer_parser(subparsers)
    setup_evaluate_parser(subparsers)
    setup_config_parser(subparsers)
    setup_setup_parser(subparsers)
    setup_rules_parser(subparsers)
    setup_thresholds_parser(subparsers)
    setup_profiles_parser(subparsers)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Handle commands
    if args.command == 'train':
        handle_train(args)
    elif args.command == 'infer':
        handle_infer(args)
    elif args.command == 'evaluate':
        handle_evaluate(args)
    elif args.command == 'config':
        handle_config(args)
    elif args.command == 'setup':
        handle_setup(args)
    elif args.command == 'rules':
        handle_rules(args)
    elif args.command == 'thresholds':
        handle_thresholds(args)
    elif args.command == 'profiles':
        handle_profiles(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()