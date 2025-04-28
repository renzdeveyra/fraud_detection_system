#!/usr/bin/env python
"""
Main entry point for the Fraud Detection System.

This script provides a unified command-line interface to access
all functionality of the fraud detection system, including:
- Training the expert systems
- Running inference on transactions
- Evaluating model performance
- Managing system configuration

Usage:
    python main.py train [options]
    python main.py infer [options]
    python main.py evaluate [options]
    python main.py config [options]
"""

import sys
import argparse
from infrastructure.utils import logger

# Import pipeline modules
from pipelines.training_pipeline import train_classifier, train_anomaly_detector, update_rules
from pipelines.inference_pipeline import (
    load_experts, process_single_transaction,
    process_from_file, process_from_stream
)


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

    # Parse arguments
    args = parser.parse_args()

    # Handle commands
    if args.command == 'train':
        handle_train(args)
    elif args.command == 'infer':
        handle_infer(args)
    elif args.command == 'evaluate':
        handle_evaluate(args)
    elif args.command == 'config':
        handle_config(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()