#!/usr/bin/env python
"""
Inference pipeline for credit card fraud detection system.
Processes transactions through both expert systems and mediator.
"""

import os
import sys
import argparse
import pandas as pd
import json
import time
from typing import Dict, List, Any, Union

from infrastructure.config import load_paths, load_params, get_project_root
from infrastructure.utils import (
    logger, log_execution_time, load_model, load_transaction_batch,
    save_results
)
from infrastructure.memory import ContextBuffer, KnowledgeGraph
from experts.fraud_classifier.predict import FraudClassifierExpert
from experts.anomaly_detector.detect import AnomalyDetectorExpert
from experts.coordination.mediator import ExpertMediator


@log_execution_time
def load_experts():
    """Load trained expert models and initialize system"""
    try:
        # Load models
        classifier_model = load_model('classifier')
        anomaly_model = load_model('anomaly')
        
        # Initialize context
        context = ContextBuffer()
        knowledge_graph = KnowledgeGraph()
        
        # Initialize experts
        classifier = FraudClassifierExpert(classifier_model, context)
        detector = AnomalyDetectorExpert(anomaly_model, context)
        
        # Initialize mediator
        mediator = ExpertMediator(classifier, detector)
        
        logger.info("Expert systems loaded successfully")
        return mediator
        
    except Exception as e:
        logger.error(f"Error loading expert systems: {str(e)}")
        raise


@log_execution_time
def process_single_transaction(mediator, transaction):
    """Process a single transaction through the fraud detection system"""
    result = mediator.process_transaction(transaction)
    
    # Log the result
    decision = result.get('decision', 'UNKNOWN')
    transaction_id = transaction.get('id', 'unknown')
    logger.info(f"Transaction {transaction_id} processed with decision: {decision}")
    
    return result


@log_execution_time
def process_batch(mediator, transactions):
    """Process a batch of transactions"""
    results = []
    
    for idx, transaction in enumerate(transactions):
        try:
            # Add transaction index as ID if not present
            if 'id' not in transaction:
                transaction['id'] = f"TX_{idx}"
                
            result = process_single_transaction(mediator, transaction)
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing transaction {idx}: {str(e)}")
            # Add failed transaction to results
            results.append({
                'transaction_id': transaction.get('id', f"TX_{idx}"),
                'error': str(e),
                'decision': 'ERROR'
            })
    
    return results


@log_execution_time
def process_from_file(mediator, input_file, output_file=None):
    """Process transactions from file"""
    try:
        # Load transactions
        transactions_df = load_transaction_batch(input_file)
        transactions = transactions_df.to_dict('records')
        
        logger.info(f"Loaded {len(transactions)} transactions from {input_file}")
        
        # Process transactions
        results = process_batch(mediator, transactions)
        
        # Save results if output file specified
        if output_file:
            save_results(results, output_file)
            logger.info(f"Results saved to {output_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error processing file {input_file}: {str(e)}")
        raise


@log_execution_time
def process_from_stream(mediator, stream_source):
    """Process transactions from stream source"""
    try:
        # This is a placeholder for stream processing
        # In a real system, this would connect to Kafka, RabbitMQ, etc.
        logger.info(f"Stream processing from {stream_source} not implemented yet")
        
        # Placeholder implementation using polling
        while True:
            # Check for new transactions
            try:
                # Simulate batch arrival every 5 seconds
                time.sleep(5)
                
                # Process batch
                # In a real system, this would pull from the stream
                # For now, we'll just break after one iteration
                break
                
            except KeyboardInterrupt:
                logger.info("Stream processing interrupted")
                break
            
    except Exception as e:
        logger.error(f"Error in stream processing: {str(e)}")
        raise


def main():
    """Main inference pipeline entry point"""
    parser = argparse.ArgumentParser(description='Run fraud detection inference')
    parser.add_argument('--input', type=str, help='Input file with transactions')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--stream', type=str, help='Stream source for real-time processing')
    parser.add_argument('--transaction', type=str, help='Single JSON transaction to process')
    
    args = parser.parse_args()
    
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
            parser.print_help()
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error in inference pipeline: {str(e)}")
        sys.exit(1)
        
    logger.info("Inference pipeline completed successfully")


if __name__ == "__main__":
    main()