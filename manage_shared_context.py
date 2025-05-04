#!/usr/bin/env python
"""
Script to manage the shared context for the fraud detection system.

This script provides utilities to:
1. Reset the recent_frauds.json file to align with core dataset features
2. Clean up existing entries to remove non-core features
3. View the current structure of the file
4. Add example entries with the correct structure

Usage:
    python manage_shared_context.py reset   # Reset the file completely
    python manage_shared_context.py clean   # Clean up existing entries
    python manage_shared_context.py view    # View the current file
    python manage_shared_context.py add     # Add example entries
"""

import os
import json
import argparse
from datetime import datetime
import copy

# Import from the fraud detection system
try:
    from infrastructure.config import load_paths, get_project_root
    from infrastructure.utils import logger
except ImportError:
    # Fallback for direct script execution
    import sys
    sys.path.append('.')
    from infrastructure.config import load_paths, get_project_root
    from infrastructure.utils import logger

# Define the core dataset features
CORE_FEATURES = [
    'distance_from_home',
    'distance_from_last_transaction',
    'ratio_to_median_purchase_price',
    'repeat_retailer',
    'used_chip',
    'used_pin_number',
    'online_order'
]

def get_file_path():
    """Get the path to the recent_frauds.json file"""
    paths = load_paths()
    return os.path.join(
        get_project_root(),
        paths.get('shared', {}).get('fraud_history', 'experts/coordination/shared_context/recent_frauds.json')
    )

def reset_file():
    """Reset the recent_frauds.json file to a clean state"""
    file_path = get_file_path()
    
    # Create a clean structure with one example entry
    clean_data = {
        "recent_frauds": [
            {
                "transaction_features": {
                    "distance_from_home": 150.0,
                    "distance_from_last_transaction": 120.0,
                    "ratio_to_median_purchase_price": 5.2,
                    "repeat_retailer": 0,
                    "used_chip": 0,
                    "used_pin_number": 0,
                    "online_order": 1
                },
                "classifier_score": 0.95,
                "anomaly_score": -0.8,
                "rule_violations": 7,
                "timestamp": datetime.now().isoformat()
            }
        ],
        "fraud_patterns": [],
        "last_updated": datetime.now().isoformat()
    }
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save the clean file
    with open(file_path, 'w') as f:
        json.dump(clean_data, f, indent=2)
    
    print(f"✅ Reset recent_frauds.json file at {file_path}")
    return file_path

def clean_file():
    """Clean up existing entries to remove non-core features"""
    file_path = get_file_path()
    
    try:
        # Load existing data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Make a copy for comparison
        original_count = len(data.get("recent_frauds", []))
        
        # Clean up each entry
        cleaned_frauds = []
        for entry in data.get("recent_frauds", []):
            # Create a new clean entry
            clean_entry = {
                "transaction_features": {},
                "timestamp": entry.get("timestamp", datetime.now().isoformat())
            }
            
            # Copy classifier and anomaly scores if they exist
            if "classifier_score" in entry:
                clean_entry["classifier_score"] = entry["classifier_score"]
            elif "classifier_insights" in entry:
                clean_entry["classifier_score"] = float(entry["classifier_insights"]) / 10.0
            
            if "anomaly_score" in entry:
                clean_entry["anomaly_score"] = entry["anomaly_score"]
            elif "anomaly_insights" in entry and "score" in entry["anomaly_insights"]:
                clean_entry["anomaly_score"] = entry["anomaly_insights"]["score"]
            
            # Copy rule violations if they exist
            if "rule_violations" in entry:
                clean_entry["rule_violations"] = entry["rule_violations"]
            
            # Extract core features from the entry
            features_dict = None
            if "transaction_features" in entry:
                features_dict = entry["transaction_features"]
            elif "features" in entry:
                features_dict = entry["features"]
            
            # If we found features, extract only the core ones
            if features_dict:
                for feature in CORE_FEATURES:
                    if feature in features_dict:
                        clean_entry["transaction_features"][feature] = features_dict[feature]
            
            # Only add entries that have at least some core features
            if len(clean_entry["transaction_features"]) > 0:
                cleaned_frauds.append(clean_entry)
        
        # Update the data with cleaned entries
        data["recent_frauds"] = cleaned_frauds
        data["last_updated"] = datetime.now().isoformat()
        
        # Save the cleaned file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Cleaned recent_frauds.json file at {file_path}")
        print(f"   Original entries: {original_count}")
        print(f"   Cleaned entries: {len(cleaned_frauds)}")
        return file_path
    
    except Exception as e:
        print(f"❌ Error cleaning file: {str(e)}")
        return None

def view_file():
    """View the current structure of the file"""
    file_path = get_file_path()
    
    try:
        # Load existing data
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Print summary
        print(f"=== Recent Frauds File Summary ===")
        print(f"File path: {file_path}")
        print(f"Last updated: {data.get('last_updated', 'Unknown')}")
        print(f"Number of fraud entries: {len(data.get('recent_frauds', []))}")
        print(f"Number of fraud patterns: {len(data.get('fraud_patterns', []))}")
        
        # Print first entry if available
        if len(data.get("recent_frauds", [])) > 0:
            print("\n=== First Fraud Entry ===")
            first_entry = data["recent_frauds"][0]
            print(json.dumps(first_entry, indent=2))
            
            # Check if it has the correct structure
            if "transaction_features" in first_entry:
                features = set(first_entry["transaction_features"].keys())
                core_features = set(CORE_FEATURES)
                
                print("\n=== Feature Analysis ===")
                print(f"Core features present: {features.intersection(core_features)}")
                print(f"Core features missing: {core_features - features}")
                print(f"Non-core features present: {features - core_features}")
        
        return data
    
    except Exception as e:
        print(f"❌ Error viewing file: {str(e)}")
        return None

def add_examples():
    """Add example entries with the correct structure"""
    file_path = get_file_path()
    
    try:
        # Load existing data
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except:
            # Create new data if file doesn't exist
            data = {
                "recent_frauds": [],
                "fraud_patterns": [],
                "last_updated": datetime.now().isoformat()
            }
        
        # Create example entries
        examples = [
            {
                "transaction_features": {
                    "distance_from_home": 150.0,
                    "distance_from_last_transaction": 120.0,
                    "ratio_to_median_purchase_price": 5.2,
                    "repeat_retailer": 0,
                    "used_chip": 0,
                    "used_pin_number": 0,
                    "online_order": 1
                },
                "classifier_score": 0.95,
                "anomaly_score": -0.8,
                "rule_violations": 7,
                "timestamp": datetime.now().isoformat()
            },
            {
                "transaction_features": {
                    "distance_from_home": 25.0,
                    "distance_from_last_transaction": 80.0,
                    "ratio_to_median_purchase_price": 3.5,
                    "repeat_retailer": 0,
                    "used_chip": 1,
                    "used_pin_number": 0,
                    "online_order": 0
                },
                "classifier_score": 0.75,
                "anomaly_score": -0.5,
                "rule_violations": 4,
                "timestamp": datetime.now().isoformat()
            },
            {
                "transaction_features": {
                    "distance_from_home": 5.0,
                    "distance_from_last_transaction": 2.0,
                    "ratio_to_median_purchase_price": 0.8,
                    "repeat_retailer": 1,
                    "used_chip": 1,
                    "used_pin_number": 1,
                    "online_order": 0
                },
                "classifier_score": 0.05,
                "anomaly_score": 0.2,
                "rule_violations": 0,
                "timestamp": datetime.now().isoformat()
            }
        ]
        
        # Add examples to the data
        data["recent_frauds"] = examples + data.get("recent_frauds", [])
        data["last_updated"] = datetime.now().isoformat()
        
        # Save the updated file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Added {len(examples)} example entries to recent_frauds.json")
        return file_path
    
    except Exception as e:
        print(f"❌ Error adding examples: {str(e)}")
        return None

def main():
    """Main function to parse arguments and execute commands"""
    parser = argparse.ArgumentParser(
        description="Manage the shared context for the fraud detection system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Add commands
    parser.add_argument('command', choices=['reset', 'clean', 'view', 'add'],
                       help='Command to execute')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == 'reset':
        reset_file()
    elif args.command == 'clean':
        clean_file()
    elif args.command == 'view':
        view_file()
    elif args.command == 'add':
        add_examples()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
