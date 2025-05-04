#!/usr/bin/env python
"""
Script to reset the recent_frauds.json file to align with the core dataset features.

This script:
1. Creates a clean version of the recent_frauds.json file
2. Ensures it only includes the core features from the dataset
3. Maintains the expected structure for the fraud detection system

Usage:
    python reset_recent_frauds.py
"""

import os
import json
from datetime import datetime

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

def reset_recent_frauds():
    """Reset the recent_frauds.json file to align with core dataset features"""
    # Get the file path from configuration
    paths = load_paths()
    file_path = os.path.join(
        get_project_root(),
        paths.get('shared', {}).get('fraud_history', 'experts/coordination/shared_context/recent_frauds.json')
    )
    
    # Define the core dataset features
    core_features = [
        'distance_from_home',
        'distance_from_last_transaction',
        'ratio_to_median_purchase_price',
        'repeat_retailer',
        'used_chip',
        'used_pin_number',
        'online_order'
    ]
    
    # Create a clean structure with one example entry using only core features
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
    
    print(f"✅ Created clean recent_frauds.json file at {file_path}")
    print(f"✅ File now contains only the core dataset features: {', '.join(core_features)}")
    
    # Return the path for verification
    return file_path

def verify_file(file_path):
    """Verify the file was created correctly"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if the file has the expected structure
        if "recent_frauds" in data and "fraud_patterns" in data and "last_updated" in data:
            print("✅ File structure is correct")
            
            # Check if recent_frauds contains entries
            if len(data["recent_frauds"]) > 0:
                print("✅ File contains example fraud entries")
                
                # Check if the entries have the correct features
                first_entry = data["recent_frauds"][0]
                if "transaction_features" in first_entry:
                    features = first_entry["transaction_features"]
                    print(f"✅ First entry contains these features: {', '.join(features.keys())}")
                else:
                    print("❌ First entry does not have transaction_features")
            else:
                print("❌ File does not contain any fraud entries")
        else:
            print("❌ File structure is incorrect")
    except Exception as e:
        print(f"❌ Error verifying file: {str(e)}")

if __name__ == "__main__":
    print("Resetting recent_frauds.json file to align with core dataset features...")
    file_path = reset_recent_frauds()
    verify_file(file_path)
    print("\nDone! The recent_frauds.json file has been reset.")
    print("As you run the fraud detection system with your dataset, it will populate")
    print("this file with fraud entries that match your core features.")
