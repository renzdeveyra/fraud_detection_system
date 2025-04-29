# Shared Context for Fraud Detection System

This directory contains shared context data used by the fraud detection system.

## Contents

### recent_frauds.json

This file stores recent fraud patterns detected by the system. It's used by the Expert Mediator to compare new transactions against known fraud patterns.

The file contains:
- A list of recent fraud transactions with their core features
- Classifier and anomaly scores for each fraud transaction
- Rule violations for each fraud transaction
- Timestamp information

The system uses this data to identify transactions that match patterns of previously detected fraud.

## Core Dataset Features

The fraud detection system focuses on these core features:
- `distance_from_home` - Distance from home where the transaction happened
- `distance_from_last_transaction` - Distance from last transaction
- `ratio_to_median_purchase_price` - Ratio of purchase price to median purchase price
- `repeat_retailer` - Whether the transaction is with a repeat retailer (0/1)
- `used_chip` - Whether chip was used (0/1)
- `used_pin_number` - Whether PIN was used (0/1)
- `online_order` - Whether it was an online order (0/1)

## Usage

The shared context is primarily used by:
1. The Expert Mediator to make context-aware decisions
2. The Context Manager to track fraud patterns
3. The Fraud Classifier and Anomaly Detector to compare transactions against known patterns

## Maintenance

The `recent_frauds.json` file is automatically maintained by the system:
- New fraud transactions are added to the beginning of the list
- The list is limited to the 100 most recent fraud transactions
- The file is updated whenever a transaction is identified as fraudulent
