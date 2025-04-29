# Fraud Detection Expert Systems

This directory contains the expert systems used in the fraud detection system. Each expert specializes in a different approach to fraud detection, and they work together through the Expert Mediator to make final decisions.

## Architecture

The fraud detection system uses a multi-expert approach:

1. **Fraud Classifier Expert** - Supervised learning approach that identifies known fraud patterns
2. **Anomaly Detector Expert** - Unsupervised learning approach that identifies unusual transactions
3. **Expert Mediator** - Coordinates decisions between experts using configurable fusion rules

## Directory Structure

```
experts/
├── common/                  # Shared utilities and functions
│   ├── __init__.py
│   └── utils.py             # Common utility functions
├── fraud_classifier/        # Supervised learning expert
│   ├── models/              # Trained classifier models
│   ├── rules/               # Fraud detection rules
│   ├── evaluation/          # Model evaluation reports
│   ├── predict.py           # Prediction logic
│   └── train.py             # Training logic
├── anomaly_detector/        # Unsupervised learning expert
│   ├── models/              # Trained anomaly detection models
│   ├── thresholds/          # Anomaly threshold configuration
│   ├── evaluation/          # Model evaluation reports
│   ├── detect.py            # Detection logic
│   └── train.py             # Training logic
└── coordination/            # Expert coordination
    ├── shared_context/      # Shared context data
    ├── context_manager.py   # Context management
    ├── fusion_rules.yaml    # Rules for combining expert opinions
    └── mediator.py          # Expert mediator implementation
```

## Expert Systems

### Fraud Classifier Expert

The Fraud Classifier Expert uses supervised machine learning to identify known fraud patterns. It combines:

- **ML Model**: A trained classifier (typically logistic regression) that predicts fraud probability
- **Rule Engine**: Domain-specific rules that identify suspicious transaction characteristics
- **Confidence Scoring**: Combined ML and rule-based confidence score

Key features:
- Handles missing features gracefully
- Caches predictions for performance
- Provides feature importance for explainability
- Falls back to rule-based evaluation when ML prediction fails

### Anomaly Detector Expert

The Anomaly Detector Expert uses unsupervised learning to identify unusual transactions that deviate from normal patterns. It includes:

- **Isolation Forest**: Identifies transactions that are statistical outliers
- **Dynamic Thresholds**: Automatically adjusts anomaly thresholds based on recent data
- **Severity Classification**: Categorizes anomalies by severity (CRITICAL, HIGH, MEDIUM, LOW, NORMAL)

Key features:
- Adapts to changing transaction patterns
- Handles missing features gracefully
- Provides heuristic scoring as fallback
- Considers cluster deviation for additional context

### Expert Mediator

The Expert Mediator coordinates decisions between expert systems using configurable fusion rules. It implements:

- **Priority Rules**: Rules that can override other decisions (e.g., high confidence fraud)
- **Agreement Rules**: Rules for when experts agree or disagree
- **Context Rules**: Rules based on user context (e.g., unusual location)
- **Weighted Scoring**: Weighted combination of expert scores
- **Decision Explanation**: Detailed explanation of decision rationale

Key features:
- Configurable fusion rules in YAML format
- Different rule sets for different scenarios (new users, high-value transactions)
- Caches decisions for performance
- Maintains shared context between experts

## Common Utilities

The common utilities provide shared functionality across expert systems:

- **Feature Extraction**: Extract features from transactions for ML models
- **Model Compatibility**: Check if models are compatible with available features
- **Heuristic Scoring**: Calculate heuristic fraud and anomaly scores
- **Safe Prediction**: Make predictions with error handling
- **Model Versioning**: Track model versions
- **Caching**: Cache predictions for performance

## Usage

Each expert can be used independently or together through the mediator:

```python
# Initialize experts
classifier = FraudClassifierExpert(classifier_model, rules_path)
detector = AnomalyDetectorExpert(anomaly_model, context)

# Initialize mediator
mediator = ExpertMediator(classifier, detector)

# Process transaction
result = mediator.process_transaction(transaction)
print(f"Decision: {result['decision']}")
print(f"Explanation: {result['explanation']}")
```

## Testing

Unit tests for the expert systems are available in `tests/test_experts.py`. Run the tests with:

```
python -m unittest tests/test_experts.py
```

## Configuration

The expert systems are configured through various configuration files:

- **Fusion Rules**: `experts/coordination/fusion_rules.yaml`
- **Anomaly Thresholds**: `experts/anomaly_detector/thresholds/base_thresholds.yaml`
- **Fraud Rules**: `experts/fraud_classifier/rules/static_rules.json`

## Performance Considerations

The expert systems include several performance optimizations:

- **Prediction Caching**: Caches predictions to avoid redundant computation
- **Feature Extraction**: Efficiently extracts features from transactions
- **Decision Caching**: Caches final decisions for previously seen transactions
- **Graceful Degradation**: Falls back to simpler methods when ML models fail
