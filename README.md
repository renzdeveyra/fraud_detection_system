# Transaction-Based Fraud Detection System

A streamlined credit card fraud detection system that combines supervised learning, anomaly detection, and rule-based approaches, focusing exclusively on transaction-based features.

## Architecture

The system is built around two main expert components that work together:

1. **Fraud Classifier Expert**: A supervised machine learning model (Logistic Regression) that's trained to identify known fraud patterns based on historical data.

2. **Anomaly Detector Expert**: An unsupervised machine learning model (Isolation Forest) that identifies unusual transactions that deviate from normal patterns.

3. **Expert Mediator**: A coordination component that combines the outputs from both experts to make a final decision about each transaction.

## Core Dataset Features

The system focuses exclusively on these transaction-based features:

- `distance_from_home` - Distance from home where the transaction happened
- `distance_from_last_transaction` - Distance from last transaction
- `ratio_to_median_purchase_price` - Ratio of purchase price to median purchase price
- `repeat_retailer` - Whether the transaction is with a repeat retailer (0/1)
- `used_chip` - Whether chip was used (0/1)
- `used_pin_number` - Whether PIN was used (0/1)
- `online_order` - Whether it was an online order (0/1)

These features provide a comprehensive view of transaction patterns without relying on user profile information.

## Installation

1. Clone the repository:

   ```
   git clone <repository-url>
   cd fraud_detection_system
   ```

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Prepare your data:
   - Place your raw transaction data in `data/raw/creditcard.csv`
   - Or specify a custom data path when running the training pipeline

## Usage

The system provides a unified command-line interface through `main.py`:

### Training Models

Train both expert systems:

```
python main.py train [options]
```

Options:

- `--data PATH`: Path to data file (CSV)
- `--test-size SIZE`: Test split size (default: 0.2)
- `--contamination RATE`: Anomaly contamination rate (default: 0.01)
- `--skip-classifier`: Skip classifier training
- `--skip-detector`: Skip anomaly detector training
- `--update-rules`: Update rules after training

### Running Inference

Process transactions through the fraud detection system:

```
python main.py infer [options]
```

Options:

- `--input FILE`: Input file with transactions
- `--output FILE`: Output file for results
- `--stream SOURCE`: Stream source for real-time processing
- `--transaction JSON`: Single JSON transaction to process

### Evaluating Models

Evaluate model performance:

```
python main.py evaluate [options]
```

Options:

- `--data PATH`: Path to evaluation data file (CSV)
- `--model TYPE`: Model to evaluate (classifier, anomaly, or both)

### Managing Configuration

View or update system configuration:

```
python main.py config [options]
```

Options:

- `--show`: Show current configuration
- `--update KEY=VALUE`: Update configuration parameter

## Project Structure

- `/config`: Configuration files
- `/data`: Raw and processed data
- `/experts`: Expert system implementations
  - `/fraud_classifier`: Supervised learning model
  - `/anomaly_detector`: Unsupervised learning model
  - `/coordination`: Expert mediator
- `/infrastructure`: Core utilities and shared components
  - `/config`: Configuration management
  - `/utils`: Utility functions
  - `/memory`: Shared context and knowledge management
- `/pipelines`: Training and inference workflows
- `/logs`: System logs

## License

[Your License]

## Contributing

[Contribution guidelines]
