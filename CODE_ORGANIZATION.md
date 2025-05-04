# Code Organization Guidelines

This document outlines the code organization standards for the Fraud Detection System.

## Import Standards

We use the following import order convention:

1. Standard library imports
2. Third-party library imports
3. First-party imports (our own modules)
4. Local imports (from the same package)

Example:
```python
# Standard library
import os
import json
from typing import Dict, List, Any

# Third-party libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# First-party imports
from infrastructure.config import load_paths
from infrastructure.utils import logger

# Local imports
from .utils import calculate_score
```

We use isort to maintain consistent import ordering. Configuration is in `.isort.cfg`.

## Code Style

We follow PEP 8 style guidelines with some customizations defined in `.style.yapf`.

### Docstrings

We use Google-style docstrings:

```python
def function_name(param1, param2):
    """
    Short description of the function.

    Longer description with more details if needed.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ExceptionType: When and why this exception is raised
    """
```

## Project Structure

- `/config`: Configuration files
- `/data`: Raw and processed data
- `/experts`: Expert system implementations
  - `/fraud_classifier`: Supervised learning model
  - `/anomaly_detector`: Unsupervised learning model
  - `/coordination`: Expert mediator
  - `/common`: Shared utilities for experts
- `/infrastructure`: Core utilities and shared components
  - `/config`: Configuration management
  - `/utils`: Utility functions
  - `/memory`: Shared context and knowledge management
- `/pipelines`: Training and inference workflows
- `/logs`: System logs
- `/src`: Legacy code (use infrastructure modules for new code)

## Utility Functions

Common utility functions are centralized in the `infrastructure/utils` package:

- `logger.py`: Logging utilities
- `data_loader.py`: Data loading and processing
- `model_ops.py`: Model operations (save, load)
- `caching.py`: Caching mechanisms
- `error_handling.py`: Error handling utilities

## Configuration Management

All configuration should be loaded through the `infrastructure/config` module:

```python
from infrastructure.config import load_paths, load_params, get_project_root

# Load file paths
paths = load_paths()

# Load model parameters
params = load_params()

# Get project root directory
root_dir = get_project_root()
```

## Error Handling

Use the standardized error handling utilities:

```python
from infrastructure.utils import handle_errors, ProcessingError

@handle_errors(error_type=ProcessingError, log_level="error")
def process_data(data):
    # Function implementation
    pass
```

## Caching

Use the standardized caching utilities:

```python
from infrastructure.utils import ModelCache, TransactionCache

# Create a cache for model predictions
model_cache = ModelCache(max_size=1000, model_name="classifier")

# Create a cache for transaction processing
transaction_cache = TransactionCache(max_size=1000)
```

## Legacy Code Compatibility

For backward compatibility with legacy code, use the compatibility layer in `src/utils_compat.py`.
New code should use the infrastructure modules directly.
