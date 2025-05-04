"""
Utility functions for the fraud detection system.
This module is maintained for backward compatibility.
New code should use the infrastructure module directly.
"""

# Import from compatibility layer
from src.utils_compat import (
    load_config,
    setup_logger,
    save_model,
    load_model
)

# Re-export for backward compatibility
__all__ = [
    'load_config',
    'setup_logger',
    'save_model',
    'load_model'
]
