"""
Configuration module for the Enhanced Ant Bot System.
"""

__version__ = "1.0.0"

# Import core configurations
try:
    from .core_config import *
except ImportError:
    # Fallback if core_config doesn't exist
    pass 