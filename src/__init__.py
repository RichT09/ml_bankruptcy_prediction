"""
Bankruptcy Prediction ML Package
Master Finance - Advanced Programming
HEC Lausanne - Fall 2025
"""

__version__ = "1.0.0"
__author__ = "Master Finance Student"
__institution__ = "HEC Lausanne"

# Import modules for easy access
from . import config
from . import data_features
from . import preprocessing
from . import models
from . import evaluation
from . import eda_interpretation

__all__ = [
    'config',
    'data_features',
    'preprocessing',
    'models',
    'evaluation',
    'eda_interpretation',
]
