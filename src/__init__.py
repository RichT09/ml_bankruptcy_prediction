"""
Bankruptcy Prediction ML Package
Master Finance - Advanced Programming
"""

__author__ = "Richard Tschumi"
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
