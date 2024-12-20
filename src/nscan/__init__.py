"""NSCAN: News-Stock Cross-Attention Network"""

from .model import NSCAN, confidence_weighted_loss
from .utils import load_preprocessed_datasets
from . import config # this will create the necessary directories, if they don't already exist

__version__ = "0.1.0"
__all__ = ["NSCAN",
           "load_preprocessed_datasets",
           "confidence_weighted_loss"]