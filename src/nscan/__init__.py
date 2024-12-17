"""NSCAN: News-Stock Cross-Attention Network"""

from .model.model import MultiStockPredictor, MultiStockPredictorWithConfidence
from .utils.data import load_preprocessed_datasets

__version__ = "0.1.0"
__all__ = ["MultiStockPredictor", 
           "MultiStockPredictorWithConfidence",
           "load_preprocessed_datasets"]