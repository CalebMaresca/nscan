from .data.dataset import NewsReturnDataset

from .data.loading import (
    load_preprocessed_datasets,
    load_returns_and_sp500_data
)

from .data.collation import collate_fn

from .preprocessing.preprocess_data import (
    preprocess_and_save,
    clean_duplicates
)

from .loss import confidence_weighted_loss

__all__ = [
    # From data/dataset.py
    'NewsReturnDataset',

    # From data/loading.py
    'load_preprocessed_datasets',
    'load_returns_and_sp500_data',

    # From data/collation.py
    'collate_fn',
    
    # From preprocess_data.py
    'preprocess_and_save',
    'clean_duplicates',

    # From loss.py
    'confidence_weighted_loss'
]