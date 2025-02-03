import torch
from datasets import load_from_disk
import pandas as pd
from .dataset import NewsReturnDataset

def load_preprocessed_datasets(data_dir):
    """Loads preprocessed datasets from disk and creates dataset objects.
    
    Args:
        data_dir (Path): Directory containing the preprocessed datasets
        
    Returns:
        tuple: Contains:
            - train_dataset (NewsReturnDataset): Training dataset
            - val_dataset (NewsReturnDataset): Validation dataset
            - test_dataset (NewsReturnDataset): Test dataset
            - metadata (dict): Dataset metadata
    """
    # Load the preprocessed datasets
    try:
        dataset_dict = load_from_disk(data_dir)
        metadata = torch.load(data_dir / 'metadata.pt')
    except Exception as e:
        print(f"Error loading datasets with Path: {e}. Trying with str().")
        # Fallback to using str() for loading
        dataset_dict = load_from_disk(str(data_dir))
        metadata = torch.load(str(data_dir / 'metadata.pt'))
    
    # Create custom datasets
    train_dataset = NewsReturnDataset(dataset_dict['train'])
    val_dataset = NewsReturnDataset(dataset_dict['validation'])
    test_dataset = NewsReturnDataset(dataset_dict['test'])
    
    return train_dataset, val_dataset, test_dataset, metadata

def load_returns_and_sp500_data(years, data_dir):
    """Loads stock returns and S&P 500 constituent data for specified years.
    
    Args:
        years (list): List of years to load data for
        data_dir (Path): Directory containing the returns data files
        
    Returns:
        tuple: Contains:
            - returns_by_year (dict): Maps years to DataFrames of daily returns
            - sp500_by_year (dict): Maps years to lists of S&P 500 PERMNOs
    """
    # Load returns data by year
    returns_by_year = {}
    sp500_by_year = {}
    
    for year in years:
        # Load returns DataFrame for this year
        file_path = data_dir / f"{year}_returns.csv"
        df = pd.read_csv(file_path)

        # Check raw data before pivot
        print(f"\nYear {year}:")
        print(f"Raw data NaN count: {df['DlyRet'].isna().sum()}")
        
        # Pivot the data to get dates as rows and PERMNOs as columns
        returns_df = df.pivot(
            index='DlyCalDt', 
            columns='PERMNO', 
            values='DlyRet'
        ).sort_index(axis=1)  # Sort columns (PERMNOs)

        # Check pivoted data
        print(f"Pivoted data NaN count: {returns_df.isna().sum().sum()}")
        print(f"Total cells: {returns_df.size}")
        print(f"NaN percentage: {(returns_df.isna().sum().sum() / returns_df.size) * 100:.2f}%")

        # Fill NaN values with 0 (or another appropriate value)
        returns_df = returns_df.fillna(0)
        
        returns_by_year[str(year)] = returns_df
        # Get unique sorted PERMNOs for this year
        sp500_by_year[str(year)] = sorted(df['PERMNO'].unique().tolist())
        
    return returns_by_year, sp500_by_year