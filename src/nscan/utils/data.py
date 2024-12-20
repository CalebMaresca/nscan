import torch
from datasets import load_from_disk
import pandas as pd

class NewsReturnDataset(torch.utils.data.Dataset):
    """A PyTorch Dataset for news articles and their associated stock returns.
    
        This dataset handles the preprocessing of news articles and their corresponding
        stock market returns, with optional limiting of articles per day.
        
        Attributes:
            preprocessed_dataset: HF dataset containing preprocessed news articles
            max_articles_per_day: Maximum number of articles to keep per day
        """
    def __init__(self, preprocessed_dataset, max_articles_per_day=None):
        self.articles = preprocessed_dataset
        if max_articles_per_day is not None:
            # Create a dictionary mapping dates to lists of indices
            date_to_indices = {}
            for i, date in enumerate(self.articles['date']):
                if date not in date_to_indices:
                    date_to_indices[date] = []
                date_to_indices[date].append(i)
            
            # Collect indices of articles to keep
            keep_indices = []
            for date in sorted(date_to_indices.keys()):
                keep_indices.extend(date_to_indices[date][:max_articles_per_day])
            
            # Filter dataset using the collected indices
            self.articles = self.articles.select(sorted(keep_indices))
        
    def __len__(self):
        return len(self.articles)
        
    def __getitem__(self, idx):
        article = self.articles[idx]
        
        return {
            'input_ids': torch.tensor(article['input_ids']),
            'attention_mask': torch.tensor(article['attention_mask']),
            'stock_indices': torch.tensor(article['stock_indices']),
            'returns': torch.tensor(article['returns'], dtype=torch.float32),
            'date': article['date'],
            'next_date': article['next_date']
        }

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
    dataset_dict = load_from_disk(data_dir)
    metadata = torch.load(data_dir / 'metadata.pt')
    
    # Create custom datasets
    train_dataset = NewsReturnDataset(dataset_dict['train'])
    val_dataset = NewsReturnDataset(dataset_dict['validation'])
    test_dataset = NewsReturnDataset(dataset_dict['test'])
    
    return train_dataset, val_dataset, test_dataset, metadata
    
def collate_fn(batch):
    # Filter out None values
    batch = [b for b in batch if b is not None]
    
    # Find maximum number of stocks in this batch
    max_stocks = max(b['stock_indices'].size(0) for b in batch)
    
    # Pad stock indices and returns
    padded_indices = []
    padded_returns = []
    
    for b in batch:
        num_stocks = b['stock_indices'].size(0)
        # Pad stock indices with 0
        padded_idx = torch.nn.functional.pad(
            b['stock_indices'], 
            (0, max_stocks - num_stocks), 
            value=0
        )
        padded_indices.append(padded_idx)
        
        # Pad returns with 0
        padded_ret = torch.nn.functional.pad(
            b['returns'], 
            (0, max_stocks - num_stocks), 
            value=0.0
        )
        padded_returns.append(padded_ret)
    
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'stock_indices': torch.stack(padded_indices),
        'returns': torch.stack(padded_returns),
        'date': [b['date'] for b in batch],
        'next_date': [b['next_date'] for b in batch]
    }

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