import torch
from datasets import load_from_disk
import os

class NewsReturnDataset(torch.utils.data.Dataset):
    def __init__(self, preprocessed_dataset):
        """
        Args:
            preprocessed_dataset: HF dataset with preprocessed articles
        """
        self.articles = preprocessed_dataset
        
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
    # Load the preprocessed datasets
    dataset_dict = load_from_disk(data_dir)
    metadata = torch.load(os.path.join(data_dir, 'metadata.pt'))
    
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
    # Load returns data by year
    returns_by_year = {}
    sp500_by_year = {}
    
    for year in years:
        # Load returns DataFrame for this year
        file_path = os.path.join(data_dir, f"{year}_returns.csv")
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