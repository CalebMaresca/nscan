import torch

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
            'returns': torch.tensor(article['returns'], dtype=torch.float32)
        }
    
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
        'returns': torch.stack(padded_returns)
    }