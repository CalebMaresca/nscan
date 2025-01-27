import torch

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