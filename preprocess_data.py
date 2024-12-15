import os
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
import torch
from transformers import AutoTokenizer

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

def preprocess_and_save(
    articles_dataset,
    returns_by_year,
    sp500_by_year,
    tokenizer,
    val_start_year,
    test_start_year,
    save_dir,
    max_length=512
):
    # Create master list of all unique stocks across all years
    all_stocks = sorted(list(set().union(*sp500_by_year.values())))
    
    # Create mapping from symbol to master index
    symbol_to_idx = {symbol: idx+1 for idx, symbol in enumerate(all_stocks)} # we start indexing at 1 to use 0 as padding token
    
    # Create year-specific stock indices
    year_stock_indices = {}
    for year, symbols in sp500_by_year.items():
        year_stock_indices[year] = [symbol_to_idx[s] for s in symbols]

    trading_dates_by_year = {
        year: sorted(returns_df.index.tolist())
        for year, returns_df in returns_by_year.items()
    }
    
    def find_next_trading_date(date, year):
        """Find the first trading date that comes after the given date"""
        trading_dates = trading_dates_by_year.get(str(year))
        if trading_dates is None:
            return None
        
        for trading_date in trading_dates:
            if trading_date > date:
                return trading_date
        return None
    
    def process_article(article):
        date = article['Date'].split()[0]
        year = str(int(date[:4]))
        
        if year not in returns_by_year:
            return None
            
        next_date = find_next_trading_date(date, year)
        if next_date is None:
            return None

        # Tokenize article
        tokenized = tokenizer(
            article['Article'],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )
        
        # Get stock indices for this year
        stock_indices = year_stock_indices[year]
        
        # Get returns for these stocks
        returns = returns_by_year[year].loc[next_date].values
        
        # Check for NaN values in returns
        if np.isnan(returns).any():
            returns = np.nan_to_num(returns, 0.0)

        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'stock_indices': stock_indices,
            'returns': returns.tolist(),  # Convert numpy array to list for storage
            'date': date,
            'next_date': next_date
        }
    
    train_articles = []
    val_articles = []
    test_articles = []
    
    # Process all articles
    for i, article in enumerate(articles_dataset):
        if i % 1000 == 0:
            print(f"Processing article {i}")
            
        processed = process_article(article)
        if processed is None:
            continue
            
        year = int(processed['date'][:4])
        if year < val_start_year:
            train_articles.append(processed)
        elif year < test_start_year:
            val_articles.append(processed)
        else:
            test_articles.append(processed)
    
    # Create and save datasets
    dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_articles),
        'validation': Dataset.from_list(val_articles),
        'test': Dataset.from_list(test_articles)
    })
    
    # Save the datasets
    dataset_dict.save_to_disk(save_dir)
    
    # Save metadata
    metadata = {
        'all_stocks': all_stocks,
        'symbol_to_idx': symbol_to_idx,
        'max_length': max_length,
        'tokenizer_name': tokenizer.name_or_path
    }
    torch.save(metadata, os.path.join(save_dir, 'metadata.pt'))
    
    print(f"Saved preprocessed datasets to {save_dir}")
    print(f"Split sizes: Train={len(train_articles)}, Val={len(val_articles)}, Test={len(test_articles)}")

if __name__ == "__main__":
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("FinText/FinText-Base-2007")

    # Load data
    years = range(2006, 2023)
    data_dir = "/home/ccm7752/DL_Systems/nscan/data"
    returns_by_year, sp500_by_year = load_returns_and_sp500_data(years, os.path.join(data_dir, "returns"))
    articles_path = os.path.join(data_dir, "raw", "FNSPID-date-corrected.csv")
    articles_dataset = load_dataset(
        "csv", 
        data_files=articles_path, 
        split="train",
        cache_dir='/scratch/ccm7752/dataset_cache'
    )
    
    # Create save directory
    save_dir = os.path.join(os.environ['SCRATCH'], "DL_Systems/project/preprocessed_datasets")
    os.makedirs(save_dir, exist_ok=True)

    preprocess_and_save(
        articles_dataset,
        returns_by_year,
        sp500_by_year,
        tokenizer,
        val_start_year=2022,
        test_start_year=2023,
        save_dir=save_dir
    )