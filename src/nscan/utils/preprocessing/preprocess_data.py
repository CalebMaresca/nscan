import os
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
import torch
from transformers import AutoTokenizer
import gc
import shutil
from nscan.utils import load_returns_and_sp500_data
from nscan.config import RETURNS_DATA_DIR, RAW_DATA_DIR, PREPROCESSED_DATA_DIR, CACHE_DIR


def clean_duplicates(file_path):
    """
    Remove duplicate rows from a CSV file and save the cleaned data back to the same file.
    
    Args:
        file_path: Path to the CSV file to clean
        
    Returns:
        None
    """
    print(f"Cleaning {file_path}...")
    
    # Read the data
    df = pd.read_csv(file_path)
    
    # Count rows before
    rows_before = len(df)
    
    # Drop duplicates (keeping first occurrence)
    df = df.drop_duplicates()
    
    # Count rows after
    rows_after = len(df)
    rows_removed = rows_before - rows_after
    
    print(f"Removed {rows_removed} duplicate rows")
    
    # Save cleaned data
    df.to_csv(file_path, index=False)
    print(f"Saved cleaned data to {file_path}")

def create_process_function(returns_by_year, year_stock_indices, tokenizer, max_length=512):
    """
    Create a processing function for batched article data preprocessing.
    
    Args:
        returns_by_year: Dict mapping years to DataFrames containing stock returns
        year_stock_indices: Dict mapping years to lists of stock indices
        tokenizer: HuggingFace tokenizer instance
        max_length: Maximum sequence length for tokenization (default: 512)
        
    Returns:
        function: A processing function that takes a batch of articles and returns processed features
    """
    def process_batch(batch):
        processed_data = {
            'input_ids': [],
            'attention_mask': [],
            'stock_indices': [],
            'returns': [],
            'date': [],
            'next_date': []
        }
        
        valid_indices = []  # Keep track of which articles we want to keep
        for idx, date in enumerate(batch['Date']):
            date = date.split()[0]
            year = str(int(date[:4]))
            
            if year not in returns_by_year:
                continue
                
            # Find next trading date
            returns_df = returns_by_year[year]
            future_returns = returns_df[returns_df.index > date]
            if len(future_returns) == 0:
                continue
                
            next_date = future_returns.index[0]
            
            # Get stock indices and returns
            stock_indices = year_stock_indices[year]
            returns = future_returns.iloc[0].values
            
            if np.isnan(returns).any():
                returns = np.nan_to_num(returns, 0.0)

            processed_data['stock_indices'].append(stock_indices)
            processed_data['returns'].append(returns.tolist())
            processed_data['date'].append(date)
            processed_data['next_date'].append(next_date)
            valid_indices.append(idx)  # Keep track of valid articles
            
        # Batch tokenization
        if valid_indices:
            valid_articles = [batch['Article'][i] for i in valid_indices]
            tokenized = tokenizer(
                valid_articles,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors=None
            )
            processed_data['input_ids'] = tokenized['input_ids']
            processed_data['attention_mask'] = tokenized['attention_mask']
            
        return processed_data

    return process_batch

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
    """
    Preprocess article dataset and save train/validation/test splits.
    
    Args:
        articles_dataset: Dataset containing articles to process
        returns_by_year: Dict mapping years to DataFrames containing stock returns
        sp500_by_year: Dict mapping years to lists of S&P 500 symbols
        tokenizer: HuggingFace tokenizer instance
        val_start_year: Year to start validation split
        test_start_year: Year to start test split
        save_dir: Directory to save processed datasets
        max_length: Maximum sequence length for tokenization (default: 512)
        
    Returns:
        None
    """
    print("Starting preprocessing...", flush=True)
    
    # Create master list of all unique stocks
    all_stocks = sorted(list(set().union(*sp500_by_year.values())))
    symbol_to_idx = {symbol: idx+1 for idx, symbol in enumerate(all_stocks)}
    
    # Create year-specific stock indices
    year_stock_indices = {}
    for year, symbols in sp500_by_year.items():
        year_stock_indices[year] = [symbol_to_idx[s] for s in symbols]

    # Create processing function
    process_batch = create_process_function(
        returns_by_year,
        year_stock_indices,
        tokenizer,
        max_length
    )

    print("Processing articles...", flush=True)
    processed_dataset = articles_dataset.map(
        process_batch,
        batched=True,
        batch_size=100,  # Adjust based on memory
        num_proc=4,      # Match your CPU cores
        remove_columns=articles_dataset.column_names,
        desc="Processing articles"
    )

    print("Splitting dataset...", flush=True)

    def get_year(example):
        return int(example['date'][:4])

    train_dataset = processed_dataset.filter(
        lambda x: get_year(x) < val_start_year,
        num_proc=4
    )
    val_dataset = processed_dataset.filter(
        lambda x: val_start_year <= get_year(x) < test_start_year,
        num_proc=4
    )
    test_dataset = processed_dataset.filter(
        lambda x: get_year(x) >= test_start_year,
        num_proc=4
    )

    # Create and save datasets
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    print("Saving datasets...", flush=True)
    dataset_dict.save_to_disk(save_dir)
    
    # Save metadata
    metadata = {
        'all_stocks': all_stocks,
        'symbol_to_idx': symbol_to_idx,
        'max_length': max_length,
        'tokenizer_name': tokenizer.name_or_path
    }
    torch.save(metadata, save_dir / 'metadata.pt')
    
    print(f"Saved preprocessed datasets to {save_dir}")
    print(f"Split sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

if __name__ == "__main__":
    years = range(2006, 2024)

    print("Script starting", flush=True)

    print("Cleaning duplicates...", flush=True)
    for year in years:
        clean_duplicates(RETURNS_DATA_DIR / f"{year}_returns.csv")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("FinText/FinText-Base-2007")
    print("Tokenizer initialized successfully!", flush=True)

    # Load data
    print("Loading data...", flush=True)
    returns_by_year, sp500_by_year = load_returns_and_sp500_data(years, RETURNS_DATA_DIR)
    
    articles_path = RAW_DATA_DIR / "FNSPID-date-corrected.csv"
    articles_dataset = load_dataset(
        "csv", 
        data_files=articles_path, 
        split="train",
        cache_dir=CACHE_DIR
    )
    print("Data loaded! About to preprocess and save", flush=True)

    # Create save directory
    save_dir = PREPROCESSED_DATA_DIR
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
