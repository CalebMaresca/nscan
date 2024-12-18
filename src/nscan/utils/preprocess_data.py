import os
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
import torch
from transformers import AutoTokenizer
import gc
import shutil
from nscan.utils import load_returns_and_sp500_data

def clean_duplicates(file_path):
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
    df.to_csv(os.path.join("data/returns", file_path), index=False)
    print(f"Saved cleaned data to {file_path}")

def create_process_function(returns_by_year, year_stock_indices, tokenizer, max_length=512):
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

def preprocess_in_chunks(articles_dataset, process_batch, chunk_size=100000, temp_dir="/tmp/processing"):
    os.makedirs(temp_dir, exist_ok=True)
    total_len = len(articles_dataset)
    chunk_paths = []
    
    print(f"Processing {total_len} articles in chunks of {chunk_size}")
    
    for i in range(0, total_len, chunk_size):
        print(f"\nProcessing chunk {i//chunk_size + 1}/{(total_len+chunk_size-1)//chunk_size}")
        
        # Process chunk
        chunk = articles_dataset.select(range(i, min(i + chunk_size, total_len)))
        processed_chunk = chunk.map(
            process_batch,
            batched=True,
            batch_size=100,
            remove_columns=chunk.column_names,
            num_proc=4,
            desc=f"Processing articles {i}-{min(i + chunk_size, total_len)}"
        )
        
        # Save chunk to disk
        chunk_path = os.path.join(temp_dir, f"chunk_{i}.dataset")
        processed_chunk.save_to_disk(chunk_path)
        chunk_paths.append(chunk_path)
        
        # Clear memory
        del chunk
        del processed_chunk
        gc.collect()
    
    print("\nCombining chunks...")
    # Load and combine all chunks
    all_chunks = [Dataset.load_from_disk(path) for path in chunk_paths]
    final_dataset = concatenate_datasets(all_chunks)
    
    print("Cleaning up temporary files...")
    # Clean up temporary files
    for path in chunk_paths:
        shutil.rmtree(path)
    
    return final_dataset

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
    torch.save(metadata, os.path.join(save_dir, 'metadata.pt'))
    
    print(f"Saved preprocessed datasets to {save_dir}")
    print(f"Split sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

if __name__ == "__main__":
    years = range(2006, 2024)
    #data_dir = "/home/ccm7752/DL_Systems/nscan/data"
    data_dir = "data"

    print("Script starting", flush=True)

    print("Cleaning duplicates...", flush=True)
    for year in years:
        clean_duplicates(os.path.join(data_dir, "returns", f"{year}_returns.csv"))

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("FinText/FinText-Base-2007")
    print("Tokenizer initialized successfully!", flush=True)

    # Load data
    print("Loading data...", flush=True)
    returns_by_year, sp500_by_year = load_returns_and_sp500_data(years, os.path.join(data_dir, "returns"))
    
    articles_path = os.path.join(data_dir, "raw", "FNSPID-date-corrected.csv")
    articles_dataset = load_dataset(
        "csv", 
        data_files=articles_path, 
        split="train",
        cache_dir='/scratch/ccm7752/dataset_cache'
    )
    print("Data loaded! About to preprocess and save", flush=True)

    # Create save directory
    #save_dir = os.path.join(os.environ['SCRATCH'], "DL_Systems/project/preprocessed_datasets")
    save_dir = "data/preprocessed_datasets"
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
