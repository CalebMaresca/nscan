import os
import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from datetime import datetime
#from typing import Dict, List
import pandas as pd
import numpy as np
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air.integrations.wandb import WandbLoggerCallback
from src.nscan.model import MultiStockPredictorWithConfidence, confidence_weighted_loss


def load_data(years, data_dir):
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

class NewsReturnDataset(torch.utils.data.Dataset):
    def __init__(self, articles_dataset, returns_by_year, sp500_by_year, tokenizer):
        """
        Args:
            articles_dataset: HF dataset with news articles
            returns_by_year: Dict[str, pd.DataFrame] mapping years to returns DataFrames
            sp500_by_year: Dict[str, List[str]] mapping years to SP500 symbols
            tokenizer: Tokenizer for processing articles
        """
        self.articles = articles_dataset
        self.returns_by_year = returns_by_year
        self.sp500_by_year = sp500_by_year
        self.tokenizer = tokenizer

        # Precompute next_dates for all articles
        self.next_dates = {}  # map article idx to next trading date
        for idx in range(len(articles_dataset)):
            article = articles_dataset[idx]
            date = article['Date'].split()[0]
            year = date[:4]
            year_dates = returns_by_year[year].index
            next_date = year_dates[year_dates > date][0]
            self.next_dates[idx] = next_date
        
        # Create master list of all unique stocks across all years
        self.all_stocks = sorted(list(set().union(*sp500_by_year.values())))
        
        # Create mapping from symbol to master index
        self.symbol_to_idx = {symbol: idx+1 for idx, symbol in enumerate(self.all_stocks)} # we start indexing at 1 to use 0 as padding token
        
        # Create year-specific stock indices
        self.year_stock_indices = {}
        for year, symbols in sp500_by_year.items():
            self.year_stock_indices[year] = [self.symbol_to_idx[s] for s in symbols]
        
    def __len__(self):
        return len(self.articles)
        
    def __getitem__(self, idx):
        article = self.articles[idx]
        date = article['Date'].split()[0] # should be YYYY-MM-DD
        year = date[:4]
        
        next_date = self.next_dates[idx]

        # Get indices for this year's SP500 stocks
        stock_indices = self.year_stock_indices[year]
        
        # Get returns for this year's stocks
        returns = self.returns_by_year[year].loc[next_date].values

        # Check for NaN values
        if np.isnan(returns).any():
            print(f"Warning: NaN returns found for date {next_date}")
            # Fill NaN values with 0 or another appropriate value
            returns = np.nan_to_num(returns, 0.0)

        # Tokenize article
        inputs = self.tokenizer(
            article['Article'],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors=None #"pt" ?
        )
        
        return {
            'input_ids': torch.tensor(inputs['input_ids']),
            'attention_mask': torch.tensor(inputs['attention_mask']),
            'stock_indices': torch.tensor(stock_indices),
            'returns': torch.tensor(returns, dtype=torch.float32)
        }

def create_dataset_splits(articles_dataset, returns_by_year, sp500_by_year, tokenizer, 
                         val_start_year, test_start_year):
    """
    Split dataset into train, validation, and test sets based on years.
    
    Args:
        articles_dataset: HF dataset with news articles
        returns_by_year: Dict mapping years to returns DataFrames
        sp500_by_year: Dict mapping years to SP500 symbols
        tokenizer: Tokenizer for processing articles
        val_start_year: Year to start validation set (inclusive)
        test_start_year: Year to start test set (inclusive)
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    def is_valid_article(article):
        # Filter out articles that don't have returns data for the year or next day returns
        date = article['Date'].split()[0]
        year = date[:4]
        
        # First check if we have returns data for this year
        if year not in returns_by_year:
            return False
            
        # Check for next-day returns
        year_dates = returns_by_year[year].index
        next_dates = year_dates[year_dates > date]
        if len(next_dates) == 0:
            return False
            
        return True
    
    # Split articles by year
    train_articles = articles_dataset.filter(
        lambda x: is_valid_article(x) and int(x['Date'][:4]) < val_start_year
    )
    
    val_articles = articles_dataset.filter(
        lambda x: is_valid_article(x) and val_start_year <= int(x['Date'][:4]) < test_start_year
    )
    
    test_articles = articles_dataset.filter(
        lambda x: is_valid_article(x) and int(x['Date'][:4]) >= test_start_year
    )
    
    # Create datasets
    train_dataset = NewsReturnDataset(
        train_articles, returns_by_year, sp500_by_year, tokenizer
    )
    
    val_dataset = NewsReturnDataset(
        val_articles, returns_by_year, sp500_by_year, tokenizer
    )
    
    test_dataset = NewsReturnDataset(
        test_articles, returns_by_year, sp500_by_year, tokenizer
    )
    
    return train_dataset, val_dataset, test_dataset

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

class Trainer:
    def __init__(
        self,
        config,
        train_dataset,
        val_dataset,
        tokenizer,
        checkpoint_dir=None
    ):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize model
        self.model = MultiStockPredictorWithConfidence(
            num_stocks=len(train_dataset.all_stocks),  # Total unique stocks
            num_decoder_layers=config["num_decoder_layers"],
            num_heads=config["num_heads"],
            num_pred_layers=config["num_pred_layers"],
            attn_dropout=config["attn_dropout"],
            ff_dropout=config["ff_dropout"],
            encoder_name=config["encoder_name"]
        ).to(self.device)

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True if self.device == "cuda" else False
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True if self.device == "cuda" else False
        )

        self.tokenizer = tokenizer
        self.max_length = config["max_length"]
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )
        
        # Scheduler and scaler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config["num_epochs"]
        )
        self.scaler = GradScaler()

        self.num_epochs = config["num_epochs"]
        self.validation_freq = config["validation_freq"]

        # Load checkpoint if provided
        if checkpoint_dir:
            checkpoint = torch.load(os.path.join(checkpoint_dir, "checkpoint.pt"))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
    def move_batch(self, batch):
        # Move to device
        return {
            'input_ids': batch['input_ids'].to(self.device),
            'attention_mask': batch['attention_mask'].to(self.device),
            'stock_indices': batch['stock_indices'].to(self.device),
            'returns': batch['returns'].to(self.device)
        }
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            device_batch = self.move_batch(batch)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            with autocast(device_type=self.device):
                predictions, confidences = self.model(
                    input={
                        'input_ids': device_batch['input_ids'],
                        'attention_mask': device_batch['attention_mask']
                    },
                    stock_indices=device_batch['stock_indices']
                )
                
                # Calculate loss
                loss = confidence_weighted_loss(predictions, device_batch['returns'], confidences)
            
            # Backward pass with AMP
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()

            # Validate and report every validation_freq batches
            if (batch_idx + 1) % self.validation_freq == 0:
                current_train_loss = total_loss / (batch_idx + 1)
                val_loss = self.validate()
                
                # Calculate progress within epoch
                progress = (batch_idx + 1) / num_batches
                current_epoch = self.current_epoch + progress
                
                # Report to Ray
                train.report({
                    "train_loss": current_train_loss,
                    "val_loss": val_loss,
                    "epoch": current_epoch
                })
                
                print(f"Epoch {self.current_epoch + 1}, Batch {batch_idx + 1}/{num_batches}")
                print(f"Train Loss: {current_train_loss:.4f}")
                print(f"Val Loss: {val_loss:.4f}")
                print("-" * 30)
            
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                device_batch = self.move_batch(batch)
                
                with autocast(device_type=self.device):
                    predictions, confidences = self.model(
                        input={
                            'input_ids': device_batch['input_ids'],
                            'attention_mask': device_batch['attention_mask']
                        },
                        stock_indices=device_batch['stock_indices']
                    )
                    
                    loss = confidence_weighted_loss(predictions, device_batch['returns'], confidences)
                    total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        best_val_loss = float('inf')
        self.current_epoch = 0
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            start_time = datetime.now()

            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pt')
            

            # Report metrics to Ray
            train.report({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch + 1
            })

            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"Time: {datetime.now() - start_time}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Best Val Loss: {best_val_loss:.4f}")
            print("-" * 50)


def main():
    import ray

    os.environ['HF_HOME'] = '/scratch/ccm7752/huggingface_cache'
    os.environ['TRANSFORMERS_CACHE'] = '/scratch/ccm7752/huggingface_cache'
    os.makedirs('/scratch/ccm7752/huggingface_cache', exist_ok=True)
    os.makedirs('/scratch/ccm7752/dataset_cache', exist_ok=True)

    # Load tokenizer
    encoder_name = "FinText/FinText-Base-2007"
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    # Load data
    years = range(2006, 2023)
    data_dir = "/home/ccm7752/DL_Systems/nscan/data"
    returns_by_year, sp500_by_year = load_data(years, os.path.join(data_dir, "returns"))
    articles_path = os.path.join(data_dir, "raw", "FNSPID-date-corrected.csv")
    articles_dataset = load_dataset(
        "csv", 
        data_files=articles_path, 
        split="train",
        cache_dir='/scratch/ccm7752/dataset_cache'
    )
   
    # Create dataset splits once
    train_dataset, val_dataset, test_dataset = create_dataset_splits(
        articles_dataset, 
        returns_by_year, 
        sp500_by_year,
        tokenizer,
        val_start_year=2022,
        test_start_year=2023
    )
    
    # Get number of GPUs from SLURM environment variable
    num_gpus = torch.cuda.device_count()

    # Initialize Ray with proper resources
    ray.init(
        num_gpus=num_gpus,  # Specify number of GPUs available
        log_to_driver=True,  # Enable logging
    )

    # Define search space
    config = {
        "num_decoder_layers": tune.choice([2, 3, 4]),
        "num_heads": tune.choice([4, 6]),
        "num_pred_layers": tune.choice([2, 3, 4]),
        "attn_dropout": tune.uniform(0.1, 0.3),
        "ff_dropout": tune.uniform(0.1, 0.3),
        "encoder_name": encoder_name,
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "batch_size": tune.choice([64, 128, 256, 512]),
        "max_length": 512,  # Fixed
        "num_epochs": 1,  # Fixed
        "validation_freq": 1000
    }

    # Initialize ASHA scheduler
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=10,  # Max epochs
        grace_period=2,  # Min epochs before pruning
        reduction_factor=2
    )

    # Initialize search algorithm
    search_alg = HyperOptSearch(
        metric="val_loss",
        mode="min"
    )

    # Define train_model function for Ray Tune
    def train_model(config, train_dataset=train_dataset, val_dataset=val_dataset):
        trainer = Trainer(
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            tokenizer=tokenizer
        )
        return trainer.train()

    # Start tuning
    ray_results_dir = os.path.join(os.environ['SCRATCH'], "DL_Systems/project/ray_results")
    analysis = tune.run(
        train_model,
        config=config,
        scheduler=scheduler,
        search_alg=search_alg,
        num_samples=8,  # Total trials
        resources_per_trial={"gpu": 1, "cpu": 4},
        callbacks=[WandbLoggerCallback(
            project="stock-predictor",
            api_key=os.getenv("WANDB_API_KEY"),
            log_config=True
        )],
        storage_path=ray_results_dir,
        name="stock_predictor_tune"
    )

    # Print best config
    print("Best config:", analysis.get_best_config(metric="val_loss", mode="min"))

    ray.shutdown()


if __name__ == "__main__":
    main()
