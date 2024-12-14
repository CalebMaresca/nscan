import os
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from datetime import datetime
from typing import Dict, List
import pandas as pd
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.integration.wandb import WandbLoggerCallback
from .model import MultiStockPredictorWithConfidence, confidence_weighted_loss


def load_data(years):
    # Load returns data by year
    returns_by_year = {}
    sp500_by_year = {}
    
    for year in years:
        # Load returns DataFrame for this year
        returns_df = pd.read_csv(f"data/returns/{year}_returns.csv")
        returns_df.set_index('Date', inplace=True)
        returns_by_year[year] = returns_df
        
        # Get SP500 symbols for this year
        sp500_by_year[year] = list(returns_df.columns)
    
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
        
        # Create master list of all unique stocks across all years
        all_stocks = set()
        for symbols in sp500_by_year.values():
            all_stocks.update(symbols)
        self.all_stocks = sorted(list(all_stocks))  # Sort for consistent indexing
        
        # Create mapping from symbol to master index
        self.symbol_to_idx = {symbol: idx for idx, symbol in enumerate(self.all_stocks)}
        
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
        next_date = get_next_trading_day(date)
        
        # Skip if no returns data (return None and filter in collate_fn)
        if next_date not in self.returns_by_year[year].index:
            return None

        # Get indices for this year's SP500 stocks
        stock_indices = self.year_stock_indices[year]
        
        # Get returns for this year's stocks
        returns = self.returns_by_year[year].loc[next_date].values

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
    
    # Split articles by year
    train_articles = articles_dataset.filter(
        lambda x: int(x['Date'][:4]) < val_start_year
    )
    
    val_articles = articles_dataset.filter(
        lambda x: val_start_year <= int(x['Date'][:4]) < test_start_year
    )
    
    test_articles = articles_dataset.filter(
        lambda x: int(x['Date'][:4]) >= test_start_year
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
    # Filter out None values (samples we couldn't use)
    batch = [b for b in batch if b is not None]
    
    return {
        'input_ids': torch.stack([b['input_ids'] for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
        'stock_indices': torch.stack([b['stock_indices'] for b in batch]),
        'returns': torch.stack([b['returns'] for b in batch])
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
            with autocast():
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
                
                # Report to Ray Tune
                tune.report(
                    train_loss=current_train_loss,
                    val_loss=val_loss,
                    epoch=current_epoch
                )
                
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
                
                with autocast():
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
    
    def train(self, num_epochs: int):
        best_val_loss = float('inf')
        self.current_epoch = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start_time = datetime.now()

            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pt')
            

            # Log to WandB # Report metrics to Ray Tune
            tune.report(
                train_loss=train_loss,
                val_loss=val_loss,
                epoch=epoch + 1
            )

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Time: {datetime.now() - start_time}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Best Val Loss: {best_val_loss:.4f}")
            print("-" * 50)

def train_model(config, checkpoint_dir=None):    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("FinText/FinText-Base-2007")
    
    # Load dataset
    years = range(2006, 2023)
    returns_by_year, sp500_by_year = load_data(years)
    articles_dataset = load_dataset("csv", data_files="data/raw/FNSPID-date-corrected.csv", split="train")
    train_dataset, val_dataset, test_dataset = create_dataset_splits(
        articles_dataset,
        returns_by_year,
        sp500_by_year,
        tokenizer,
        val_start_year=2022,
        test_start_year=2023
    )
    
    # Create trainer
    trainer = Trainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        tokenizer=tokenizer,
        checkpoint_dir=checkpoint_dir
    )
    
    # Train
    trainer.train()




if __name__ == "__main__":
    import ray
    from ray.tune.utils.util import find_free_port

    # Get number of GPUs from SLURM environment variable
    num_gpus = int(os.environ.get('SLURM_GPUS', 4))  # Default to 4 if not in SLURM

    # Find an available port
    dashboard_port = find_free_port()
    
    # Initialize Ray with proper resources
    ray.init(
        num_gpus=num_gpus,  # Specify number of GPUs available
        log_to_driver=True,  # Enable logging
        dashboard_port=dashboard_port,  # Ray dashboard for monitoring
        local_dir="./ray_results"  # Where to store results
    )

    # Define search space
    config = {
        "num_decoder_layers": tune.choice([2, 3, 4]),
        "num_heads": tune.choice([4, 6]),
        "num_pred_layers": tune.choice([2, 3, 4]),
        "attn_dropout": tune.uniform(0.1, 0.3),
        "ff_dropout": tune.uniform(0.1, 0.3),
        "encoder_name": "FinText/FinText-Base-2007",
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "batch_size": tune.choice([16, 32, 64]),
        "max_length": 512,  # Fixed
        "num_epochs": 10,  # Fixed
        "validation_freq": 1000
    }

    # Initialize ASHA scheduler
    scheduler = ASHAScheduler(
        max_t=10,  # Max epochs
        grace_period=2,  # Min epochs before pruning
        reduction_factor=2
    )

    # Initialize search algorithm
    search_alg = HyperOptSearch(
        metric="val_loss",
        mode="min"
    )

    # Start tuning
    analysis = tune.run(
        train_model,
        config=config,
        scheduler=scheduler,
        search_alg=search_alg,
        num_samples=20,  # Total trials
        resources_per_trial={"gpu": 1, "cpu": 4},
        callbacks=[WandbLoggerCallback(
            project="stock-predictor",
            api_key=os.getenv("WANDB_API_KEY"),
            log_config=True
        )],
        local_dir="./ray_results",
        name="stock_predictor_tune"
    )

    # Print best config
    print("Best config:", analysis.get_best_config(metric="val_loss", mode="min"))

    ray.shutdown()