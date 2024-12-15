import os
import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from datetime import datetime
#from typing import Dict, List
import pandas as pd
import numpy as np
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air.integrations.wandb import WandbLoggerCallback
from src.nscan.model import MultiStockPredictorWithConfidence, confidence_weighted_loss


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
        'returns': torch.stack(padded_returns)
    }

class Trainer:
    def __init__(
        self,
        config,
        train_dataset,
        val_dataset,
        checkpoint_dir=None
    ):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize model
        self.model = MultiStockPredictorWithConfidence(
            num_stocks=config["num_stocks"],  # Total unique stocks
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

    train_dataset, val_dataset, test_dataset, metadata = load_preprocessed_datasets(
        os.path.join(os.environ['SCRATCH'], "DL_Systems/project/preprocessed_datasets")
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
        "encoder_name": metadata["tokenizer_name"],
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "batch_size": tune.choice([64, 128, 256, 512]),
        "max_length": metadata["max_length"],  # Fixed
        "num_stocks": len(metadata["all_stocks"]),
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
            val_dataset=val_dataset
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
