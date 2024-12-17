import os
import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from datetime import datetime
#from typing import Dict, List
from ray import train
from nscan.model import MultiStockPredictorWithConfidence, confidence_weighted_loss
from nscan.utils.data import collate_fn

class Trainer:
    def __init__(
        self,
        config,
        train_dataset,
        val_dataset,
        checkpoint_dir=None,
        load_checkpoint=None
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
        
        self.model = torch.compile(self.model, mode='max-autotune-no-cudagraphs')
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=8,
            pin_memory=True if self.device == "cuda" else False,
            prefetch_factor=8,  # Number of batches loaded in advance by each worker
            persistent_workers=True  # Keep worker processes alive between epochs
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True if self.device == "cuda" else False,
            prefetch_factor=8,  # Number of batches loaded in advance by each worker
            persistent_workers=True  # Keep worker processes alive between epochs
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

        # Create checkpoint directory in scratch
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Load checkpoint if provided
        if load_checkpoint:
            checkpoint = torch.load(os.path.join(load_checkpoint))
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
        best_val_loss = float('inf')
        
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

                # Save model if validation loss improves
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint = {
                        'epoch': self.current_epoch,
                        'batch': batch_idx,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'val_loss': val_loss,
                        'train_loss': current_train_loss
                    }
                    
                    # Save checkpoint with trial ID in filename
                    checkpoint_path = os.path.join(
                        self.checkpoint_dir,
                        f'epoch{self.current_epoch}_batch{batch_idx}.pt'
                    )
                    torch.save(checkpoint, checkpoint_path)
                
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
    
    def validate(self, full_validation=False, num_batches=100):  # num_batches default can be adjusted
        self.model.eval()
        total_loss = 0
        
        if not full_validation:
            # Get total number of batches in val_loader
            total_batches = len(self.val_loader)
            # Randomly select batch indices
            batch_indices = torch.randperm(total_batches)[:num_batches]
            batch_indices = sorted(batch_indices.tolist())  # Sort for efficiency
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Skip batches not in our random selection during partial validation
                if not full_validation and batch_idx not in batch_indices:
                    continue
                    
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
                
                # Break early if we've processed all our selected batches
                if not full_validation and batch_idx > batch_indices[-1]:
                    break
        
        # Divide by actual number of batches processed
        divisor = len(self.val_loader) if full_validation else num_batches
        return total_loss / divisor
    
    def train(self):
        best_val_loss = float('inf')
        self.current_epoch = 0
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            start_time = datetime.now()

            train_loss = self.train_epoch()
            val_loss = self.validate(full_validation=True)
            self.scheduler.step()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss
                }
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f'best_model_epoch{epoch}.pt'
                )
                torch.save(checkpoint, checkpoint_path)
            

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

