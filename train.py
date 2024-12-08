import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datetime import datetime
from typing import Dict, List
from .model import MultiStockPredictor

class NewsReturnsDataset(Dataset):
    def __init__(self, articles: List[str], dates: List[datetime], returns: Dict[datetime, torch.Tensor]):
        """
        Args:
            articles: List of news articles
            dates: List of datetime objects for each article
            returns: Dict mapping dates to return matrices (date -> returns tensor)
        """
        self.articles = articles
        self.dates = dates
        self.returns = returns
        
    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        article = self.articles[idx]
        date = self.dates[idx]
        
        # Placeholder: Get next trading day's returns
        # TODO: Implement proper trading day calendar logic
        next_trading_day = date  # This should be replaced with proper logic
        target_returns = self.returns[next_trading_day]
        
        return {
            'Article': article,
            'date': date,
            'returns': target_returns
        }

def collate_fn(batch):
    """
    Custom collate function to handle the batch creation
    """
    return {
        'Article': [item['Article'] for item in batch],
        'dates': [item['date'] for item in batch],
        'returns': torch.stack([item['returns'] for item in batch])
    }

class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        tokenizer,
        batch_size: int = 32,  # Now this is number of articles per batch
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 10,
        k_stocks: int = 50,
        max_length: int = 512,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,  # Adjust based on your system
            pin_memory=True if device == "cuda" else False
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True if device == "cuda" else False
        )
        self.tokenizer = tokenizer
        self.device = device
        self.k_stocks = k_stocks
        self.max_length = max_length
        
        # Initialize optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=num_epochs
        )
        
        # AMP scaler
        self.scaler = GradScaler()
        
        # Loss function
        self.loss = nn.MSELoss()
        
    def process_batch(self, batch):
        # Tokenize the articles
        encoded = self.tokenizer(
            batch['Article'],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        return_labels = batch['returns'].to(self.device)  # Shape: (batch_size, num_stocks)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'return_labels': return_labels
        }
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch in self.train_loader:
            processed_batch = self.process_batch(batch)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            with autocast():
                predictions, stock_logits = self.model(
                    {
                        'input_ids': processed_batch['input_ids'],
                        'attention_mask': processed_batch['attention_mask']
                    },
                    k=self.k_stocks
                )
                
                # Get returns for selected stocks only
                top_k_values, top_k_indices = torch.topk(stock_logits, self.k_stocks)
                selected_returns = torch.gather(processed_batch['return_labels'], 1, top_k_indices)
                
                # Calculate loss
                loss = self.loss(predictions, selected_returns)
            
            # Backward pass with AMP
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                processed_batch = self.process_batch(batch)
                
                with autocast():
                    predictions, stock_logits = self.model(
                        {
                            'input_ids': processed_batch['input_ids'],
                            'attention_mask': processed_batch['attention_mask']
                        },
                        k=self.k_stocks
                    )
                    
                    top_k_values, top_k_indices = torch.topk(stock_logits, self.k_stocks)
                    selected_returns = torch.gather(processed_batch['return_labels'], 1, top_k_indices)
                    
                    loss = self.loss(predictions, selected_returns)
                    total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self, num_epochs: int):
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.scheduler.step()
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pt')
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print("-" * 50)

# Example usage:
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from datasets import load_dataset
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("FinText/FinText-Base-2007")
    
    # Load dataset
    dataset = load_dataset("sabareesh88/FNSPID_nasdaq_sorted")
    # Split into train/val and create DataLoaders
    # ... (implement your data loading logic here)
    
    # Initialize model
    model = MultiStockPredictor(
        num_stocks=500, 
        num_decoder_layers=4,
        num_heads=4,
        attn_dropout=0.1,
        ff_dropout=0.1
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,  # Your DataLoader instances
        val_loader=val_loader,
        tokenizer=tokenizer,
        learning_rate=1e-4,
        num_epochs=10,
        k_stocks=3,
        max_length=512
    )
    
    # Train the model
    trainer.train(num_epochs=10)