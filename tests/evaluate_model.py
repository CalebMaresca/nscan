import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from datasets import load_from_disk
from nscan.backtesting.backtesting import run_backtest
from nscan.utils.data import NewsReturnDataset, load_returns_and_sp500_data, collate_fn
from nscan.model.model import MultiStockPredictorWithConfidence, confidence_weighted_loss

def move_batch(batch, device):
    # Move to device
    return {
        'input_ids': batch['input_ids'].to(device),
        'attention_mask': batch['attention_mask'].to(device),
        'stock_indices': batch['stock_indices'].to(device),
        'returns': batch['returns'].to(device)
    }

def get_model_predictions(model, test_dataset, device):
    model.eval()
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=8
    )
    
    all_predictions = []
    all_confidences = []
    all_returns = []
    all_dates = []
    all_stock_indices = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in test_loader:
                
            device_batch = move_batch(batch, device)
            
            with autocast(device_type=str(device)):
                predictions, confidences = model(
                    input={
                        'input_ids': device_batch['input_ids'],
                        'attention_mask': device_batch['attention_mask']
                    },
                    stock_indices=device_batch['stock_indices']
                )
                
                loss = confidence_weighted_loss(predictions, device_batch['returns'], confidences)
                total_loss += loss.item()
            
            # Store predictions and metadata
            all_predictions.append(predictions.cpu())  # List of (batch_size, num_stocks_per_date) tensors
            all_confidences.append(confidences.cpu())  # List of (batch_size, num_stocks_per_date) tensors
            all_returns.append(batch['returns'])       # List of (batch_size, num_stocks_per_date) tensors
            all_dates.extend(batch['date'])           # List of dates (strings)
            all_stock_indices.append(batch['stock_indices'])  # List of (batch_size, num_stocks_per_date) tensors
    
    return {
        'test_loss': total_loss / len(test_loader),
        'predictions': torch.cat(all_predictions),     # shape (total_samples, num_stocks_per_date)
        'confidences': torch.cat(all_confidences),     # shape (total_samples, num_stocks_per_date)
        'returns': torch.cat(all_returns),             # shape (total_samples, num_stocks_per_date)
        'dates': all_dates,                            # List of total_samples-many dates (strings)
        'stock_indices': torch.cat(all_stock_indices)  # shape (total_samples, num_stocks_per_date)
    }


def run_model_evaluation(test_years, checkpoint_path, config_path, data_dir):
    """Load model, generate predictions, and evaluate trading strategy performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the preprocessed datasets and metadata
    dataset_dict = load_from_disk(os.path.join(data_dir, "preprocessed_datasets"))  # This loads the DatasetDict
    test_dataset = NewsReturnDataset(dataset_dict['test'], max_articles_per_day=16)  # Get the test split

    # Load checkpoint which contains model state
    checkpoint = torch.load(checkpoint_path)
    
    # Load config
    with open(config_path, 'r') as f:
        best_config = json.load(f)

    # Initialize model with best config
    model = MultiStockPredictorWithConfidence(
        num_stocks=best_config["num_stocks"],
        num_decoder_layers=best_config["num_decoder_layers"],
        num_heads=best_config["num_heads"],
        num_pred_layers=best_config["num_pred_layers"],
        attn_dropout=best_config["attn_dropout"],
        ff_dropout=best_config["ff_dropout"],
        encoder_name=best_config["encoder_name"],
        use_flash=False
    )
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)

    test_results = get_model_predictions(model, test_dataset, device)
    print(f"Test Loss: {test_results['test_loss']:.4f}")

    # Save results
    save_dict = {
        'predictions': test_results['predictions'].numpy(),
        'confidences': test_results['confidences'].numpy(),
        'returns': test_results['returns'].numpy(),
        'dates': test_results['dates'],
        'stock_indices': test_results['stock_indices'].numpy(),
        'test_loss': test_results['test_loss']
    }
    
    save_path = os.path.join(data_dir, "evaluation_results.npz")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, **save_dict)
    print(f"Saved evaluation results to {save_path}")

    # Load returns data
    returns_by_year, sp500_by_year = load_returns_and_sp500_data(test_years, os.path.join(data_dir, "returns"))
    
    # Run backtest
    backtest_results = run_backtest(
        test_results,
        returns_by_year,
        sp500_by_year
    )

if __name__ == "__main__":
    test_years = [2023]
    checkpoint_path = os.path.join(os.environ['SCRATCH'], "DL_Systems/project/checkpoints/best_model/epoch0_batch19999.pt")
    config_path = os.path.join(os.environ['SCRATCH'], "DL_Systems/project/checkpoints/best_model/params.json")
    data_dir = os.path.join(os.environ['SCRATCH'], "DL_Systems/project/data")
    #checkpoint_path = os.path.abspath("checkpoints/best_model/epoch0_batch19999.pt")
    #config_path = os.path.abspath("checkpoints/best_model/params.json")
    #data_dir = os.path.abspath("data")
    
    run_model_evaluation(test_years, checkpoint_path, config_path, data_dir)
