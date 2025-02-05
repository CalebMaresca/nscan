import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from datasets import load_from_disk
from nscan.backtesting import run_backtest
from nscan.utils import NewsReturnDataset, load_returns_and_sp500_data, collate_fn
from nscan import NSCAN, confidence_weighted_loss
from nscan.config import CHECKPOINT_DIR, DATA_DIR, PREPROCESSED_DATA_DIR

# For use in environments where torch._dynamo is not available
#import torch._dynamo
#torch._dynamo.config.suppress_errors = True

def move_batch(batch, device):
    """Move batch data to specified device.
    
    Args:
        batch (dict): Batch dictionary containing input_ids, attention_mask, 
                     stock_indices, and returns tensors
        device (torch.device): Target device to move tensors to
    
    Returns:
        dict: Batch with all tensors moved to specified device
    """
    return {
        'input_ids': batch['input_ids'].to(device),
        'attention_mask': batch['attention_mask'].to(device),
        'stock_indices': batch['stock_indices'].to(device),
        'returns': batch['returns'].to(device)
    }

def get_model_predictions(model, test_dataset, device):
    """Generate predictions using the model on the test dataset.
    
    Args:
        model (NSCAN): The trained NSCAN model
        test_dataset (NewsReturnDataset): Dataset containing test samples
        device (torch.device): Device to run inference on
        
    Returns:
        dict: Dictionary containing:
            - test_loss (float): Average loss across test batches
            - predictions (torch.Tensor): Model predictions (shape: total_samples x num_stocks_per_date)
            - confidences (torch.Tensor): Prediction confidences (shape: total_samples x num_stocks_per_date)
            - returns (torch.Tensor): Actual returns (shape: total_samples x num_stocks_per_date)
            - dates (list): List of dates as strings
            - stock_indices (torch.Tensor): Stock indices (shape: total_samples x num_stocks_per_date)
    """
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
        for idx, batch in enumerate(test_loader):
            print(f"Inferencing batch {idx} of {len(test_loader)}", flush=True)    
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

def plot_results(test_results, data_dir):
    """Plot results from model evaluation.
    
    Args:
        test_results (dict): Dictionary containing predictions, confidences, returns, dates, and metrics
        data_dir (Path): Directory to save plots
    """
    # 1. Portfolio Value Over Time
    #fig = backtest_results.plot(open=False, high=False, low=False, volume=False)[0][0]
    #fig.savefig(data_dir / 'backtest_plot.png')
    #    plt.close(fig)

    # 2. Plot predicted vs actual returns scatter
    plt.figure(figsize=(10, 10))
    plt.scatter(test_results['returns'].flatten(), test_results['predictions'].flatten(), alpha=0.1)
    plt.xlabel('Actual Returns')
    plt.ylabel('Predicted Returns')
    plt.title('Predicted vs Actual Returns')
    plt.grid(True)
    max_val = max(abs(plt.xlim()[0]), abs(plt.xlim()[1]))
    plt.plot([-max_val, max_val], [-max_val, max_val], 'r--')  # Perfect prediction line
    plt.tight_layout()
    plt.savefig(data_dir / 'predicted_vs_actual.png')
    plt.close()
    
    # 3. Plot prediction error distribution
    errors = test_results['predictions'] - test_results['returns']
    plt.figure(figsize=(10, 6))
    plt.hist(errors.flatten(), bins=50)
    plt.title('Distribution of Prediction Errors')
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(data_dir / 'error_distribution.png')
    plt.close()

def run_model_evaluation(test_years, checkpoint_path, config_path, data_dir):
    """Load model from checkpoint and evaluate its performance on test data.
    
    Generates predictions on test data, calculates metrics, and creates visualization plots.
    Results and plots are saved to the specified data directory.
    
    Args:
        test_years (list): List of years to evaluate on
        checkpoint_path (Path): Path to model checkpoint file
        config_path (Path): Path to model configuration JSON file
        data_dir (Path): Directory to save evaluation results and plots
        
    Saves:
        - evaluation_results.npz: Contains predictions, confidences, returns, dates, and metrics
        - predicted_vs_actual.png: Scatter plot of predicted vs actual returns
        - error_distribution.png: Histogram of prediction errors
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the preprocessed datasets and metadata
    dataset_dict = load_from_disk(PREPROCESSED_DATA_DIR)  # This loads the DatasetDict
    test_dataset = NewsReturnDataset(dataset_dict['test'], max_articles_per_day=4)  # Get the test split

    # Load checkpoint which contains model state
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load config
    with open(config_path, 'r') as f:
        best_config = json.load(f)

    # Initialize model with best config
    model = NSCAN(
        num_stocks=best_config["num_stocks"],
        num_decoder_layers=best_config["num_decoder_layers"],
        num_heads=best_config["num_heads"],
        num_pred_layers=best_config["num_pred_layers"],
        attn_dropout=best_config["attn_dropout"],
        ff_dropout=best_config["ff_dropout"],
        encoder_name=best_config["encoder_name"],
        use_flash=False
    )
    model = torch.compile(model, mode='max-autotune-no-cudagraphs')
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
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
    
    save_path = data_dir / "evaluation_results.npz"
    np.savez(save_path, **save_dict)
    print(f"Saved evaluation results to {save_path}")

    # Load returns data
    returns_by_year, sp500_by_year = load_returns_and_sp500_data(test_years, data_dir / "returns")
    
    # Run backtest
    backtest_results = run_backtest(
        test_results,
        returns_by_year,
        sp500_by_year
    )
    
    plot_results(test_results, data_dir)

if __name__ == "__main__":
    test_years = [2023]
    checkpoint_path = CHECKPOINT_DIR / "best_model/epoch1_batch2999.pt"
    config_path = CHECKPOINT_DIR / "best_model/params.json"
    data_dir = DATA_DIR
    
    run_model_evaluation(test_years, checkpoint_path, config_path, data_dir)
