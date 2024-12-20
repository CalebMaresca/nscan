"""
NSCAN (News-Stock Cross-Attention Network) Training Script

This module implements the distributed training pipeline for the NSCAN (News-Stock 
Cross-Attention Network) model, which predicts multiple stock returns simultaneously
using financial news data and cross-attention mechanisms.

The training infrastructure leverages Ray Train for distributed training and supports:
- Multi-GPU training across multiple workers
- Integration with Weights & Biases (WandB) for experiment tracking
- Checkpoint management for model saving and loading
- Optimized hyperparameters for the transformer-based architecture
"""

import os
from ray import train
from ray.train.torch import TorchTrainer
from ray.air.integrations.wandb import setup_wandb
from nscan.utils.data import load_preprocessed_datasets
from nscan.training.trainer import Trainer
from nscan.config import PREPROCESSED_DATA_DIR, CHECKPOINT_DIR


def train_func(config):
    """Training function to be executed in each worker.
    
    This function handles the core training logic for each distributed worker, including:
    - Loading and preprocessing datasets
    - Setting up WandB logging
    - Initializing and executing the training process
    
    Args:
        config (dict): Configuration dictionary containing training parameters and settings
                      including data_dir, model architecture params, and training hyperparameters
    
    Returns:
        dict: Training results including metrics and model state
    """
    # Get data_dir from config
    data_dir = config.pop("data_dir")

    # Setup WandB
    wandb_run = setup_wandb(
        config=config,
        api_key=os.getenv("WANDB_API_KEY"),
        project="stock-predictor"
    )

    train_dataset, val_dataset, _, metadata = load_preprocessed_datasets(data_dir)
    
    # Update config with metadata
    config.update({
        "encoder_name": metadata["tokenizer_name"],
        "max_length": metadata["max_length"],
        "num_stocks": len(metadata["all_stocks"])
    })
    
    trainer = Trainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        checkpoint_dir=train.get_context().get_trial_dir(),
        load_checkpoint=config.get("load_checkpoint"),
        wandb=wandb_run
    )
    
    return trainer.train()

def train_model(config, data_dir, checkpoint_dir, load_checkpoint=None):
    """Setup and launch distributed training using Ray Train.
    
    Args:
        config (dict): Model and training configuration parameters
        data_dir (str): Directory containing preprocessed training data
        checkpoint_dir (str): Directory for saving model checkpoints
        load_checkpoint (str, optional): Path to checkpoint to resume training from. Defaults to None.
    
    Returns:
        ray.air.result.Result: Results object containing training metrics and final model state
    """
    # Add data_dir to config so it's available in training function
    config["data_dir"] = data_dir
    config["load_checkpoint"] = load_checkpoint

    trainer = TorchTrainer(
        train_func,
        train_loop_config=config,
        scaling_config=train.ScalingConfig(num_workers=4, use_gpu=True),  # Adjust number of workers as needed
        run_config=train.RunConfig(
            storage_path=checkpoint_dir
        )
    )
    
    results = trainer.fit()
    return results

def main(load_checkpoint=None):
    """Main entry point for model training.
    
    Initializes training configuration with optimal hyperparameters and launches the training process.
    These hyperparameters were determined through previous hyperparameter tuning.
    
    Args:
        load_checkpoint (str, optional): Path to checkpoint to resume training from. Defaults to None.
    """
    config = {
        "num_decoder_layers": 4,
        "num_heads": 4,
        "num_pred_layers": 4,
        "attn_dropout": 0.25,
        "ff_dropout": 0.2,
        "lr": 0.000036,
        "weight_decay": 1e-5,
        "batch_size": 64,
        "num_epochs": 5,
        "validation_freq": 1000,
        "use_flash": False
    }
    
    # Train model
    train_model(config, PREPROCESSED_DATA_DIR, CHECKPOINT_DIR, load_checkpoint=load_checkpoint)

if __name__ == "__main__":
    main()