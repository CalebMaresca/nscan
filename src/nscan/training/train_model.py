import os
import torch
from nscan.utils.data import load_preprocessed_datasets, create_dataloaders
from nscan.training.trainer import Trainer

def train_single_model(config, data_dir, checkpoint_dir, load_checkpoint=None):
    """Train a single model with given configuration"""
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
        checkpoint_dir=checkpoint_dir,
        load_checkpoint=load_checkpoint
    )
    
    return trainer.train()

def main():
    # Configuration for single model training
    config = {
        "num_decoder_layers": 3,
        "num_heads": 6,
        "num_pred_layers": 3,
        "attn_dropout": 0.2,
        "ff_dropout": 0.2,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "batch_size": 128,
        "num_epochs": 10,
        "validation_freq": 100
    }
    
    # Setup paths
    data_dir = os.path.join(os.environ.get('SCRATCH', 'data'), "DL_Systems/project/preprocessed_datasets")
    checkpoint_dir = os.path.join(os.environ.get('SCRATCH', 'checkpoints'), "DL_Systems/project/single_model")
    
    # Train model
    train_single_model(config, data_dir, checkpoint_dir, load_checkpoint=None)

if __name__ == "__main__":
    main()