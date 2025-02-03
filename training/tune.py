import os
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air.integrations.wandb import WandbLoggerCallback
from trainer import Trainer
from nscan.utils.data import load_preprocessed_datasets
from nscan.config import PREPROCESSED_DATA_DIR, RESULTS_DIR, CACHE_DIR, CHECKPOINT_DIR
from nscan.model.nscan import NSCANConfig

def main():
    """
    Main function for hyperparameter tuning using Ray Tune.
    
    Performs distributed hyperparameter optimization for the stock prediction model using
    the ASHA scheduler and HyperOpt search algorithm. Integrates with Weights & Biases
    for experiment tracking.
    """
    import ray

    # For HPC
    wandb_api_key = os.getenv("WANDB_API_KEY")
    data_dir = PREPROCESSED_DATA_DIR
    ray_results_dir = RESULTS_DIR / "ray_results"
    num_cpus_per_trial = 12
    os.environ['HF_HOME'] = str(CACHE_DIR)

    train_dataset, val_dataset, _, metadata = load_preprocessed_datasets(data_dir)
    
    # Get number of GPUs from SLURM environment variable
    num_gpus = torch.cuda.device_count()

    # Initialize Ray with proper resources
    ray.init(
        num_gpus=num_gpus,  # Specify number of GPUs available
        log_to_driver=True,  # Enable logging
    )

    train_dataset_ref = ray.put(train_dataset)
    val_dataset_ref = ray.put(val_dataset)

    # Define search space
    config = {
        # Model config parameters
        "num_decoder_layers": tune.choice([2, 3, 4]),
        "num_heads": tune.choice([4, 6]),
        "num_pred_layers": tune.choice([2, 3, 4]),
        "attn_dropout": tune.uniform(0.1, 0.3),
        "ff_dropout": tune.uniform(0.1, 0.3),
        "encoder_name": metadata["tokenizer_name"],
        "num_stocks": len(metadata["all_stocks"]),
        "use_flash": False,
        
        # Training config parameters
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "batch_size": tune.choice([128]),
        "num_epochs": 1,  # Fixed
        "validation_freq": 100,
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
    def train_model(config):
        """
        Training function for a single trial in the hyperparameter search.
        
        Args:
            config (dict): Hyperparameter configuration for the trial, including:
                - num_decoder_layers (int): Number of transformer decoder layers
                - num_heads (int): Number of attention heads
                - num_pred_layers (int): Number of prediction layers
                - attn_dropout (float): Attention dropout rate
                - ff_dropout (float): Feed-forward dropout rate
                - encoder_name (str): Name of the pretrained encoder
                - lr (float): Learning rate
                - weight_decay (float): Weight decay coefficient
                - batch_size (int): Training batch size
                - num_stocks (int): Number of stocks in the dataset
                - num_epochs (int): Number of training epochs
                - validation_freq (int): Frequency of validation steps
                - use_flash (bool): Whether to use flash attention
        
        Returns:
            dict: Dictionary containing training metrics including validation loss
        """
        train_dataset = ray.get(train_dataset_ref)
        val_dataset = ray.get(val_dataset_ref)

        trial_params = f"lrs{config['num_decoder_layers']}_heads{config['num_heads']}_predlrs{config['num_pred_layers']}attndrop{config['attn_dropout']:.2e}_ffdrop{config['ff_dropout']:.2e}_lr{config['lr']:.2e}_dec{config['weight_decay']:.2e}_bat{config['batch_size']}"
        checkpoint_dir = CHECKPOINT_DIR / f'trial_{trial_params}'

        model_config = NSCANConfig(
            num_decoder_layers=config["num_decoder_layers"],
            num_heads=config["num_heads"],
            num_pred_layers=config["num_pred_layers"],
            attn_dropout=config["attn_dropout"],
            ff_dropout=config["ff_dropout"],
            encoder_name=config["encoder_name"],
            num_stocks=config["num_stocks"],
            use_flash=config["use_flash"]
        )

        # Create trainer config
        trainer_config = {
            "model_config": model_config,
            "lr": config["lr"],
            "weight_decay": config["weight_decay"],
            "batch_size": config["batch_size"],
            "num_epochs": config["num_epochs"],
            "validation_freq": config["validation_freq"]
        }

        trainer = Trainer(
            config=trainer_config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            checkpoint_dir=checkpoint_dir
        )
        return trainer.train()

    # Start tuning
    analysis = tune.run(
        train_model,
        config=config,
        scheduler=scheduler,
        search_alg=search_alg,
        num_samples=1,  # Total trials
        resources_per_trial={"gpu": 1, "cpu": num_cpus_per_trial},
        callbacks=[WandbLoggerCallback(
            project="stock-predictor",
            api_key=wandb_api_key,
            log_config=True
        )],
        storage_path=ray_results_dir,
        name="stock_predictor_tune",
        trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}"  # for local, default is too long
    )

    # Print best config
    print("Best config:", analysis.get_best_config(metric="val_loss", mode="min"))

    ray.shutdown()


if __name__ == "__main__":
    main()
