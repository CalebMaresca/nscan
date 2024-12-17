import os
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air.integrations.wandb import WandbLoggerCallback
from nscan.training.trainer import Trainer
from nscan.utils.data import load_preprocessed_datasets


def main():
    import ray

    # For HPC
    wandb_api_key = os.getenv("WANDB_API_KEY")
    data_dir = os.path.join(os.environ['SCRATCH'], "DL_Systems/project/preprocessed_datasets")
    ray_results_dir = os.path.join(os.environ['SCRATCH'], "DL_Systems/project/ray_results")
    num_cpus_per_trial = 5
    os.environ['HF_HOME'] = '/scratch/ccm7752/huggingface_cache'
    os.makedirs('/scratch/ccm7752/huggingface_cache', exist_ok=True)
    os.makedirs('/scratch/ccm7752/dataset_cache', exist_ok=True)

    # For local
    # data_dir = os.path.abspath("data/preprocessed_datasets")
    # ray_results_dir = os.path.abspath("logs/ray_results")
    # os.makedirs(ray_results_dir, exist_ok=True)
    # num_cpus_per_trial = 8

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
        "num_decoder_layers": tune.choice([2, 3, 4]),
        "num_heads": tune.choice([4, 6]),
        "num_pred_layers": tune.choice([2, 3, 4]),
        "attn_dropout": tune.uniform(0.1, 0.3),
        "ff_dropout": tune.uniform(0.1, 0.3),
        "encoder_name": metadata["tokenizer_name"],
        "lr": tune.loguniform(1e-5, 1e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "batch_size": tune.choice([128]),
        "max_length": metadata["max_length"],  # Fixed
        "num_stocks": len(metadata["all_stocks"]),
        "num_epochs": 1,  # Fixed
        "validation_freq": 100
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
        train_dataset = ray.get(train_dataset_ref)
        val_dataset = ray.get(val_dataset_ref)

        trial_params = f"lrs{config['num_decoder_layers']}_heads{config['num_heads']}_predlrs{config['num_pred_layers']}attndrop{config['attn_dropout']:.2e}_ffdrop{config['ff_dropout']:.2e}_lr{config['lr']:.2e}_dec{config['weight_decay']:.2e}_bat{config['batch_size']}"
        checkpoint_dir = os.path.join(os.environ['SCRATCH'], 'DL_Systems/project/checkpoints', 
                                    f'trial_{trial_params}')

        trainer = Trainer(
            config=config,
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
        num_samples=8,  # Total trials
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
