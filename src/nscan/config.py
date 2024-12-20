from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RETURNS_DATA_DIR = DATA_DIR / "returns"
PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed_datasets"
CACHE_DIR = DATA_DIR / "cache"

# Checkpoint directory
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Results and output directories
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
LOGS_DIR = RESULTS_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PREPROCESSED_DATA_DIR, CACHE_DIR,
                 CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)