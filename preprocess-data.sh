#!/bin/bash

#SBATCH --job-name=preprocess_data
#SBATCH --output=logs/preprocess_data_%j.out
#SBATCH --error=logs/preprocess_data_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0

# Set cache directory for huggingface
export HF_HOME=/scratch/$USER/huggingface_cache

# Create cache directories if they don't exist
mkdir -p $HF_HOME

singularity exec \
    --overlay $SCRATCH/DL_Systems/project/overlay-25GB-500K.ext3:r \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash << 'ENDOFCOMMANDS'

source /ext3/env.sh

conda activate py311

# Run the preprocessing script
python $HOME/DL_Systems/nscan/preprocess_data.py

ENDOFCOMMANDS



