#!/bin/bash
#SBATCH --job-name=stock_pred    # Name of the job
#SBATCH --nodes=1                # Request one node
#SBATCH --ntasks-per-node=1      # Run 1 task
#SBATCH --gres=gpu:a100:4        # Request 4 A100 GPUs
#SBATCH --cpus-per-task=64        # 64 CPU cores per task
#SBATCH --mem=128G                # Request 128GB of memory
#SBATCH --time=6:00:00          # Time limit of 6 hours
#SBATCH --output=logs/%x-%j.out       # Output file: jobname-jobid.out
#SBATCH --error=logs/%x-%j.err        # Error file: jobname-jobid.err
#SBATCH --export=ALL             # Export all environment variables

# Print important environment information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Start time: $(date)"

singularity exec --nv \
    --overlay $SCRATCH/DL_Systems/project/overlay-25GB-500K.ext3:r \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash << 'ENDOFCOMMANDS'

source /ext3/env.sh
conda activate py311

# Set up wandb API key
export WANDB_API_KEY=$(cat $SCRATCH/DL_Systems/project/.wandb_key)

# Create a directory for this run
RUN_DIR="$SCRATCH/DL_Systems/project/stock_pred_runs/tune/$SLURM_JOB_ID"
mkdir -p $RUN_DIR

# Run the training script and save the dashboard port
python -m nscan.training.tune 2>&1 | tee $RUN_DIR/tune.log

ENDOFCOMMANDS

echo "Job finished: $(date)"
