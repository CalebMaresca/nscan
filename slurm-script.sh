#!/bin/bash
#SBATCH --job-name=stock_pred    # Name of the job
#SBATCH --nodes=1                # Request one node
#SBATCH --ntasks-per-node=4      # Run 4 tasks (one per GPU)
#SBATCH --gres=gpu:v100:4        # Request 4 V100 GPUs
#SBATCH --cpus-per-task=6        # 6 CPU cores per task
#SBATCH --mem=64G                # Request 64GB of memory
#SBATCH --time=24:00:00          # Time limit of 24 hours
#SBATCH --output=%x-%j.out       # Output file: jobname-jobid.out
#SBATCH --error=%x-%j.err        # Error file: jobname-jobid.err
#SBATCH --export=ALL             # Export all environment variables

# Print important environment information
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $SLURMD_NODENAME"
echo "Start time: $(date)"

singularity exec --nv \
    --overlay overlay-25GB-500K.ext3:rw \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash << 'ENDOFCOMMANDS'

source /ext3/env.sh

conda activate py311

# Set up wandb API key
export WANDB_API_KEY=$(cat ~/.wandb_key)

# Create a directory for this run
RUN_DIR="~/stock_pred_runs/$SLURM_JOB_ID"
mkdir -p $RUN_DIR

# Run the training script and save the dashboard port
python train.py 2>&1 | tee $RUN_DIR/train.log

# The port will be printed in the output, you can find it with:
# grep "Ray dashboard running on port:" $RUN_DIR/train.log

ENDOFCOMMANDS

echo "Job finished: $(date)"