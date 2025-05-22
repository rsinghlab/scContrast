#!/bin/bash
#SBATCH --job-name=multibatch_jhu            # appears in `squeue`
#SBATCH --partition=3090-gcondo      # change to your GPU/CPU partition
#SBATCH --exclude=gpu[2607-2609]
#SBATCH --gres=gpu:4                     # number / type of GPUs per run
#SBATCH --cpus-per-task=2                # adjust to match DataLoader workers
#SBATCH --mem=192G                        # or whatever your data need
#SBATCH --time=24:00:00                  # wall‑clock limit
#SBATCH --output=logs/%x_%A_%a.out       # one log per seed
#SBATCH --array=1-1                      # <‑‑ five tasks, seeds 1‑5

# ---------- 1.  Environment ----------

# module purge
# module load cuda/12.2                    # or your site’s CUDA module
# source ~/envs/pytorch/bin/activate       # activate the venv/conda env
source ../pytorch.venv/bin/activate

# ---------- 2.  Pick the seed ----------
SEED=${SLURM_ARRAY_TASK_ID}              

# ---------- 3.  Run your code ----------
# srun python VICRegExpander_fixed_loss.py --seed "$SEED"
srun python VICRegExpander_jhu.py --seed "$SEED"