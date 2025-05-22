#!/bin/bash
#SBATCH --job-name=scrna_v3p5                 # appears in `squeue`
#SBATCH --partition=3090-gcondo               # GPU/CPU partition
#SBATCH --exclude=gpu[2607-2609]
#SBATCH --gres=gpu:4                          # 1 GPU per array task
#SBATCH --cpus-per-task=8                     # adjust to match DataLoader workers
#SBATCH --mem=192G                            # or whatever your data need
#SBATCH --time=24:00:00                       # wall‑clock limit
#SBATCH --output=logs/%x_%A_%a.out            # one log per task
#SBATCH --array=0-5%6                         # six tasks, all can run at once
#SBATCH --mail-user=winston_y_li@brown.edu    # <‑‑ change me if needed
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS

# ---------- 1.  Environment ----------
source ../pytorch.venv/bin/activate           # or `module load …`

# ---------- 2.  Pick the script ----------
# The order must match the SBATCH --array indices (0‑5)
FILES=(
  "VICRegExpander_normalized_ctCS_ctmsgG_DO_gRS_gSS.py"
  "VICRegExpander_normalized_ctCS_ctmsgG_DO_gRS.py"
  "VICRegExpander_normalized_ctCS_ctmsgG_DO_gSS.py"
  "VICRegExpander_normalized_ctCS_ctmsgG_gRS_gSS.py"
  "VICRegExpander_normalized_ctCS_DO_gRS_gSS.py"
  "VICRegExpander_normalized_ctmsgG_DO_gRS_gSS.py"
)
SCRIPT=${FILES[$SLURM_ARRAY_TASK_ID]}

echo "[$(date)]  Running ${SCRIPT} on ${SLURM_NODELIST}"
echo "GPU(s):   ${SLURM_GPUS}  |  CPUs: ${SLURM_CPUS_PER_TASK}"

# ---------- 3.  Run the selected file ----------
srun python "${SCRIPT}"
