#!/bin/bash

for SEED in 1 2 3 4 5; do
    echo "[INFO] Starting seed ${SEED} at $(date)"
    python train_scRNA.py --seed "$SEED"
done