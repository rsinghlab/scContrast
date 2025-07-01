#!/usr/bin/env python3

import os, sys
from pathlib import Path

# -------------------------
# 1. Standard Imports
# -------------------------
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from torch import nn
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
import anndata
import pickle

# For hyperparameter optimization
import optuna

# -------------------------
# 2. Adjust Path Variables
# -------------------------
script_dir = Path(__file__).resolve().parent
repo_dir = script_dir.parent
src_dir = repo_dir / "src"
data_dir = repo_dir / "data"
sys.path.append(str(src_dir))
sys.path.append(str(data_dir))

# -------------------------
# 3. Project-Specific Imports
# -------------------------
from dataset.read_scrnaseq_data import (
    read_cell_by_gene_matrix,
    convert_cell_by_gene_matrix_to_anndata,
    read_hires_metadata_file,
    ensure_same_genes_in_dataframes,
)
from dataset.preprocessing import scrna_seq_normalization
from dataset.dataloader import AnnDataDataset

from model.scRNA_AE import scRNASeqAE
from model.scRNA_E_C_fixed_loss import ContrastiveLoss, VICRegLoss, scRNASeqE_VICRegExpander
from augmentations import *

# -------------------------
# 4. Hyperparameters & Setup
# -------------------------
PARAMETERS = {
    "hvgs": 18244,
    "num_genes": 18244,
    "latent_dimension": 128,
    "target_sum": 10000,
    "batch_size": 8192,
    "num_epochs": 50,
    "n_trials": 1,
    "trial_epochs": 1,
}

VERSION = 'v3,5'

num_genes = PARAMETERS["num_genes"]
seed_everything(42, workers=True)

# -------------------------
# 5. Load Data
# -------------------------
def load_tabula_muris_data():
    tm_dataset_path = data_dir / "pickled" / "tabula_muris" / f"tm_dataset_train_tissues_length_normalized_{VERSION}.pkl"
    tm_dataloader_path = data_dir / "pickled" / "tabula_muris" / f"tm_dataloader_train_tissues_length_normalized_{VERSION}.pkl"
    tm_adata_train_path = data_dir / "pickled" / "tabula_muris" / f"tm_adata_train_length_normalized_{VERSION}.pkl"
    tm_adata_test_path = data_dir / "pickled" / "tabula_muris" / f"tm_adata_test_length_normalized_{VERSION}.pkl"

    with open(tm_dataset_path, "rb") as f:
        tm_dataset = pickle.load(f)
    with open(tm_dataloader_path, "rb") as f:
        tm_dataloader = pickle.load(f)
    with open(tm_adata_train_path, "rb") as f:
        tm_adata_train = pickle.load(f)
    with open(tm_adata_test_path, "rb") as f:
        tm_adata_test = pickle.load(f)

    print("Loaded Tabula Muris data!")
    print(f"Train set: {tm_adata_train.shape}, Test set: {tm_adata_test.shape}")
    return tm_dataset, tm_dataloader, tm_adata_train, tm_adata_test


# -------------------------
# 6. Main
# -------------------------
if __name__ == "__main__":
    PARAMETERS["num_genes"] = num_genes
    torch.set_float32_matmul_precision("medium")
    seed_everything(42, workers=True)

    tm_dataset, _, tm_adata_train, tm_adata_test = load_tabula_muris_data()
    tm_dataset_train, tm_dataset_val = random_split(tm_dataset, [0.8, 0.2])

    train_dataloader = DataLoader(
        tm_dataset_train, batch_size=PARAMETERS["batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(
        tm_dataset_val, batch_size=PARAMETERS["batch_size"], shuffle=False, drop_last=False
    )

    precomputed_dir = data_dir / 'pickled' / 'tabula_muris' / 'precomputed'
    gene_clusters_path = precomputed_dir / f'tm_dataset_train_tissues_length_normalized_{VERSION}_precomputed_gene_clusters.pkl'
    mu_sigma_path = precomputed_dir / f'tm_dataset_train_tissues_length_normalized_{VERSION}_precomputed_mu_sigma.pkl'

    with open(gene_clusters_path, 'rb') as f:
        gene_clusters = pickle.load(f)
    with open(mu_sigma_path, 'rb') as f:
        mu_sigma = pickle.load(f)

    for k, v in gene_clusters.items():
        globals()[k] = v
    for k, v in mu_sigma.items():
        globals()[k] = v

    print('Loaded precomputed!')

    augmentations_pipeline = [
        {'fn': per_cell_type_cell_shuffle, 'needs_cell_types': True, 'kwargs': {}},
        {'fn': per_cell_type_significant_genes_gaussian, 'needs_cell_types': True,
         'kwargs': lambda model: {'gene_dict': model.most_significant_genes_dict,
                                  'mu_sigma_dict': model.cell_type_msg_mu_sigma,
                                  'gene_name_to_index': model.gene_name_to_index,
                                  'sigma_value': 1e-1}},
        {'fn': dropout_augmentation, 'needs_cell_types': False,
         'kwargs': lambda model: {'dropout_rate': model.dropout_rate_DO}},
        {'fn': cell_type_specific_scaling_augmentation, 'needs_cell_types': True, 'kwargs': {}},
        {'fn': global_gene_subsample, 'needs_cell_types': False,
         'kwargs': lambda model: {'dropout_rate': model.dropout_rate_gSS}},
    ]

    final_model = scRNASeqE_VICRegExpander(
        PARAMETERS,
        cell_type_mu_sigma=cell_type_mu_sigma,
        global_mu_sigma=global_mu_sigma,
        cell_type_msg_mu_sigma=cell_type_msg_mu_sigma,
        cell_type_lsg_mu_sigma=cell_type_lsg_mu_sigma,
        most_significant_genes_dict=most_significant_genes_dict,
        least_significant_genes_dict=least_significant_genes_dict,
        gene_networks=gene_networks,
        gene_names=gene_names,
        code_to_celltype=code_to_celltype,
        celltype_to_code=celltype_to_code,
        gene_name_to_index=gene_name_to_index,
        index_to_gene_name=index_to_gene_name,
        gene_dispersions=gene_dispersions,
        sim_weight=25.0,
        var_weight=25.0,
        cov_weight=1,
        std_target=1.0,
        eps=1e-4,
        dropout_rate_DO=0.5,
        dropout_rate_gSS=0.5,
        sigma_fill=0.5,
        augmentations_pipeline=augmentations_pipeline
    )

    augmentations_used = final_model.augmentations_used
    experiment_name = f"genes{PARAMETERS['num_genes']}_batch{PARAMETERS['batch_size']}_latent{PARAMETERS['latent_dimension']}_fixed_loss"
    augmentations_used_str = '_'.join(augmentations_used)
    print(f'{augmentations_used_str=}')

    checkpoint_path = f'best_trial_results/{VERSION}/checkpoints/'
    os.makedirs(checkpoint_path, exist_ok=True)

    early_stop = pl.callbacks.EarlyStopping('val_loss_vicreg', patience=10)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss_vicreg',
        filename=f'{experiment_name}_{augmentations_used_str}_epoch={{epoch}}-val_loss={{val_loss_vicreg:.4f}}',
        dirpath=checkpoint_path,
        auto_insert_metric_name=False,
    )

    trainer = Trainer(
        max_epochs=PARAMETERS["num_epochs"],
        devices=-1,
        strategy="ddp",
        precision="bf16-mixed",
    )
    trainer.fit(final_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    final_epoch = trainer.current_epoch
    final_loss = trainer.callback_metrics['val_loss_vicreg']

    output_folder = f"figures/{VERSION}/{experiment_name}_{augmentations_used_str}_epoch={final_epoch}_final-loss={final_loss:.4f}"
    os.makedirs(f'{output_folder}/test', exist_ok=True)

    with torch.no_grad():
        latent_test = final_model.encoder(torch.tensor(tm_adata_test.X.toarray(), dtype=torch.float32))
    tm_adata_test.obsm['X_latent'] = latent_test.detach().cpu().numpy()

    sc.pp.neighbors(tm_adata_test, use_rep='X_latent')
    sc.tl.umap(tm_adata_test)

    fig = sc.pl.umap(tm_adata_test, color='Celltype', show=False, return_fig=True)
    fig.savefig(f"{output_folder}/test/celltype.png")

    # === Save for SCIB metrics ===
    tm_adata_test.obsm['X_emb'] = tm_adata_test.obsm['X_latent']
    with open("adata_test_for_metrics_1.pkl", "wb") as f:
        pickle.dump(tm_adata_test, f)
    print("Saved AnnData object for SCIB metrics to: adata_test_for_metrics_1.pkl")
