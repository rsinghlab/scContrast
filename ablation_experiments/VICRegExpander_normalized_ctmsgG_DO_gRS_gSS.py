#!/usr/bin/env python3

import os, sys
from pathlib import Path
import argparse, secrets, random  #  NEW

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
from model.scRNA_E_C import ContrastiveLoss, VICRegLoss, scRNASeqE_VICRegExpander
from augmentations import *

# -------------------------
# 4. Hyperparameters & Setup
# -------------------------
PARAMETERS = {
    "hvgs": 18244,
    "num_genes": 18244,
    "latent_dimension": 128, # Originally 50
    "target_sum": 10000,
    "batch_size": 8192,
    "num_epochs": 50,
}

VERSION = 'v3,5'

num_genes = PARAMETERS["num_genes"]

# seed_everything(42, workers=True)

# -------------------------
# 5. Load Data
# -------------------------
def load_tabula_muris_data():
    """
    Loads your Tabula Muris data from pickled files and returns:
      - tm_dataset
      - tm_dataloader
      - tm_adata_train
      - tm_adata_test
    Adjust paths or file names as needed.
    """
    # Example paths (change to your naming as needed)
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

def get_seed_from_cli() -> int:
    """
    --seed SEED : use the supplied integer
    (no flag)   : draw a random 32-bit seed from /dev/urandom
    """
    parser = argparse.ArgumentParser(
        description="Train scRNA-Seq model; choose or auto-generate a random seed."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed. If omitted, a random 32-bit seed is generated.",
    )
    args = parser.parse_args()
    return args.seed if args.seed is not None else secrets.randbelow(2**32)


# -------------------------
# 9. Main (Run Tuning)
# -------------------------
if __name__ == "__main__":
    # 0) Setup
    SEED = get_seed_from_cli()
    print(f'Using seed: {SEED}')

    PARAMETERS["num_genes"] = num_genes
    torch.set_float32_matmul_precision("medium")
    seed_everything(SEED, workers=True)

    # 1) Load data
    tm_dataset, _, tm_adata_train, tm_adata_test = load_tabula_muris_data()

    # 1.1) Create train/validation dataloaders
    tm_dataset_train, tm_dataset_val = random_split(tm_dataset, [0.8, 0.2])

    # Now create DataLoaders from these two subsets
    train_dataloader = DataLoader(
        tm_dataset_train,
        batch_size=PARAMETERS["batch_size"],
        shuffle=True,  # Typically shuffle the training data
    )
    val_dataloader = DataLoader(
        tm_dataset_val,
        batch_size=PARAMETERS["batch_size"],
        shuffle=False,
        drop_last=False
    )

    precomputed_dir = data_dir / 'pickled' / 'tabula_muris' / 'precomputed'
    precomputed_gene_clusters_path =  precomputed_dir / f'tm_dataset_train_tissues_length_normalized_{VERSION}_precomputed_gene_clusters.pkl'
    precomputed_mu_sigma_path = precomputed_dir / f'tm_dataset_train_tissues_length_normalized_{VERSION}_precomputed_mu_sigma.pkl'

    with open(precomputed_gene_clusters_path, 'rb') as f:
        precomputed_gene_clusters = pickle.load(f)

    with open(precomputed_mu_sigma_path, 'rb') as f:
        precomputed_mu_sigma = pickle.load(f)

    for k, v in precomputed_gene_clusters.items():
        globals()[k] = v

    for k, v in precomputed_mu_sigma.items():
        globals()[k] = v

    print('loaded precomputed!')

    augmentations_pipeline = [
        # {
        #     'fn': per_cell_type_cell_shuffle,
        #     'needs_cell_types': True,
        #     'kwargs': {}
        # },
        # {
        #     'fn': global_per_gene_gaussian,
        #     'needs_cell_types': False,
        #     'kwargs': lambda model: {'sigma_fill': model.sigma_fill, 'gene_dispersions': model.gene_dispersions}
        # }, 
        {
            'fn': per_cell_type_significant_genes_gaussian,
            'needs_cell_types': True,
            'kwargs': lambda model: {
                'gene_dict': model.most_significant_genes_dict, 
                'mu_sigma_dict': model.cell_type_msg_mu_sigma,
                'gene_name_to_index': model.gene_name_to_index, 
                'sigma_value': 1e-1 }
        },
        # {
        #     'fn': global_per_gene_shuffle,
        #     'needs_cell_types': False,
        #     'kwargs': {}
        # },
        {
            'fn': dropout_augmentation,
            'needs_cell_types': False,
            'kwargs': lambda model: {'dropout_rate': model.dropout_rate_DO}
        },
        {
            'fn': global_random_scaling_augmentation,
            'needs_cell_types': False,
            'kwargs': {}
        },
        # {
        #     'fn': cell_type_specific_scaling_augmentation,
        #     'needs_cell_types': True,
        #     'kwargs': {}
        # },
        {
            'fn': global_gene_subsample,
            'needs_cell_types': False,
            'kwargs': lambda model: {'dropout_rate': model.dropout_rate_gSS}
        },
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
        sim_weight=2.5,
        var_weight=1.0,
        cov_weight=0.1,
        std_target=1.0,
        eps=1e-4,
        dropout_rate_DO=0.65,
        dropout_rate_gSS=0.40,
        sigma_fill=0.5,
        augmentations_pipeline=augmentations_pipeline
    )
    
    augmentations_used = final_model.augmentations_used
    experiment_name = f"genes{PARAMETERS['num_genes']}_batch{PARAMETERS['batch_size']}_latent{PARAMETERS['latent_dimension']}_gene_normalized_seed{SEED}"
    augmentations_used_str = '_'.join(augmentations_used)
    print(f'{augmentations_used_str=}')

    best_trial_results_checkpoints_path = f'best_trial_results/{VERSION}/checkpoints/'
    os.makedirs(best_trial_results_checkpoints_path, exist_ok=True)

    early_stop = pl.callbacks.EarlyStopping('val_loss_vicreg', patience=10)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss_vicreg',
        filename=(f'{experiment_name}_'
                  f'{augmentations_used_str}_'
                  'epoch={epoch}-val_loss={val_loss_vicreg:.4f}'),
        dirpath=best_trial_results_checkpoints_path,
        auto_insert_metric_name=False,
    )
    final_trainer = Trainer(
        max_epochs=PARAMETERS["num_epochs"],
        accelerator='gpu',
        devices=-1,
        strategy="ddp",
        precision="bf16-mixed",
        callbacks=[early_stop, checkpoint_callback],
        # callbacks=[checkpoint_callback],
    )
    final_trainer.fit(final_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    best_checkpoint_path = checkpoint_callback.best_model_path
    import re

    match = re.search(r"epoch=(\d+)-val_loss=(\d+\.\d+)", best_checkpoint_path)
    if match:
        saved_epoch = int(match.group(1))
        saved_loss = float(match.group(2))

    # 5) Save best results
    best_trial_results_path = f'best_trial_results/{VERSION}'
    os.makedirs(best_trial_results_path, exist_ok=True)

    final_epoch = final_trainer.current_epoch
    final_loss = final_trainer.callback_metrics['val_loss_vicreg']
    
    # 5) Visualize
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['figure.autolayout'] = False
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['figure.dpi'] = 600

    output_folder = f"figures/{VERSION}/{experiment_name}_{augmentations_used_str}_epoch={saved_epoch}_final-loss={saved_loss:.4f}"
    # output_folder = f"figures/{VERSION}/{experiment_name}_{augmentations_used_str}_epoch={final_epoch}_final-loss={final_loss:.4f}"
    os.makedirs(output_folder, exist_ok=True)
    # os.makedirs(f'{output_folder}/train', exist_ok=True)
    os.makedirs(f'{output_folder}/test', exist_ok=True)

    print(f'Visualizing results; saving into {output_folder}')
    '''
    # Convert to numpy and visualize with UMAP
    with torch.no_grad():
        latent_representations = final_model.encoder(
            torch.tensor(tm_adata_train.X.toarray(), dtype=torch.float32)
            )
    latent_np = latent_representations.detach().cpu().numpy()
    tm_adata_train.obsm['X_latent'] = latent_np  # Store the latents in the obsm dictionary

    tm_adata_train = tm_adata_train.copy() #NOTE: Apparently AnnData==0.10.8 requires this..?

    sc.pp.neighbors(tm_adata_train, use_rep='X_latent')  # Compute neighbors using latent space
    sc.tl.umap(tm_adata_train)  # Run UMAP

    # UMAP train tissues
    fig = sc.pl.umap(
        tm_adata_train, color='Celltype', show=False, return_fig=True, palette=list(mpl.colors.CSS4_COLORS.values())
    )
    fig.savefig(f"{output_folder}/train/celltype.png")
    plt.close(fig)

    fig = sc.pl.umap(
        tm_adata_train, color='method', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/train/method.png")
    plt.close(fig)

    fig = sc.pl.umap(
        tm_adata_train, color='tissue', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/train/tissue.png")
    plt.close(fig)

    fig = sc.pl.umap(
        tm_adata_train, color='celltype_tech_availability', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/train/celltype_tech_availability.png")
    plt.close(fig)
    '''
    with torch.no_grad():
        latent_representations_test = final_model.encoder(
            torch.tensor(tm_adata_test.X.toarray(), dtype=torch.float32)
            )
    latent_np_test = latent_representations_test.detach().numpy()
    tm_adata_test.obsm['X_latent'] = latent_np_test

    tm_adata_test = tm_adata_test.copy()
        
    sc.pp.neighbors(tm_adata_test, use_rep='X_latent')  # Compute neighbors using latent space
    sc.tl.umap(tm_adata_test)  # Run UMAP

    # UMAP test
    fig = sc.pl.umap(
        tm_adata_test, color='Celltype', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/celltype.png")
    plt.close(fig)

    fig = sc.pl.umap(
        tm_adata_test, color='method', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/method.png")
    plt.close(fig)

    fig = sc.pl.umap(
        tm_adata_test, color='tissue', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/tissue.png")
    plt.close(fig)

    fig = sc.pl.umap(
        tm_adata_test, color='celltype_tech_availability', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/celltype_tech_availability.png")
    plt.close(fig)

    # UMAP individual test tissues, 10x only
    fig = sc.pl.umap(
        tm_adata_test[
            (tm_adata_test.obs['tissue'] == 'Skin') &
            (tm_adata_test.obs['tech'] == '10x')].copy(), color='Celltype', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/celltype_Skin_10x.png")
    plt.close(fig)

    fig = sc.pl.umap(
        tm_adata_test[
            (tm_adata_test.obs['tissue'] == 'Liver') &
            (tm_adata_test.obs['tech'] == '10x')].copy(), color='Celltype', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/celltype_Liver_10x.png")
    plt.close(fig)

    fig = sc.pl.umap(
        tm_adata_test[
            (tm_adata_test.obs['tissue'] == 'Limb_Muscle') &
            (tm_adata_test.obs['tech'] == '10x')].copy(), color='Celltype', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/celltype_Limb_Muscle_10x.png")
    plt.close(fig)

    fig = sc.pl.umap(
        tm_adata_test[
            (tm_adata_test.obs['tissue'] == 'Pancreas') &
            (tm_adata_test.obs['tech'] == '10x')].copy(), color='Celltype', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/celltype_Pancreas_10x.png")
    plt.close(fig)

    '''
    # UMAP marrow
    fig = sc.pl.umap(
        tm_adata_train[tm_adata_train.obs['tissue'] == 'Marrow'].copy(), color='Celltype', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/train/celltype_Marrow.png")
    plt.close(fig)

    fig = sc.pl.umap(
        tm_adata_train[tm_adata_train.obs['tissue'] == 'Marrow'].copy(), color='tech', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/train/method_Marrow.png")
    plt.close(fig)
    '''

    # UMAP Liver
    fig = sc.pl.umap(
        tm_adata_test[tm_adata_test.obs['tissue'] == 'Liver'].copy(), color='Celltype', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/celltype_Liver.png")
    plt.close(fig)

    fig = sc.pl.umap(
        tm_adata_test[tm_adata_test.obs['tissue'] == 'Liver'].copy(), color='tech', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/method_Liver.png")
    plt.close(fig)

    # UMAP Skin
    fig = sc.pl.umap(
        tm_adata_test[tm_adata_test.obs['tissue'] == 'Skin'].copy(), color='Celltype', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/celltype_Skin.png")
    plt.close(fig)

    fig = sc.pl.umap(
        tm_adata_test[tm_adata_test.obs['tissue'] == 'Skin'].copy(), color='tech', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/method_Skin.png")
    plt.close(fig)

    # UMAP Limb_Muscle
    fig = sc.pl.umap(
        tm_adata_test[tm_adata_test.obs['tissue'] == 'Limb_Muscle'].copy(), color='Celltype', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/celltype_Limb_Muscle.png")
    plt.close(fig)

    fig = sc.pl.umap(
        tm_adata_test[tm_adata_test.obs['tissue'] == 'Limb_Muscle'].copy(), color='tech', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/method_Limb_Muscle.png")
    plt.close(fig)

    # UMAP Pancreas
    fig = sc.pl.umap(
        tm_adata_test[tm_adata_test.obs['tissue'] == 'Pancreas'].copy(), color='Celltype', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/celltype_Pancreas.png")
    plt.close(fig)

    fig = sc.pl.umap(
        tm_adata_test[tm_adata_test.obs['tissue'] == 'Pancreas'].copy(), color='tech', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/method_Pancreas.png")
    plt.close(fig)

