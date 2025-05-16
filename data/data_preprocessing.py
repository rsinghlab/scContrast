import os, sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
import anndata
import pickle

script_dir = Path(__file__).resolve().parent.parent
print(f'{script_dir=}')
repo_dir = script_dir
src_dir = repo_dir / 'src'
data_dir = repo_dir / 'data'
sys.path.append(str(src_dir))
sys.path.append(str(data_dir))

from src.dataset.dataloader import AnnDataDataset

PARAMETERS = {
    'hvgs': 20116,
    'num_genes': 20116,
    # 'hvgs': 5000,
    # 'num_genes': 5000,
    'latent_dimension': 50,
    'target_sum': 10000,
    'batch_size': 128,
    'num_epochs': 1,
}

# Commented because I already pickled tm_adata_train/test
tm_droplet_data = sc.read(
    # r'./data/raw/tabula_muris/TM_droplet.h5ad',
    data_dir / 'raw' / 'tabula_muris' / 'TM_droplet.h5ad'
    # backup_url="https://figshare.com/ndownloader/files/23938934",
)
tm_facs_data = sc.read(
    # r'./data/raw/tabula_muris/TM_facs.h5ad',
    data_dir / 'raw' / 'tabula_muris' / 'TM_facs.h5ad'
    # backup_url="https://figshare.com/ndownloader/files/23939711",
)
tm_droplet_data_tissues = tm_droplet_data.obs.tissue.tolist()
tm_droplet_data_tissues = {t for t in tm_droplet_data_tissues}
tm_droplet_data_tissues
print(f'{tm_droplet_data_tissues=}')
print(f'{len(tm_droplet_data_tissues)=}')

tm_facs_data_tissues = tm_facs_data.obs.tissue.tolist()
tm_facs_data_tissues = {t for t in tm_facs_data_tissues}
tm_facs_data_tissues
print(f'{tm_facs_data_tissues=}')
print(f'{len(tm_facs_data_tissues)=}')

tm_all_tissues = tm_droplet_data_tissues.union(tm_facs_data_tissues)
# tm_all_tissues
print(f'{len(tm_all_tissues)=}')

# train_tissues = tm_shared_tissues[:-4]
# test_tissues = tm_shared_tissues[-4:]

# print(f'{train_tissues=}')
# print(f'{test_tissues=}')

# train_tissues=['Large_Intestine', 'Spleen', 'Mammary_Gland', 'Lung', 'Kidney', 'Thymus', 'Bladder', 'Tongue', 'Marrow', 'Trachea']
test_tissues={'Skin', 'Liver', 'Limb_Muscle', 'Pancreas'}
train_tissues = tm_all_tissues.difference(test_tissues) # v3,5
print(train_tissues)
print(test_tissues)
tm_droplet_data = tm_droplet_data[
    (~tm_droplet_data.obs.cell_ontology_class.isna())
].copy()
tm_facs_data = tm_facs_data[
    (~tm_facs_data.obs.cell_ontology_class.isna())
].copy()
gene_len = pd.read_csv(
    "https://raw.githubusercontent.com/chenlingantelope/HarmonizationSCANVI/master/data/gene_len.txt",
    delimiter=" ",
    header=None,
    index_col=0,
)
gene_len.head()
import numpy as np
from scipy import sparse

gene_len = gene_len.reindex(tm_facs_data.var.index).dropna()

tm_facs_data = tm_facs_data[:, gene_len.index].copy()   # break the view

gene_len_vec = gene_len[1].values.astype(np.float32)
median_len  = np.median(gene_len_vec)

# column‑wise scaling in CSC format
X = tm_facs_data.X.tocsc(copy=True)        # -> (n_cells × n_genes)
X = X.multiply(1.0 / gene_len_vec)         # divide each column by its length
X = X.multiply(median_len)                 # multiply by the median length
X.data = np.rint(X.data)                   # round only the non‑zero entries

tm_facs_data.X = X.tocsr()                 # store back as CSR (Scanpy’s default)

tm_droplet_train = tm_droplet_data[
    (tm_droplet_data.obs['tissue'].isin(train_tissues))  
    & (~tm_droplet_data.obs.cell_ontology_class.isna())
].copy()

tm_facs_train = tm_facs_data[
    (tm_facs_data.obs['tissue'].isin(train_tissues))  
    & (~tm_facs_data.obs.cell_ontology_class.isna())
].copy()

tm_droplet_train.obs["tech"] = "10x"
tm_facs_train.obs["tech"] = "SS2"
tm_adata_train = tm_droplet_train.concatenate(tm_facs_train)
tm_droplet_test = tm_droplet_data[
    (tm_droplet_data.obs['tissue'].isin(test_tissues))  
    & (~tm_droplet_data.obs.cell_ontology_class.isna())
].copy()

tm_facs_test = tm_facs_data[
    (tm_facs_data.obs['tissue'].isin(test_tissues))  
    & (~tm_facs_data.obs.cell_ontology_class.isna())
].copy()

tm_droplet_test.obs["tech"] = "10x"
tm_facs_test.obs["tech"] = "SS2"
tm_adata_test = tm_droplet_test.concatenate(tm_facs_test)
print(f'{len(tm_adata_train)=}')
print(f'{len(tm_adata_test)=}')
sc.pp.normalize_total(tm_adata_train, target_sum=1e4)
sc.pp.log1p(tm_adata_train)
sc.pp.highly_variable_genes(
    tm_adata_train,
    batch_key="tech",
)

tm_adata_train.X = np.nan_to_num(tm_adata_train.X, nan=0)

num_genes = len(tm_adata_train.var.index)
PARAMETERS['hvgs'] = num_genes

hvg_genes = tm_adata_train.var.index[tm_adata_train.var['highly_variable']].tolist()

# tm_adata_train = tm_adata_train[:, tm_adata_train.var.index.isin(hvg_genes)]
sc.pp.normalize_total(tm_adata_test, target_sum=1e4)
sc.pp.log1p(tm_adata_test)

tm_adata_test.X = np.nan_to_num(tm_adata_test.X, nan=0)

# tm_adata_test = tm_adata_test[:, tm_adata_test.var.index.isin(hvg_genes)]
tm_adata_train.obs.rename(columns={'cell_ontology_class': 'Celltype'}, inplace=True)
tm_adata_test.obs.rename(columns={'cell_ontology_class': 'Celltype'}, inplace=True)
tm_adata_test
celltype_techs = tm_adata_train.obs.groupby("Celltype")["tech"].unique()

# 2) Build a dictionary mapping each cell type to "only_10x", "only_SS2", or "both"
celltype_status = {}
for celltype, tech_list in celltype_techs.items():
    tech_set = set(tech_list)
    if len(tech_set) == 1:
        if "10x" in tech_set:
            celltype_status[celltype] = "only_10x"
        else:
            celltype_status[celltype] = "only_SS2"
    else:
        celltype_status[celltype] = "both"

# 3) Create a new column in .obs indicating whether a cell's type is only_10x, only_SS2, or both
tm_adata_train.obs["celltype_tech_availability"] = (
    tm_adata_train.obs["Celltype"].map(celltype_status)
)
celltype_techs = tm_adata_test.obs.groupby("Celltype")["tech"].unique()

# 2) Build a dictionary mapping each cell type to "only_10x", "only_SS2", or "both"
celltype_status = {}
for celltype, tech_list in celltype_techs.items():
    tech_set = set(tech_list)
    if len(tech_set) == 1:
        if "10x" in tech_set:
            celltype_status[celltype] = "only_10x"
        else:
            celltype_status[celltype] = "only_SS2"
    else:
        celltype_status[celltype] = "both"

# 3) Create a new column in .obs indicating whether a cell's type is only_10x, only_SS2, or both
tm_adata_test.obs["celltype_tech_availability"] = (
    tm_adata_test.obs["Celltype"].map(celltype_status)
)
tm_adata_test.obs['Celltype'].replace(
    to_replace='pancreatic ductal cel',
    value='pancreatic ductal cell',
    inplace=True
)
tm_adata_test.obs['celltype_tech_availability']

print('pickling tm_adata')
pickled_dir = data_dir / 'pickled' / 'tabula_muris'
with open(pickled_dir / 'tm_adata_train_length_normalized_v3,5.pkl', 'wb') as f:
    pickle.dump(tm_adata_train, f)

with open(pickled_dir / 'tm_adata_test_length_normalized_v3,5.pkl', 'wb') as f: # NOTE: v3 test already has both sex test tissues
    pickle.dump(tm_adata_test, f)

print('loading pickled tm_adata_train/test')
pickled_dir = data_dir / 'pickled' / 'tabula_muris'
with open(pickled_dir / 'tm_adata_train_length_normalized_v3,5.pkl', 'rb') as f:
    tm_adata_train = pickle.load( f)

with open(pickled_dir / 'tm_adata_test_length_normalized_v3,5.pkl', 'rb') as f: # NOTE: v3 test already has both sex test tissues
    tm_adata_test = pickle.load( f)


print('beginning tm_dataset')
tm_dataset = AnnDataDataset(tm_adata_train)
print('beginning tm_dataloader')
tm_dataloader = DataLoader(tm_dataset, batch_size=PARAMETERS['batch_size'], shuffle=True)

print('pickling tm_dataset/dataloader')
with open(pickled_dir / 'tm_dataset_train_tissues_length_normalized_v3,5.pkl', 'wb') as f: # NOTE: 3,5 because apparently v3 already has both sexes
    pickle.dump(tm_dataset, f)

with open(pickled_dir / 'tm_dataloader_train_tissues_length_normalized_v3,5.pkl', 'wb') as f:
    pickle.dump(tm_dataloader, f)


# Define functions to precompute data-dependent variables
def precompute_gene_clusters(dataset):
    most_significant_genes_dict = dataset.most_significant_genes_dict
    least_significant_genes_dict = dataset.least_significant_genes_dict
    gene_networks = dataset.gene_networks
    cell_type_categories = dataset.cell_type_categories
    code_to_celltype = dataset.code_to_celltype
    celltype_to_code = dataset.celltype_to_code
    gene_names = dataset.gene_names
    gene_name_to_index = dataset.gene_name_to_index
    index_to_gene_name = dataset.index_to_gene_name
    gene_dispersions = dataset.gene_dispersions
    print('Precomputed gene clusters!')
    return (most_significant_genes_dict, least_significant_genes_dict,
            gene_networks, gene_names, code_to_celltype, celltype_to_code,
            gene_name_to_index, index_to_gene_name, gene_dispersions)

def precompute_mu_sigma(dataloader, most_significant_genes_dict, least_significant_genes_dict, gene_name_to_index):
    all_expression_matrix = []
    cell_types_data = {}
    cell_types_msg_data = {}
    cell_types_lsg_data = {}
    for batch in dataloader:
        expression_matrix, cell_types = batch
        all_expression_matrix.append(expression_matrix)
        
        for cell_type in torch.unique(cell_types):
            cell_type = int(cell_type)
            cell_type_mask = cell_types == cell_type
            cell_type_expression_matrix = expression_matrix[cell_type_mask]
            # All genes
            if cell_type not in cell_types_data:
                cell_types_data[cell_type] = []
            cell_types_data[cell_type].append(cell_type_expression_matrix)

            # Most significant genes
            msg_genes = most_significant_genes_dict[cell_type]
            msg_gene_indices = [gene_name_to_index[g] for g in msg_genes]
            msg_significant_gene_matrix = cell_type_expression_matrix[:, msg_gene_indices]
            if cell_type not in cell_types_msg_data:
                cell_types_msg_data[cell_type] = []
            cell_types_msg_data[cell_type].append(msg_significant_gene_matrix)
            
            # Least significant genes
            lsg_genes = least_significant_genes_dict[cell_type]
            lsg_gene_indices = [gene_name_to_index[g] for g in lsg_genes]
            lsg_significant_gene_matrix = cell_type_expression_matrix[:, lsg_gene_indices]
            if cell_type not in cell_types_lsg_data:
                cell_types_lsg_data[cell_type] = []
            cell_types_lsg_data[cell_type].append(lsg_significant_gene_matrix)

    cell_type_mu_sigma = {}
    cell_type_msg_mu_sigma = {}
    cell_type_lsg_mu_sigma = {}
    # All genes
    for cell_type, cell_type_expression_matrix in cell_types_data.items():
        data_tensor = torch.cat(cell_type_expression_matrix, dim=0)
        mu = torch.mean(data_tensor, dim=0)
        sigma = torch.std(data_tensor, dim=0, unbiased=False)
        sigma = torch.clamp(sigma, min=1e-8)
        cell_type_mu_sigma[int(cell_type)] = (mu, sigma)
    
    # Most significant genes
    for cell_type, matrices in cell_types_msg_data.items():
        data_tensor = torch.cat(matrices, dim=0)
        mu = torch.mean(data_tensor, dim=0)
        sigma = torch.std(data_tensor, dim=0, unbiased=False)
        sigma = torch.clamp(sigma, min=1e-8)
        dispersion = sigma**2 / mu
        cell_type_msg_mu_sigma[int(cell_type)] = (mu, sigma, dispersion)
    
    # Least significant genes
    for cell_type, matrices in cell_types_lsg_data.items():
        data_tensor = torch.cat(matrices, dim=0)
        mu = torch.mean(data_tensor, dim=0)
        sigma = torch.std(data_tensor, dim=0, unbiased=False)
        sigma = torch.clamp(sigma, min=1e-8)
        dispersion = sigma**2 / mu
        cell_type_lsg_mu_sigma[int(cell_type)] = (mu, sigma, dispersion)

    data_tensor = torch.cat(all_expression_matrix, dim=0)
    global_mu_sigma = (torch.mean(data_tensor, dim=0),
                       torch.std(data_tensor, dim=0, unbiased=False))

    return cell_type_mu_sigma, global_mu_sigma, cell_type_msg_mu_sigma, cell_type_lsg_mu_sigma

# Precompute data-dependent variables before model initialization
(most_significant_genes_dict, least_significant_genes_dict,
 gene_networks, gene_names, code_to_celltype, celltype_to_code,
 gene_name_to_index, index_to_gene_name, gene_dispersions) = precompute_gene_clusters(tm_dataset)

cell_type_mu_sigma, global_mu_sigma, cell_type_msg_mu_sigma, cell_type_lsg_mu_sigma = precompute_mu_sigma(
    tm_dataloader, most_significant_genes_dict, least_significant_genes_dict, gene_name_to_index)
precomputed_dir = data_dir / 'pickled' / 'tabula_muris' / 'precomputed'
precomputed_dir.mkdir(parents=True, exist_ok=True)

precomputed_gene_clusters_path =  precomputed_dir / 'tm_dataset_train_tissues_length_normalized_v3,5_precomputed_gene_clusters.pkl'
with open(precomputed_gene_clusters_path, 'wb') as f:
    pickle.dump(
        {
            "most_significant_genes_dict": most_significant_genes_dict,
            "least_significant_genes_dict": least_significant_genes_dict,
            "gene_networks": gene_networks,
            "gene_names": gene_names,
            "code_to_celltype": code_to_celltype,
            "celltype_to_code": celltype_to_code,
            "gene_name_to_index": gene_name_to_index,
            "index_to_gene_name": index_to_gene_name,
            "gene_dispersions": gene_dispersions,
        },
        f,
    )

precomputed_mu_sigma_path = precomputed_dir / 'tm_dataset_train_tissues_length_normalized_v3,5_precomputed_mu_sigma.pkl'
with open(precomputed_mu_sigma_path, "wb") as f:
    pickle.dump(
        {
            "cell_type_mu_sigma": cell_type_mu_sigma,
            "global_mu_sigma": global_mu_sigma,
            "cell_type_msg_mu_sigma": cell_type_msg_mu_sigma,
            "cell_type_lsg_mu_sigma": cell_type_lsg_mu_sigma,
        },
        f,
    )