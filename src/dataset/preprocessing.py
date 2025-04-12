import numpy as np
import scanpy as sc
import anndata as ad


############################################# scRNA-seq preprocessing scripts ###########################################
def scrna_seq_normalization(adata: ad.AnnData, PARAMETERS: dict) -> ad.AnnData:
    sc.pp.filter_genes(adata, min_counts=3)  # Keep genes with at least 3 count
    sc.pp.normalize_total(adata, target_sum=PARAMETERS['target_sum'])  # Normalize total counts per cell
    sc.pp.log1p(adata)  # Log-transform the data
    
    sc.pp.highly_variable_genes(adata, n_top_genes=PARAMETERS['hvgs']) 

    return adata

