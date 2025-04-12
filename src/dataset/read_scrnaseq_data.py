import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad

def read_cell_by_gene_matrix(path: str) -> pd.DataFrame:
    format = path.split('/')[-1].split('.')[-2]
    
    if format == 'csv':
        sep = ','
    elif format == 'tsv':
        sep = '\t'
    else:
        sep = None
    
    cell_by_gene_data = pd.read_csv(
        path, sep=sep,
        comment='#', index_col=0, header=0
    )
    
    cell_by_gene_data = cell_by_gene_data.T
    
    return cell_by_gene_data


def convert_cell_by_gene_matrix_to_anndata(cell_by_gene_pd: pd.DataFrame, PARAMETERS: dict) -> ad.AnnData:
    # Create AnnData object
    adata = ad.AnnData(cell_by_gene_pd)

    # # Keep genes with at least 3 count
    # sc.pp.filter_genes(adata, min_counts=3)  
    # print('filtered')

    # Normalize the total counts per cell
    sc.pp.normalize_total(adata, target_sum=PARAMETERS['target_sum'])
    
    # Log-transform the data
    sc.pp.log1p(adata)
    
    # Check if HVGs are provided, if not compute HVGs
    sc.pp.highly_variable_genes(adata, n_top_genes=PARAMETERS['hvgs'])
    hvgs = adata.var[adata.var['highly_variable']].index
    
    # adata.raw = adata
    
    # Handle any NaN values in the data
    adata.X = np.nan_to_num(adata.X, nan=0)
    
    return adata, hvgs


def read_hires_metadata_file(path: str) -> pd.DataFrame:
    metadata = pd.read_excel(path, header=0, index_col=0)
    
    if 'Cell type' in metadata.columns: 
        metadata.rename(columns={'Cell type': 'Celltype'}, inplace=True)

    return metadata

def ensure_same_genes_in_dataframes(df1, df2, fill_value=0):
    union_columns = list(set(df1.columns).union(df2.columns))
    
    df1_aligned = df1.reindex(columns=union_columns, fill_value=fill_value)
    df2_aligned = df2.reindex(columns=union_columns, fill_value=fill_value)
    
    return df1_aligned, df2_aligned