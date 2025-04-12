import torch

from torch.utils.data import Dataset, DataLoader
import gseapy
import scanpy as sc
    
class AnnDataDataset(Dataset):
    def __init__(self, adata, filter_significant_genes=False, pval_threshold=0.05, logfc_threshold=1.0):
        self.adata = adata
        self.X = adata.X  # Expression data
        # self.X = torch.tensor(adata.X.toarray(), dtype=torch.float32)  # Expression data
        self.y = adata.obs['Celltype'].astype('category').cat.codes  # Encode cell types as integer labels
        
        self.cell_type_categories = adata.obs['Celltype'].astype('category').cat.categories
        self.gene_dispersions = adata.var['dispersions'].values
        
        self.code_to_celltype = {code: cell_type for code, cell_type in enumerate(self.cell_type_categories)}
        self.celltype_to_code = {cell_type: code for code, cell_type in enumerate(self.cell_type_categories)}

        self.most_significant_genes_dict = self._compute_significant_genes(
            pval_threshold, logfc_threshold
        )
        self.least_significant_genes_dict = self._compute_least_significant_genes(
            pval_threshold, logfc_threshold
        )

        self.gene_names = list(adata.var.index)
        self.gene_networks = self._compute_gene_networks(pval_threshold)

        self.gene_name_to_index = {gene: idx for idx, gene in enumerate(adata.var_names)}
        self.index_to_gene_name = {idx: gene for idx, gene in enumerate(adata.var_names)}
        
        if filter_significant_genes:
            self._filter_significant_genes()

    def _compute_significant_genes(self, pval_threshold, logfc_threshold):
        sc.tl.rank_genes_groups(self.adata, 'Celltype', method='wilcoxon')
        cell_type_gene_dict = {}

        for cell_type in self.cell_type_categories:
            # Extract the DataFrame for this group
            genes_df = sc.get.rank_genes_groups_df(self.adata, group=cell_type)

            # Filter significant genes
            significant_genes = genes_df[
                (genes_df['pvals_adj'] < pval_threshold) &
                (genes_df['logfoldchanges'] > logfc_threshold)
            ]

            # Map significant genes to the integer code for this cell type
            cell_type_code = self.celltype_to_code[cell_type]
            cell_type_gene_dict[cell_type_code] = significant_genes['names'].tolist()

        return cell_type_gene_dict
    
    def _compute_least_significant_genes(self, pval_threshold=0.05, logfc_threshold=1.0):
        sc.tl.rank_genes_groups(self.adata, 'Celltype', method='wilcoxon')
        cell_type_gene_dict = {}

        for cell_type in self.cell_type_categories:
            # Extract the DataFrame for this group
            genes_df = sc.get.rank_genes_groups_df(self.adata, group=cell_type)

            # Filter least significant genes
            # Select genes with high p-values and low absolute log-fold changes
            least_significant_genes = genes_df[
                (genes_df['pvals_adj'] > pval_threshold) &
                (genes_df['logfoldchanges'].abs() < logfc_threshold)
            ]

            # Map least significant genes to the integer code for this cell type
            cell_type_code = self.celltype_to_code[cell_type]
            cell_type_gene_dict[cell_type_code] = least_significant_genes['names'].tolist()

        return cell_type_gene_dict


    def _filter_significant_genes(self):
        significant_genes = set(
            gene for genes in self.gene_dict.values() for gene in genes
        )
        gene_indices = [
            self.adata.var_names.get_loc(gene) for gene in significant_genes
            if gene in self.adata.var_names
        ]
        
        # Filter X to only include significant genes
        self.X = self.X[:, gene_indices]

    def _compute_gene_networks(self, pval_threshold):
        gmt_file_path = r'./data/ontologies/m2.cp.reactome.v2024.1.Mm.symbols.gmt' # TODO: put outside
        # gmt_file_path = r'./data/ontologies/mh.all.v2024.1.Mm.symbols.gmt'

        enr = gseapy.enrichr(gene_list=self.gene_names,
                            gene_sets=gmt_file_path,
                            organism='Mouse',
                            outdir=None)
        
        enr_results = enr.results
        gene_networks = []
        for i in range(len(enr_results)):
            row = enr_results.iloc[i]
            genes = row['Genes'].split(';')

            gene_set_intersection = list(set(self.gene_names) & set(genes))
            gene_networks.append(gene_set_intersection)
        
        return gene_networks


    def __len__(self):
        return self.X.shape[0]  # Number of samples
    
    def __getitem__(self, idx):
        # X_item = torch.tensor(self.X[idx].toarray(), dtype=torch.float32).squeeze()
        # X_item = torch.tensor(self.X[idx], dtype=torch.float32).squeeze()
        X_item = torch.tensor(self.X[idx].toarray(), dtype=torch.float32).squeeze() \
            if hasattr(self.X, 'toarray') else torch.tensor(self.X[idx], dtype=torch.float32).squeeze()
        y_item = torch.tensor(self.y[idx], dtype=torch.long)  # Return integer-encoded cell type
        return X_item, y_item
