import torch

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