import numpy as np
from dataset.dataloader import AnnDataDataset
import torch

from collections import OrderedDict
# augmentations_used = OrderedDict()


def global_per_gene_gaussian(expression_matrix: np.ndarray, mu_sigma=None, gene_dispersions=None, sigma_fill=1) -> np.ndarray: # Measurement Noise
    """ Gaussian Noise with mu, std defined across all cells

    Args:
        dataset (AnnDataDataset): dataset.X = expression_matrix, dataset.y = categorical cell_types

    Returns:
        np.ndarray: noised expression_matrix
    """    
    # add_augmentation('gG')
    # Using batch-computed
    # mean = torch.mean(expression_matrix, dim=0)
    # sigma = torch.std(expression_matrix, dim=0)
    
    # # Using pre-computed
    # mu, sigma = mu_sigma

    # Using hard-coded
    # mu = torch.zeros_like(expression_matrix).to(expression_matrix.device)
    sigma = torch.full_like(expression_matrix, sigma_fill).to(expression_matrix.device)

    # # # Using gene_dispersions
    mu = torch.zeros_like(expression_matrix).to(expression_matrix.device)

    if gene_dispersions is not None:
        sigma = torch.tensor(gene_dispersions, dtype=torch.float32, device=expression_matrix.device)
        sigma = torch.clamp(sigma, min=1e-8)
        sigma = sigma.unsqueeze(0).expand_as(expression_matrix)

    # Generate noise based on mean and sigma
    # noise = torch.normal(mu, sigma.expand(expression_matrix.shape)).to(expression_matrix.device)
    noise = torch.normal(mu, sigma)

    # Add noise to the expression matrix
    noised = expression_matrix + noise

    noised = torch.clamp(noised, min=0, max=None)
    return noised

def cell_type_specific_per_gene_gaussian(expression_matrix: torch.Tensor, cell_types: torch.Tensor, mu_sigma_dict) -> torch.Tensor:
    """Add Gaussian noise with mean and std defined per unique cell type.

    Args:
        expression_matrix (torch.Tensor): The expression matrix tensor.
        cell_types (torch.Tensor): The tensor containing categorical cell type labels.

    Returns:
        torch.Tensor: The noised expression matrix.
    """

    # add_augmentation('ctG')
    # Copy expression matrix to avoid modifying the input tensor
    expression_matrix = expression_matrix.clone()

    for cell_type in torch.unique(cell_types):
        cell_type_mask = cell_types == cell_type

        # print(f'{cell_type=}')
        
        # # using pre-computed
        # mu, sigma = mu_sigma_dict[int(cell_type)]

        # Using hard-coded
        mu = torch.zeros_like(expression_matrix[cell_type_mask]).to(expression_matrix.device)
        sigma = torch.full_like(expression_matrix[cell_type_mask], 1e-3).to(expression_matrix.device)

        # Generate noise with the computed mu and sigma
        noise = torch.normal(mu, sigma)
        expression_matrix[cell_type_mask] += noise

    # Clip values to ensure no negative expressions
    return torch.clamp(expression_matrix, min=0)

def dropout_augmentation(expression_matrix: torch.Tensor, dropout_rate: float = 0.1) -> torch.Tensor:
    """Randomly dropout nonzero values at given dropout rate.

    Args:
        expression_matrix (torch.Tensor): The expression matrix tensor.
        dropout_rate (float, optional): Proportion of values to set to zero. Defaults to 0.1.

    Returns:
        torch.Tensor: Expression matrix with certain values zeroed out.
    """
    # add_augmentation(f'DO({dropout_rate})')

    # Copy the expression matrix to avoid modifying the original tensor
    expression_matrix = expression_matrix.clone()

    nonzero_indices = torch.nonzero(expression_matrix, as_tuple=False)
    num_to_drop = int(dropout_rate * len(nonzero_indices))
    drop_indices = nonzero_indices[torch.randperm(len(nonzero_indices))[:num_to_drop]]

    expression_matrix[drop_indices[:, 0], drop_indices[:, 1]] = 0

    return expression_matrix


def cell_type_specific_per_gene_shuffle(expression_matrix: torch.Tensor, cell_types: torch.Tensor) -> torch.Tensor:
    """Permutes gene expression values independently for each cell type.

    Args:
        expression_matrix (torch.Tensor): The expression matrix tensor.
        cell_types (torch.Tensor): The tensor containing cell type labels.

    Returns:
        torch.Tensor: The per-cell-type shuffled expression matrix.
    """

    # add_augmentation('ctShuffle')

    expression_matrix = expression_matrix.clone()

    for cell_type in torch.unique(cell_types):
        cell_type_mask = cell_types == cell_type
        cell_type_expression_matrix = expression_matrix[cell_type_mask]

        batch_size, num_genes = cell_type_expression_matrix.shape

        shuffled_indices = torch.argsort(torch.rand(batch_size, num_genes, device=cell_type_expression_matrix.device), dim=0)

        shuffled = torch.gather(cell_type_expression_matrix, dim=0, index=shuffled_indices)

        expression_matrix[cell_type_mask] = shuffled

    return expression_matrix

def global_per_gene_shuffle(expression_matrix: torch.Tensor) -> torch.Tensor:
    """Shuffles gene expression values globally for each gene without an explicit for-loop.

    Args:
        expression_matrix (torch.Tensor): The expression matrix tensor.

    Returns:
        torch.Tensor: Shuffled expression matrix.
    """

    # add_augmentation('gShuffle')

    expression_matrix = expression_matrix.clone()

    random_order = torch.rand(expression_matrix.size(), device=expression_matrix.device)
    sorted_indices = torch.argsort(random_order, dim=0)

    shuffled = torch.gather(expression_matrix, dim=0, index=sorted_indices)

    return shuffled

def global_gene_subsample(expression_matrix: torch.Tensor, dropout_rate: float = 0.1) -> torch.Tensor:
    """Randomly subsample expression matrix by zeroing out a proportion of gene columns.

    Args:
        expression_matrix (torch.Tensor): The expression matrix tensor.
        dropout_rate (float, optional): Proportion of gene columns to zero-out. Defaults to 0.1.

    Returns:
        torch.Tensor: Gene-subsampled expression matrix.
    """

    # add_augmentation(f'gSS({dropout_rate})')

    expression_matrix = expression_matrix.clone()

    num_genes = expression_matrix.size(1)
    num_to_drop = int(dropout_rate * num_genes)

    genes_to_drop = torch.randperm(num_genes)[:num_to_drop]
    expression_matrix[:, genes_to_drop] = 0

    return expression_matrix

def cell_subsample(expression_matrix: torch.Tensor, dropout_rate: float = 0.1) -> torch.Tensor:
    """Randomly subsample expression matrix by zeroing out a proportion of cell rows.

    Args:
        expression_matrix (torch.Tensor): The expression matrix tensor.
        dropout_rate (float, optional): Proportion of cell rows to zero-out. Defaults to 0.1.

    Returns:
        torch.Tensor: Cell-subsampled expression matrix.
    """
    # add_augmentation(f'cSS({dropout_rate})')

    expression_matrix = expression_matrix.clone()

    num_cells = expression_matrix.size(0)
    num_to_drop = int(dropout_rate * num_cells)

    cells_to_drop = torch.randperm(num_cells)[:num_to_drop]
    expression_matrix[cells_to_drop] = 0

    return expression_matrix

def global_random_scaling_augmentation(expression_matrix: torch.Tensor) -> torch.Tensor:
    """Applies random scaling augmentation to the expression matrix per cell.

    Args:
        expression_matrix (torch.Tensor): The expression matrix tensor.

    Returns:
        torch.Tensor: Scaled expression matrix.
    """

    # add_augmentation('gRS')

    scaling_factors = torch.rand(expression_matrix.size(0), 1) * 0.5 + 0.5
    scaling_factors = scaling_factors.to(expression_matrix.device)

    return expression_matrix * scaling_factors

def cell_type_specific_scaling_augmentation(expression_matrix: torch.Tensor, cell_types: torch.Tensor) -> torch.Tensor:
    """Applies random scaling augmentation to the expression matrix per cell type.

    Args:
        expression_matrix (torch.Tensor): The expression matrix tensor.
        cell_types (torch.Tensor): The tensor containing cell type labels.

    Returns:
        torch.Tensor: The scaled expression matrix.
    """

    # add_augmentation('ctRS')

    expression_matrix = expression_matrix.clone()

    for cell_type in torch.unique(cell_types):
        cell_type_mask = cell_types == cell_type

        num_cells = cell_type_mask.sum().item()
        scaling_factors = torch.rand(num_cells, 1) * 0.5 + 0.5
        scaling_factors = scaling_factors.to(expression_matrix.device)

        expression_matrix[cell_type_mask] *= scaling_factors

    return expression_matrix

def per_cell_type_significant_genes_gaussian(expression_matrix, cell_types, gene_dict, gene_name_to_index, mu_sigma_dict=None, sigma_value=0.1):

    # add_augmentation('ctmsgG')

    expression_matrix = expression_matrix.clone()

    for cell_type in torch.unique(cell_types):
        cell_type = int(cell_type)
        cell_type_mask = cell_types == cell_type
        cell_indices = torch.where(cell_type_mask)[0]

        genes = gene_dict[cell_type]
        gene_indices = [gene_name_to_index[g] for g in genes if g in gene_name_to_index]

        if not gene_indices or cell_indices.numel() == 0:
            continue  # Skip if no genes or cells are found

        significant_gene_matrix = expression_matrix[cell_indices][:, gene_indices]

        # # Hard-coded Gaussian Method
        # # Generate zero-mean Gaussian noise
        # mu = torch.zeros_like(significant_gene_matrix)
        # sigma = torch.full_like(significant_gene_matrix, sigma_value)
        
        # # Dispersion method
        mu = torch.zeros_like(significant_gene_matrix).to(significant_gene_matrix.device)
        _, _, dispersion = mu_sigma_dict[cell_type]
        sigma = torch.sqrt(torch.clamp(dispersion, min=1e-8))
        sigma = sigma.unsqueeze(0).expand_as(significant_gene_matrix).to(significant_gene_matrix.device)
        
        # mu_sigma dict method
        # mu = torch.zeros_like(significant_gene_matrix).to(significant_gene_matrix.device)
        # mu, sigma, _ = mu_sigma_dict[cell_type]
        # mu, _, dispersion = mu_sigma_dict[cell_type]
        # mu = mu.unsqueeze(0).expand_as(significant_gene_matrix).to(significant_gene_matrix.device)
        # sigma = torch.sqrt(dispersion.unsqueeze(0).expand_as(significant_gene_matrix).to(significant_gene_matrix.device))
        # print(f'{mu.shape=}')
        # print(f'{sigma.shape=}')

        gaussian_noise = torch.normal(mean=mu, std=sigma)
        noised_gene_matrix = significant_gene_matrix + gaussian_noise

        expression_matrix[cell_indices.unsqueeze(1), gene_indices] = noised_gene_matrix

    # Ensure no negative expression values
    expression_matrix = torch.clamp(expression_matrix, min=0)

    return expression_matrix

def per_cell_type_cell_shuffle(expression_matrix, cell_types):
    expression_matrix = expression_matrix.clone()

    for cell_type in torch.unique(cell_types):
        cell_type_mask = cell_types == cell_type
        cell_type_expression_matrix = expression_matrix[cell_type_mask]

        batch_size, num_genes = cell_type_expression_matrix.shape

        shuffled_indices = torch.randperm(batch_size, device=expression_matrix.device)
        shuffled = cell_type_expression_matrix[shuffled_indices]

        expression_matrix[cell_type_mask] = shuffled

    return expression_matrix

AUGMENTATION_SHORT_NAMES = {
    global_per_gene_gaussian: 'gG',
    cell_type_specific_per_gene_gaussian: 'ctG',
    dropout_augmentation: 'DO',
    cell_type_specific_per_gene_shuffle: 'ctGS',
    global_per_gene_shuffle: 'gGS', 
    global_gene_subsample: 'gSS', 
    cell_subsample: 'cSS', 
    global_random_scaling_augmentation: 'gRS', 
    cell_type_specific_scaling_augmentation: 'ctRS', 
    per_cell_type_significant_genes_gaussian: 'ctmsgG', 
    per_cell_type_cell_shuffle: 'ctCS'
    
}