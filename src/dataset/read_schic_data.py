import gzip

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
from src.dataset.normalizations import ICE_normalization
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap


REDMAP = LinearSegmentedColormap.from_list("bright_red", [(1,1,1),(1,0,0)])


def decompose_hic_contact_map(contact_map: np.ndarray, k: int):
    contact_map = np.array(contact_map)
    contact_map = np.triu(contact_map) + np.triu(contact_map, 1).T
    
    # Ensure symmetry (for undirected graphs, the contact map must be symmetric)
    if not np.allclose(contact_map, contact_map.T):
        print("Warning: Contact map is not symmetric.")
    
    degree_matrix = np.diag(np.sum(contact_map, axis=1))  # Degree matrix D
    
    laplacian_matrix = degree_matrix - contact_map  # Laplacian L = D - M
    
    if not np.allclose(laplacian_matrix, laplacian_matrix.T):
        print("Warning: Laplacian matrix is not symmetric.")
    
    eigenvalues, eigenvectors = np.linalg.eigh(contact_map)
    
    sorted_indices = np.argsort(eigenvalues)   # Indices for sorting in ascending order
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    top_k_eigenvalues = eigenvalues[:k]
    top_k_eigenvectors = eigenvectors[:, :k]
    
    return top_k_eigenvalues, top_k_eigenvectors


def reconstruct_hic_contact_map(eigenvalues, eigenvectors):
    # Construct a diagonal matrix from the eigenvalues
    lambda_matrix = np.diag(eigenvalues)
    
    # Reconstruct the original matrix using V * Lambda * V.T
    reconstructed_matrix = eigenvectors @ lambda_matrix @ eigenvectors.T
    
    return reconstructed_matrix


def library_size_normalization(matrix: np.ndarray, library_size:float = 100000000) -> np.ndarray:
    sum_reads = np.sum(matrix)
    matrix = np.divide(matrix, sum_reads) * library_size
    matrix = np.log1p(matrix)
    matrix = matrix/np.max(matrix)
    
    return matrix


def gaussian_smooth_contact_map(matrix: np.ndarray, sigma:float = 1):
    # Apply Gaussian smoothing
    smoothed = gaussian_filter(matrix, sigma=sigma)
    
    return smoothed


def create_transition_matrix(matrix: np.ndarray):
    matrix = np.array(matrix)
    matrix = matrix + 1e-10
    
    row_sums = matrix.sum(axis=1)
    transition_matrix = matrix / row_sums[:, np.newaxis]
    
    return transition_matrix


def personalized_pagerank(matrix: np.ndarray, alpha:float= 0.85, max_iter:int=100, tol:float=1e-6):
    N = matrix.shape[0]
    
    personalization = np.ones(N) / N
    personalization = personalization / np.sum(personalization)
    
    # Initial PageRank vector
    r = np.ones(N) / N
    
    # Teleport vector
    teleport = personalization.copy()
    
    # Iterative calculation
    for _ in range(max_iter):
        # Compute new PageRank vector
        r_new = alpha * matrix.T.dot(r) + (1 - alpha) * teleport
        
        # Check for convergence
        if np.linalg.norm(r_new - r, 1) < tol:
            return r_new
        
        # Update current vector
        r = r_new
    
    return r


def personalized_pagerank_smooth(matrix: np.ndarray, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    # Local smoothing filter before personalized page rank smoothing
    matrix = gaussian_smooth_contact_map(matrix)
    
    # Set diagonal to zero
    np.fill_diagonal(matrix, 0)
    
    # Create transition matrix
    transition_matrix = create_transition_matrix(matrix)
    
    # Compute PageRank vector
    pagerank_vector = personalized_pagerank(transition_matrix, alpha, max_iter, tol)
    
    # Create smoothed matrix by weighting original matrix with PageRank
    smoothed_matrix = matrix * pagerank_vector[:, np.newaxis]
    
    return smoothed_matrix


def read_pairs_header(pairs_file: str) -> dict:
    header_info = {}
    # Open the gzipped file and read line by line
    with gzip.open(pairs_file, 'rt') as f:  # 'rt' for reading text mode
        for line in f:
            if line.startswith('##'):  # Global metadata
                key, value = line.strip().split(' ', 1)
                header_info['format'] = value
            elif line.startswith('#chromosome'):  # Chromosome size info
                parts = line.strip().split(' ')
                chrom = parts[1]
                size = int(parts[2])
                header_info.setdefault('chromsizes', {})[chrom] = size
            elif line.startswith('#column'):  # Other metadata
                continue
            elif line.startswith('#'):  # Other metadata
                key, value = line.strip().split(': ', 1)
                header_info[key[1:]] = value
            
            else:
                # Stop reading when actual data begins (non-commented lines)
                break
    
    return header_info


def visualize_hic_contact_matrix(contact_matrix: np.ndarray, output_file: str) -> None:
    plt.matshow(contact_matrix, cmap=REDMAP)
    plt.axis('off')
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()


def read_schic_pairs_file(path: str, PARAMETERS) -> np.ndarray:
    header = read_pairs_header(path)
    
    chrom_cumsum = {}
    
    cum_size = 0
    
    for chrom, size in header['chromsizes'].items():
        if chrom in ['chrX', 'chrY', 'chrM']:
            continue
        chrom_cumsum[chrom] = cum_size
        cum_size += (size // PARAMETERS['resolution'] + 1)
    
    
    data = pd.read_csv(path, 
        sep='\t', 
        compression='gzip',  # Since the file is gzipped
        header=None, 
        names=['readID', 'chr1', 'pos1', 'chr2', 'pos2', 'strand1', 'strand2', 'phase0', 'phase1'],
        comment='#'
    )
    
    data = data[~(data['chr1'].isin(['chrX', 'chrY', 'chrM']) | data['chr2'].isin(['chrX', 'chrY', 'chrM']))]

    data = data[(data['chr1'].isin(['chr1']) & data['chr2'].isin(['chr1']))]
    
    data['genomewide_pos1'] = data.apply(lambda row: chrom_cumsum[row['chr1']] + (row['pos1'] // PARAMETERS['resolution']), axis=1)
    data['genomewide_pos2'] = data.apply(lambda row: chrom_cumsum[row['chr2']] + (row['pos2'] // PARAMETERS['resolution']), axis=1)
    
    data = data.groupby(['genomewide_pos1', 'genomewide_pos2']).size().reset_index(name='counts')
    
    dense_matrix = np.zeros((chrom_cumsum['chr2'], chrom_cumsum['chr2']), dtype=int)
    dense_matrix[data['genomewide_pos1'], data['genomewide_pos2']] = data['counts'].values
    
    # Add the transpose to create a full matrix
    dense_matrix = np.triu(dense_matrix) + np.triu(dense_matrix, 1).T
    
    return dense_matrix



def preprocess_schic_matrix(matrix:np.ndarray, ice_norm=True, log_norm=True, smooth=True):
    # visualize_hic_contact_matrix(matrix, 'figures/preprocess/raw.png')
    
    if ice_norm:
        matrix = ICE_normalization(matrix)
        # visualize_hic_contact_matrix(matrix, 'figures/preprocess/ice.png')
    
    if log_norm:
        matrix = library_size_normalization(matrix)
        # visualize_hic_contact_matrix(matrix, 'figures/preprocess/log_norm.png')
    
    if smooth:
        matrix = personalized_pagerank_smooth(matrix)
        # visualize_hic_contact_matrix(matrix, 'figures/preprocess/smooth.png')
    
    np.fill_diagonal(matrix, 0)
    
    return matrix


def convert_pairs_file_to_schicluster_format(path: str, path_to_output_file: str, PARAMETERS) -> np.ndarray:
    data = pd.read_csv(path, 
        sep='\t', 
        compression='gzip',  # Since the file is gzipped
        header=None, 
        names=['readID', 'chr1', 'pos1', 'chr2', 'pos2', 'strand1', 'strand2', 'phase0', 'phase1'],
        comment='#'
    )
    
    data = data[abs(data['pos1'] - data['pos2']) <= 2*PARAMETERS['resolution']]



    data = data[~(data['chr1'].isin(['chrY', 'chrM']) | data['chr2'].isin(['chrY', 'chrM']))]
    data = data[['chr1', 'pos1', 'chr2', 'pos2']]
    # data['chrom_pos1'] = data.apply(lambda row: row['pos1'] // PARAMETERS['resolution'], axis=1)
    # data['chrom_pos2'] = data.apply(lambda row: row['pos2'] // PARAMETERS['resolution'], axis=1)
    lower_triangular_df = data.copy().rename(columns={'chr1': 'chr2', 'pos1': 'pos2', 'chr2': 'chr1', 'pos2': 'pos1'})
    full_matrix_df = pd.concat([data, lower_triangular_df])

    full_matrix_df.to_csv(path_to_output_file, sep='\t', index=False, header=False)
        
    

    # for chrom, size in chromsizes.items():
    #     chrom_data = data[(data['chr1'].isin([chrom]) & data['chr2'].isin([chrom]))].copy()
    #     chrom_data = chrom_data[['chr1', 'pos1', 'chr2', 'pos2']]
        
    #     chrom_data['chrom_pos1'] = chrom_data.apply(lambda row: row['pos1'] // PARAMETERS['resolution'], axis=1)
    #     chrom_data['chrom_pos2'] = chrom_data.apply(lambda row: row['pos2'] // PARAMETERS['resolution'], axis=1)
        
    #     chrom_data = chrom_data.groupby(['chrom_pos1', 'chrom_pos2']).size().reset_index(name='counts')
        
        
    
    
