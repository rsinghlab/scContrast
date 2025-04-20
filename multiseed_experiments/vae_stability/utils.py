import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage,dendrogram

import torch
import torch.nn as nn
from torch import Tensor

import json 

def rolling_average(vec, n=3):
    acc = np.cumsum(vec)
    acc_sum = acc[n:] - acc[:-n]
    return acc_sum / n

def single_agg(mtx,metric='cosine',method='average'):
    return dendrogram(linkage(mtx, metric=metric, method=method), no_plot=True)['leaves']

def double_agg(mtx,metric='cosine',method='average'):
    row_agg = dendrogram(linkage(mtx, metric=metric, method=method), no_plot=True)['leaves']
    col_agg = dendrogram(linkage(mtx.T, metric=metric, method=method), no_plot=True)['leaves']
    return mtx[row_agg].T[col_agg].T


def rolling_average(vec, n=3):
    acc = np.cumsum(vec)
    acc_sum = acc[n:] - acc[:-n]
    return acc_sum / n

from scipy.spatial.distance import pdist,squareform
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity

def concordance_plot(m1,m2,label_1="Mtx 1",label_2="Mtx 2",title=None,
                     agglomerate=None,
                     metric="correlation",
                     path=None,
                     plot_figure=True,
                     ticks_1=None,ticks_2=None,
                     figsize=(10,10),
                     cmap="bwr",
                     cmap_clip=None,
                     plot_text=True,
                     fontsize=8,
                    ):
    
    dim_1 = m1.shape[1]
    dim_2 = m2.shape[1]

    if ticks_1 is None:
        ticks_1 = np.arange(dim_1)
    if ticks_2 is None:
        ticks_2 = np.arange(dim_2)

    stacked = np.hstack([m1,m2])

    if metric == "partial_correlation":
        covariance = np.cov(stacked.T)
        precision = np.linalg.pinv(covariance)
        normalization = np.sqrt(np.outer(np.diag(precision),np.diag(precision)))
        partial_correlation = -1 * np.divide(precision,normalization)
        dist = 1-partial_correlation

    elif metric == "cosine":
        # dist = -1 * (squareform(pdist(stacked.T,metric='cosine')) - 1)
        # dist = squareform(pdist(stacked.T,metric='cosine'))
        dist = cosine_similarity(stacked.T)
    elif metric == "correlation":
        dist = 1 - squareform(pdist(stacked.T,metric='correlation'))
    elif metric == "spearman":
        # spearman is the opposite column orientation of pdist -_-
        dist = spearmanr(stacked)[0]
    else: 
        raise Exception(f"Only the following metrics are supported: cosine, correlation, spearman. You entered {metric}")

    print(dist.shape)
    con_dist = dist[dim_1:,:dim_1]
    
    if agglomerate is True:
        row_sort = single_agg(con_dist)
        col_sort = single_agg(con_dist.T)
        con_dist = con_dist[row_sort]
        con_dist = con_dist.T[col_sort].T
        ticks_1 = np.array(ticks_1)[col_sort]
        ticks_2 = np.array(ticks_2)[row_sort]
    if isinstance(agglomerate,list):
        row_sort = agglomerate[0]
        col_sort = agglomerate[1]
        con_dist = con_dist[row_sort]
        con_dist = con_dist.T[col_sort].T
        ticks_1 = np.array(ticks_1)[col_sort]
        ticks_2 = np.array(ticks_2)[row_sort]
    if agglomerate is None:
        pass

    if not plot_figure:
        return dist

    if cmap_clip is None:
        cmap_clip = [-1,1]
    plt.figure(figsize=figsize)
    plt.imshow(con_dist,vmin=cmap_clip[0],vmax=cmap_clip[1],cmap=cmap,aspect='auto')
    if title is None:
        title = f"Concordance between \n {label_1} and {label_2}"
    plt.title(title)


    if plot_text:
        text_style = {'horizontalalignment':'center','verticalalignment':'center','fontsize':fontsize}
    
        for i in range(dim_2):
            for j in range(dim_1):
                plt.text(j,i,str(np.round(con_dist[i,j],2)),**text_style)

    
    plt.xticks(np.arange(dim_1),labels=ticks_1,rotation=90, fontsize=fontsize)
    plt.yticks(np.arange(dim_2),labels=ticks_2, fontsize=fontsize)

    plt.colorbar()
    plt.xlabel(label_1)
    plt.ylabel(label_2)

    plt.tight_layout()
    if path is not None:
        plt.savefig(path)
    plt.show()

    return dist

# For use with above, steps is a tuple of 2 ints, (# of ensemble members, dimension of model) 
def ensemble_consistency(correlations,steps,plot=True):
    limits = np.arange(0,steps[0]*steps[1],steps[1])
    all_the_best = []
    for i,feature in enumerate(correlations):
        best_neighbors = []
        for start,end in zip(limits[:-1],limits[1:]):
            best = np.max(correlations[start:end,i])
            best_neighbors.append(best)
        all_the_best.append(best_neighbors)
    all_the_best = np.array(all_the_best)
    means = np.mean(all_the_best,axis=1)
    variances = np.var(all_the_best,axis=1)

    if plot:
        plt.figure()
        plt.title("Mean and variance of the best correlates among each other ensemble")
        plt.xlabel("Mean")
        plt.ylabel("Variance")
        plt.scatter(means,variances,s=1)
        plt.show()

    return means,variances

from scipy.stats import spearmanr
from scipy.stats import rankdata
from scipy.stats import rankdata

# def best_spearman(x, z, names=None, top_n=10):
#     """
#     Computes the partial Spearman's rho correlations between x and each column in z, 
#     controlling for the other columns in z. Returns the indices of the k most negatively
#     and k most positively correlated elements of z.

#     Parameters:
#     - x: A 1D numpy array representing the target variable.
#     - z: A 2D numpy array where each column represents a potential covariate.
#     - names: A 1D numpy array containing the names of variables, defaults to integer indices.
#     - top_n: Integer, the number of top positively and negatively correlated elements to return. Default is 10

#     Returns:
#     - pos_indices: The indices of the k most positively partially correlated columns of z.
#     - neg_indices: The indices of the k most negatively partially correlated columns of z.
#     """

#     if names is None:
#         names = np.arange(data.shape[1])
    
#     x = np.asarray(x).reshape(-1, 1)
#     z = np.asarray(z)

#     # we want the scipy rankdata here for tied ranks
#     x_ranked = rankdata(x, method='average').reshape(-1, 1)
#     z_ranked = np.apply_along_axis(lambda col: rankdata(col, method='average'), 0, z)

#     stacked = np.hstack([x_ranked,z_ranked])
    
#     corr = np.corrcoef(stacked, rowvar=False)

#     stat_sort = np.argsort(corr[0,1:])
#     print(names[stat_sort[:top_n]])
#     print(names[stat_sort[-top_n:]])

#     return corr[0,1:], stat_sort

# def best_partial_spearman(x, z, names=None, top_n=10):
#     """
#     Computes the partial Spearman's rho correlations between x and each column in z, 
#     controlling for the other columns in z. Returns the indices of the k most negatively
#     and k most positively correlated elements of z.

#     Parameters:
#     - x: A 1D numpy array representing the target variable.
#     - z: A 2D numpy array where each column represents a potential covariate.
#     - names: A 1D numpy array containing the names of variables, defaults to integer indices.
#     - top_n: Integer, the number of top positively and negatively correlated elements to return. Default is 10

#     Returns:
#     - pos_indices: The indices of the k most positively partially correlated columns of z.
#     - neg_indices: The indices of the k most negatively partially correlated columns of z.
#     """

#     if names is None:
#         names = np.arange(data.shape[1])

#     x = np.asarray(x).reshape(-1, 1)
#     z = np.asarray(z)

#     # Pulling in scipy because tied ranks are important. 
#     # We could probably eliminate a dependency by just writing rankdata but scipy is fairly stable (if slow)
#     x_ranked = rankdata(x, method='average').reshape(-1, 1)
#     z_ranked = np.apply_along_axis(lambda col: rankdata(col, method='average'), 0, z)

#     stacked = np.hstack([x_ranked, z_ranked])
#     covariance_matrix = np.cov(stacked, rowvar=False)

#     precision_matrix = np.linalg.pinv(covariance_matrix)
#     partial_corr = -precision_matrix[0, 1:] / np.sqrt(precision_matrix[0, 0] * np.diag(precision_matrix)[1:])

#     stat_sort = np.argsort(partial_corr)
#     print(names[stat_sort[:top_n]])
#     print(names[stat_sort[-top_n:]])

#     return partial_corr, stat_sort


# def best_partial(x, z, names=None, top_n=10):
#     """
#     Computes the partial Spearman's rho correlations between x and each column in z, 
#     controlling for the other columns in z. Returns the indices of the k most negatively
#     and k most positively correlated elements of z.

#     Parameters:
#     - x: A 1D numpy array representing the target variable.
#     - z: A 2D numpy array where each column represents a potential covariate.
#     - names: A 1D numpy array containing the names of variables, defaults to integer indices.
#     - top_n: Integer, the number of top positively and negatively correlated elements to return. Default is 10

#     Returns:
#     - pos_indices: The indices of the k most positively partially correlated columns of z.
#     - neg_indices: The indices of the k most negatively partially correlated columns of z.
#     """

#     x = np.asarray(x).reshape(-1, 1)
#     z = np.asarray(z)

#     stacked = np.hstack([x, z])

#     covariance_matrix = np.cov(stacked, rowvar=False)
#     precision_matrix = np.linalg.pinv(covariance_matrix)

#     partial_corr = -precision_matrix[0][1:] / np.sqrt(precision_matrix[0, 0] * np.diag(precision_matrix)[1:])

#     # np.isfinite(partial_corr)
    
#     stat_sort = np.argsort(partial_corr)
#     print(names[stat_sort[:top_n]])
#     print(names[stat_sort[-top_n:]])

#     return partial_corr, stat_sort

def bulk_partial(x, z, regularization = 1e-5):
    """
    """

    x = np.asarray(x)
    z = np.asarray(z)

    stacked = np.hstack([x, z])

    covariance_matrix = np.cov(stacked, rowvar=False)
    precision_matrix = np.linalg.pinv(covariance_matrix)

    outer_sqrt_precision = np.sqrt(np.outer(np.diag(precision_matrix),np.diag(precision_matrix)))
    
    partial_corr = -precision_matrix / (outer_sqrt_precision + regularization)

    mask = (~np.isfinite(partial_corr)) & (covariance_matrix < 0.00000000001)
    if np.sum(mask) > 0:
        partial_corr[mask] = 0
        print("WARNING, DEGENERATE ELEMENTS IN PRECISION MATRIX")

    lim = x.shape[1]
    lim_corr = partial_corr[:lim,lim:]
    
    
    stat_sort = np.argsort(lim_corr,axis=1)

    res = {
        'correlations':lim_corr,
        'correlation_sort':stat_sort,
        # 'covariance_matrix':covariance_matrix,
        # 'precision_matrix':precision_matrix
    }

    return res

def bulk_correlation(x, z,regularization = 1e-5):

    dim_1 = x.shape[1]
    dim_2 = z.shape[1]
    
    corr = np.corrcoef(x.T+(np.random.random(x.T.shape)*regularization),z.T+(np.random.random(z.T.shape)*regularization))[:dim_1,dim_1:]
    stat_sort = np.argsort(corr,axis=1)

    res = {
        'correlations':corr,
        'correlation_sort':stat_sort,
    }

    return res


from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

def raw_mse(mtx):
    """
    Quickie that computes the MSE of a matrix to the trivial estimator (means).

    Parameters:
    - mtx: A 2D numpy array where each row is a sample and each column is a feature.

    Returns:
    - raw: A float that is the MSE of the trivial estimator.
    """
    means = np.mean(mtx,axis=0)
    mean_stack = np.vstack([means.copy() for _ in range(mtx.shape[0])])
    raw = mean_squared_error(mtx,mean_stack)
    return raw

def pca_recovery(mtx,validation_mtx=None,k=10):
    """
    Computes there-and-back reconstruction error for PCA on a given matrix. 

    Parameters:
    - mtx: A 2D numpy array where each row is a sample and each column is a feature. This matrix is used to fit the PCA model.
    - validation_mtx: A 2D numpy array that is actually evaluated for reconstruction error. If None, `mtx` is used for both training and validation.
    - k: Integer, the number of principal components to keep in PCA. Default is 10.

    Returns:
    - error: The Mean Squared Error (MSE) between the original validation matrix and the PCA-reconstructed matrix.
    """
    if validation_mtx is None:
        validation_mtx = mtx.copy()
    model = PCA(n_components=k)
    model.fit(mtx)
    embedding = model.transform(validation_mtx)
    recovery = model.inverse_transform(embedding) 

    error = mean_squared_error(validation_mtx,recovery)
    return error

def auto_split_range(data,cmap='PiYG',force_range=None):
    output = {
        'vmin':None,
        'vmax':None,
        'cmap':cmap
    }

    data_max = np.max(data)
    data_min = np.min(data)

    abs_range = np.max([data_max,np.abs(data_min)])

    if force_range is not None:
        abs_range = force_range
        
    output['vmin'] = -1 * abs_range
    output['vmax'] = abs_range

    return output

import json

def dump_args_to_json(path):
    """
    Intended for use as a decorator around other functions
    Dumps arguments of functions to a json at the specified location 
    
    Ex:
    @dump_args_to_json("./constructor_args.json")
    def train(foo,bar,baz):
        etc
    """
    def known_function(func):
            def known_path(*args,**kwargs):
                    arguments = {
                        'args': args,
                        'kwargs': kwargs
                    }
                    with open(path, 'w') as f:
                        json.dump(arguments, f, indent=4)
                    return func(*args,**kwargs)
            return known_path
    return known_function


def dump_factor_json(model,factor,path='./factor_properties/'):

    # Numpy arrays don't easily cast to json
    def list_cast(numpy_array):
        return [list_cast(x) if isinstance(x,np.ndarray) or isinstance(x,list) else float(x) for x in numpy_array]

    def dict_cast(dict_tree):
        for key in dict_tree:
            if isinstance(dict_tree[key],dict):
                dict_tree[key] = dict_cast(dict_tree[key])
            if isinstance(dict_tree[key],np.ndarray) or isinstance(dict_tree[key],list):
                dict_tree[key] = list_cast(dict_tree[key])

        return dict_tree

    if f"factor_{factor}" not in model.analysis:
        print(f"Warning, analysis for factor {factor} has not been run, running it now, but something may be wrong.")
        model.factor_summary(factor)
    
    factor_dict = deepcopy(model.analysis[f'factor_{factor}'])

    factor_dict = dict_cast(factor_dict)

    with open(path + f'factor_{factor}.json',mode='w') as f:
        factor_json = f.write(json.dumps(factor_dict))

    return factor_json

import biomart 

def translate_ensembl_garbage(ensembl_symbols,server='http://useast.ensembl.org/biomart',local_dataset=None,dataset=None):                                   
    # Set up connection to server           
    print("Setting up connection")
    

    if local_dataset is None:
        server = biomart.BiomartServer(server)         
        if dataset is None:
            # dataset = 'mmusculus_gene_ensembl'
            dataset = 'hsapiens_gene_ensembl'
        mart = server.datasets[dataset]               

        print("Fetching translations")

        attributes = ['ensembl_transcript_id', 'hgnc_symbol',
                    'ensembl_gene_id', 'ensembl_peptide_id']
        response = mart.search({'attributes': attributes})
        data = response.raw.data.decode('ascii')

        print("Translating")
        ensembl_to_genesymbol = {}                                                  
        for line in data.splitlines():                                              
            line = line.split('\t')                                                 
            # The entries are in the same order as in the `attributes` variable
            transcript_id = line[0]                                                 
            gene_symbol = line[1]                                                   
            ensembl_gene = line[2]                                                  
            ensembl_peptide = line[3]                                               
                                                                                    
            # Some of these keys may be an empty string. If you want, you can 
            # avoid having a '' key in your dict by ensuring the attributes
            # have a nonzero length before adding them to the dict
            ensembl_to_genesymbol[transcript_id] = gene_symbol                      
            ensembl_to_genesymbol[ensembl_gene] = gene_symbol                       
            ensembl_to_genesymbol[ensembl_peptide] = gene_symbol
    
    else:
        ensembl_to_genesymbol = json.load(open(local_dataset,mode='r'))

    translation = [ensembl_to_genesymbol.get(gene,gene) for gene in ensembl_symbols]
    return translation,ensembl_to_genesymbol

# Slated for deletion due to redundancy
# def batch_tensor(tns,batch_size=100):
#     indices = np.arange(tns.shape[0])
#     np.random.shuffle(indices)
#     batch_indices = indices[:batch_size]
#     batched = Tensor(tns[batch_indices].clone().detach())
#     batched = batched.to(tns.device)
#     return batched

def make_batch_indices(data,batch_size=100):
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    batch_indices = indices[:batch_size]
    return batch_indices

from scipy.sparse import issparse

def batch_data(data,batch_size=100,batch_indices=None,device=torch.device('cpu')):
    if batch_indices is None:
        batch_indices = make_batch_indices(data,batch_size=batch_size)
    data_selection = data[batch_indices]
    if issparse(data_selection):
        data_selection = np.array(data_selection.todense())
    batched = Tensor(data_selection)
    batched = batched.to(device)
    return batched

def weighted_batch(tns,batch_size=100,weights=None):
    if weights is None:
        weights = np.ones(tns.shape[0])
    ranks = np.random.random(tns.shape[0])
    ranks *= weights
    sort = np.argsort(ranks)
    batch_indices = sort[:batch_size]
    batched = Tensor(tns[batch_indices].clone().detach())
    batched = batched.to(tns.device)
    return batched

from itertools import product as iter_product

def permute_dictionary(pd):
    flattened = [[(k,e) for e in v] for k,v in pd.items()]
    dictified = [dict(l) for l in iter_product(*flattened)]
    return dictified


# WARNING CGPT AHEAD, THIS COULD BE WRONG OR BAD
# NEEDS TESTING 
# BUT FUCKING WHY ISN"T THIS BUILT INTO LEIDENALG

import numpy as np
from sklearn.neighbors import NearestNeighbors
import igraph as ig
import leidenalg as ldng

def leidenalg(
    data: np.ndarray,
    n_neighbors: int = 15,
    metric: str = 'euclidean',
    resolution: float = 1.0,
    random_state: int = 0,
):
    """
    Perform Leiden clustering on a generic numpy data matrix with a Scanpy-like API.
    
    Parameters
    ----------
    data : np.ndarray
        The input data matrix of shape (n_samples, n_features).
    n_neighbors : int, optional (default: 15)
        Number of nearest neighbors to use when constructing the graph.
    metric : str, optional (default: 'euclidean')
        Metric to use for building the nearest neighbor graph (e.g., 'euclidean', 'manhattan', 'cosine').
        Must be a valid metric recognized by sklearn.neighbors.NearestNeighbors.
    resolution : float, optional (default: 1.0)
        Resolution parameter for the Leiden algorithm, akin to the resolution parameter in Scanpy.
    random_state : int, optional (default: 0)
        Random seed for reproducibility.
    directed : bool, optional (default: False)
        Whether to treat the resulting graph as directed or undirected.
        
    Returns
    -------
    labels : np.ndarray
        An array of cluster labels for each row in `data`.
    """
    
    # 1. Build the k-nearest neighbors graph
    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric
    )
    nn.fit(data)
    
    # Get the indices of neighbors and distances
    distances, neighbors = nn.kneighbors(data)
    
    # 2. Construct adjacency in a coordinate list (COO) format
    n_samples = data.shape[0]
    row_indices = []
    col_indices = []
    edge_weights = []
    
    for i in range(n_samples):
        for j_idx, dist in zip(neighbors[i], distances[i]):
            # i connects to j_idx
            row_indices.append(i)
            col_indices.append(j_idx)
            # Use 1/dist if you want a weighted graph that emphasizes closer neighbors
            # Or just use 1 if you want an unweighted graph
            weight = 1.0 if dist == 0.0 else 1.0 / dist
            edge_weights.append(weight)
    
    # 3. Create an igraph Graph from adjacency
    #    - If undirected, we only need edges in one direction 
    #      or we can add them all and set the graph to be undirected.
    edges = list(zip(row_indices, col_indices))
    
    g = ig.Graph(n=n_samples, edges=edges)
    # to ensure it is treated undirected in edge attributes:
    g.simplify(combine_edges="first")
    
    # Assign weights
    g.es['weight'] = edge_weights
    
    # 4. Run Leiden algorithm
    partition = ldng.find_partition(
        g,
        partition_type=ldng.RBConfigurationVertexPartition,
        weights=g.es['weight'],
        resolution_parameter=resolution,
        seed=random_state
    )
    
    # 5. Extract cluster labels
    labels = np.array(partition.membership)
    
    return labels

