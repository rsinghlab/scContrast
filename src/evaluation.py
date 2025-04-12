import math
import warnings
import numpy as np

from sklearn import metrics
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr, spearmanr

try:
	from scipy.stats import PearsonRConstantInputWarning
except:
	from scipy.stats import ConstantInputWarning as PearsonRConstantInputWarning




def mse(A:np.ndarray, B:np.ndarray):
    """evaluate the Mean Squared Error (MSE) between two scHi-C contact matrices

    Args:
        A (np.ndarray): a 2-dimensional numpy array representing a scHi-C contact matrix
        B (np.ndarray): a 2-dimensional numpy array representing a scHi-C contact matrix

    Returns:
        a scalar representing the MSE
    """
    if A.shape != B.shape:
        raise KeyError("matrices not of the same size")
    
    mse = np.square(np.subtract(A, B)).mean()
    if math.isnan(mse):
        raise ValueError("one or more input matrices are empty")
    
    return mse


def vstrans(d1, d2):
    """
    Variance stabilizing transformation to normalize read counts before computing
    stratum correlation. This normalizes counts so that different strata share similar
    dynamic ranges.
    Parameters
    ----------
    d1 : numpy.array of floats
        Diagonal of the first matrix.
    d2 : numpy.array of floats
        Diagonal of the second matrix.
    Returns
    -------
    r2k : numpy.array of floats
        Array of weights to use to normalize counts.
    """
    # Get ranks of counts in diagonal
    ranks_1 = np.argsort(d1) + 1
    ranks_2 = np.argsort(d2) + 1
    # Scale ranks betweeen 0 and 1
    nranks_1 = ranks_1 / max(ranks_1)
    nranks_2 = ranks_2 / max(ranks_2)
    nk = len(ranks_1)
    r2k = np.sqrt(np.var(nranks_1 / nk) * np.var(nranks_2 / nk))
    return r2k


def SCC(A:np.ndarray, B:np.ndarray, max_bins:int=120, correlation_method:str='PCC'):
    """
        Compute the stratum-adjusted correlation coefficient (SCC) between two
        Hi-C matrices up to max_dist. A Pearson correlation coefficient is computed
        for each diagonal in the range of 0 to max_dist and a weighted sum of those
        coefficients is returned.
        Parameters
        ----------
        mat1 : scipy.sparse.csr_matrix
            First matrix to compare.
        mat2 : scipy.sparse.csr_matrix
            Second matrix to compare.
        max_bins : int
            Maximum distance at which to consider, in bins.
        Returns
        -------
        scc : float
            Stratum adjusted correlation coefficient.
    """
    if (len(A.shape) != 2) and (len(B.shape) != 2):
        raise ValueError("both input matrices are of the wrong shape")
    elif len(A.shape) != 2:
        raise ValueError("first input matrix is of the wrong shape (" + str(len(A.shape)) + " dimensions instead of 2)")
    elif len(B.shape) != 2:
        raise ValueError("second input matrix is of the wrong shape (" + str(len(B.shape)) + " dimensions instead of 2)")
    elif A.shape != B.shape:
        raise KeyError("matrices not of the same size")
	
    if max_bins < 0 or max_bins > int(A.shape[0] - 5):
        max_bins = int(A.shape[0] - 5)


    mat1 = csr_matrix(A)
    mat2 = csr_matrix(B)
    
    corr_diag = np.zeros(len(range(max_bins)))
    weight_diag = corr_diag.copy()
    
    for d in range(max_bins):
        d1 = mat1.diagonal(d)
        d2 = mat2.diagonal(d)
        mask = (~np.isnan(d1)) & (~np.isnan(d2))
        d1 = d1[mask]
        d2 = d2[mask]
        # Silence NaN warnings: this happens for empty diagonals and will
        # not be used in the end.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=PearsonRConstantInputWarning
            )
            if correlation_method == 'PCC':
                cor = pearsonr(d1, d2)[0]

            elif correlation_method == 'SCC':
                cor = spearmanr(d1, d2)[0]
            else:
                print('Provided invalid correlation type')
                exit(1)
            corr_diag[d] = cor

        r2k = vstrans(d1, d2)
        weight_diag[d] = len(d1) * r2k

    corr_diag, weight_diag = corr_diag[1:], weight_diag[1:]
    mask = ~np.isnan(corr_diag)
    
    corr_diag, weight_diag = corr_diag[mask], weight_diag[mask]
    
    # Normalize weights
    weight_diag /= sum(weight_diag)
    
    # Weighted sum of coefficients to get SCCs
    scc = np.nansum(corr_diag * weight_diag)
    
    return scc