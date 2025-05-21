import numpy as np

matrix = np.load("similarity_matrix.npy")
np.savetxt("similarity_matrix.txt", matrix, fmt="%.4f")
print("Matrix saved to similarity_matrix.txt")