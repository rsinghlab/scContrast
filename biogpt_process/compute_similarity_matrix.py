import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load embeddings
embeddings = np.load("embeddings.npy")

#-----------------------------------------------------------------------------
# You can raise embeddings to a power or apply softmax
# Uncomment one of the following sections to apply transformations

# # Raise all elements to an arbitrary power
# raised_power = 2
# embeddings = np.sign(embeddings) * np.abs(embeddings) ** raised_power

# Softmax the matrix
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

embeddings = np.apply_along_axis(softmax, 1, embeddings)
#-----------------------------------------------------------------------------

# Compute similarity matrix
similarity_matrix = cosine_similarity(embeddings)

## Use filename based on what you used for embeddings
# filename = f"similarity_matrix_{raised_power}.pkl"
filename = f"similarity_matrix_softmax.pkl"

# Save matrix with raised_power in the filename
with open(filename, "wb") as f:
    pickle.dump(similarity_matrix, f)
print(f"Similarity matrix saved to {filename}")

## Uncomment to plot heatmap
# # Plot heatmap
# with open("phrases.txt", "r") as f:
#     phrases = [line.strip() for line in f]

# plt.figure(figsize=(12, 10))
# ax = sns.heatmap(similarity_matrix, xticklabels=phrases, yticklabels=phrases, cmap="viridis", annot=True, fmt=".2f", annot_kws={"size":8})
# plt.title("Cosine Similarity Matrix")
# plt.xticks(rotation=90)
# plt.tight_layout()
# print("Saving heatmap to similarity_matrix.png")
# plt.savefig("similarity_matrix.png", dpi=300)
# plt.show()
