import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Load embeddings
embeddings = np.load("embeddings.npy")
similarity_matrix = cosine_similarity(embeddings)

# Save matrix
print("Saving similarity matrix to similarity_matrix.npy")
np.save("similarity_matrix.npy", similarity_matrix)

# Plot heatmap
with open("phrases.txt", "r") as f:
    phrases = [line.strip() for line in f]

plt.figure(figsize=(12, 10))
ax = sns.heatmap(similarity_matrix, xticklabels=phrases, yticklabels=phrases, cmap="viridis", annot=True, fmt=".2f", annot_kws={"size":8})
plt.title("Cosine Similarity Matrix")
plt.xticks(rotation=90)
plt.tight_layout()
print("Saving heatmap to similarity_matrix.png")
plt.savefig("similarity_matrix.png", dpi=300)
plt.show()
