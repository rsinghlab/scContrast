import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load model and tokenizer
model_name = "deepseek-ai/deepseek-coder-1.3b-base" # "microsoft/biogpt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).cuda().eval()

# Load phrases
with open("phrases.txt", "r") as f:
    phrases = [line.strip() for line in f if line.strip()]

# Compute embeddings
embeddings = []
for phrase in phrases:
    print("Processing:", phrase)
    inputs = tokenizer(phrase, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state  # shape: (1, seq_len, hidden_dim)
        mean_embedding = hidden_states.mean(dim=1).squeeze(0).cpu().numpy()
        embeddings.append(mean_embedding)

embeddings = np.stack(embeddings)

# Save embeddings to file
filename = "similarity_matrix_deepseek.pkl"

# Compute similarity matrix
similarity_matrix = cosine_similarity(embeddings)

with open(filename, "wb") as f:
    pickle.dump(similarity_matrix, f)
print(f"Similarity matrix saved to {filename}")

