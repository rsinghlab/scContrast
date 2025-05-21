import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

model_name = "microsoft/biogpt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).cuda().eval()

# Load phrases
with open("phrases.txt", "r") as f:
    phrases = [line.strip() for line in f if line.strip()]

embeddings = []

for phrase in phrases:
    print("Processing: ", phrase)
    inputs = tokenizer(phrase, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state  # shape: (1, seq_len, 1024)
        mean_embedding = hidden_states.mean(dim=1).squeeze(0).cpu().numpy()
        embeddings.append(mean_embedding)

np.save("embeddings.npy", np.stack(embeddings))
print("Embeddings saved to embeddings.npy")
