import os
os.environ["SCIB_USE_LEGACY_KNN"] = "1"

import pickle
import numpy as np
import scanpy as sc
import scib.metrics as scib_me
import tempfile
from statistics import mean, harmonic_mean

VERSION = 'v3,5'

## Choose which AnnData object to load
print("Loading AnnData object...")
with open(f"soft_similarity_experiments/adata_test_for_metrics_1.pkl", "rb") as f:
    adata = pickle.load(f)
with open(f"data/pickled/tabula_muris/tm_adata_test_length_normalized_{VERSION}.pkl", "rb") as f:
    adata_pre = pickle.load(f)
print("Done loading AnnData.")

print("Pre shape:", adata_pre.shape)
print("Post shape:", adata.shape)

print("Computing neighborhood graph using embedding 'X_emb'...")
sc.pp.neighbors(adata, use_rep="X_emb")
print("Done computing neighbors.")

print("Running Leiden clustering with optimal resolution against Celltype...")
scib_me.cluster_optimal_resolution(
    adata,
    label_key="Celltype",
    cluster_key="cluster",       # where to save the result
    directed=True,               # default; change to False if using igraph
    random_state=0               # for reproducibility
)
print("Done optimal clustering.")

print("Creating temporary directory for LISI...")
os.environ["LISI_TMP"] = tempfile.mkdtemp(prefix="lisi_")
print(f"LISI_TMP set to {os.environ['LISI_TMP']}")

# all of the metrics here

metrics = {}

print("Computing NMI cluster/label...")
metrics["NMI cluster/label"] = scib_me.nmi(adata, cluster_key="cluster", label_key="Celltype")
print("Done.")

print("Computing ARI cluster/label...")
metrics["ARI cluster/label"] = scib_me.ari(adata, cluster_key="cluster", label_key="Celltype")
print("Done.")

print("Computing Silhoutte...")
metrics["Silhoutte"] = scib_me.silhouette(adata, label_key="Celltype", embed="X_emb")
print("Done.")

print("Computing Silhoutte Batch...")
metrics["Silhoutte Batch"] = scib_me.silhouette_batch(adata, batch_key="batch", label_key="Celltype", embed="X_emb")
print("Done.")

print("Computing Isolated label F1...")
metrics["Isolated label F1"] = scib_me.isolated_labels_f1(
    adata, label_key="Celltype", batch_key="batch", embed="X_emb", cluster_key="cluster")
print("Done.")

print("Computing Isolated label asw...")
metrics["Isolated label asw"] = scib_me.isolated_labels_asw(adata, label_key="Celltype", batch_key="batch", embed="X_emb")
print("Done.")








# print("Computing lisi...")
# metrics["lisi"] = scib_me.lisi.lisi_graph(adata, label_key="Celltype", batch_key="tech", type_="embed")
# print("Done")

# print("Computing Graph iLISI...")
# metrics["Graph iLISI"] = scib_me.ilisi_graph(adata, batch_key="tech", use_rep="X_emb", type_="embed")
# print("Done.")

# print("Computing Graph cLISI...")
# metrics["Graph cLISI"] = scib_me.clisi_graph(adata, label_key="Celltype", use_rep="X_emb")
# print("Done.")









print("Computing Graph connectivity...")
metrics["Graph connectivity"] = scib_me.graph_connectivity(adata, label_key="Celltype")
print("Done.")

# print("Computing HVG overlap...")
# metrics["HVG overlap"] = scib_me.hvg_overlap(adata_pre=adata_pre, adata_post=adata, batch_key="tech")
# print("Done.")

print("Computing Cell cycle conservation...")
metrics["Cell cycle conservation"] = scib_me.cell_cycle(adata_pre=adata_pre, adata_post=adata, batch_key="tech")
print("Done.")

print("Computing PCR batch...")
metrics["PCR batch"] = 1 - scib_me.pcr(adata, covariate="tech", embed="X_emb")
print("Done.")

print("\n===== scIB Benchmark Metrics =====")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

# Compute overall batch, bio, and harmonic mean scores
batch_metrics = [
    metrics.get("Silhoutte Batch"),  # batch ASW equivalent
    metrics.get("PCR batch"),
    metrics.get("kBET"),
    metrics.get("lisi"),
    metrics.get("Graph iLISI")
]

bio_metrics = [
    metrics.get("NMI cluster/label"),
    metrics.get("ARI cluster/label"),
    metrics.get("Silhoutte"),  # label ASW equivalent
    metrics.get("Isolated label F1"),
    metrics.get("Isolated label asw"),
    metrics.get("Graph connectivity"),
    metrics.get("Graph cLISI")
]

# Remove None values if any metric was not computed
batch_metrics = [float(m) for m in batch_metrics if m is not None]
bio_metrics = [float(m) for m in bio_metrics if m is not None]

overall_batch = mean(batch_metrics) if batch_metrics else float('nan')
overall_bio = mean(bio_metrics) if bio_metrics else float('nan')
overall = harmonic_mean([overall_batch, overall_bio]) if batch_metrics and bio_metrics else float('nan')

print(f"\nOverall batch: {overall_batch:.4f}")
print(f"Overall bio: {overall_bio:.4f}")
print(f"Overall score: {overall:.4f}")