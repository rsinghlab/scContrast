import os
os.environ["SCIB_USE_LEGACY_KNN"] = "1"

import pickle
import numpy as np
import scanpy as sc
import scib.metrics as scib_me
import tempfile
from statistics import mean, harmonic_mean

## Choose which AnnData object to load

# Load the raised power
print("Loading AnnData object...")
raised_power = 1
with open(f"adata_test_for_metrics_{raised_power}.pkl", "rb") as f:
    adata = pickle.load(f)
print("Done loading AnnData.")

# # Load the softmax
# print("Loading AnnData object...")
# with open(f"adata_test_for_metrics_softmax.pkl", "rb") as f:
#     adata = pickle.load(f)
# print("Done loading AnnData.")



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

metrics = {}

print("Computing NMI cluster/label...")
metrics["NMI cluster/label"] = scib_me.nmi(adata, cluster_key="cluster", label_key="Celltype")
print("Done.")

print("Computing ARI cluster/label...")
metrics["ARI cluster/label"] = scib_me.ari(adata, cluster_key="cluster", label_key="Celltype")
print("Done.")

print("Computing Label ASW...")
metrics["Label ASW"] = scib_me.silhouette(adata, label_key="Celltype", embed="X_emb")
print("Done.")

print("Computing Isolated label F1...")
metrics["Isolated label F1"] = scib_me.isolated_labels_f1(
    adata, label_key="Celltype", batch_key="batch", embed="X_emb", cluster_key="cluster")
print("Done.")

# print("Computing Isolated label silhouette...")
# metrics["Isolated label silhouette"] = scib_me.isolated_labels_silhouette(adata, label_key="Celltype", embed="X_emb")
# print("Done.")

print("Computing Batch ASW...")
metrics["Batch ASW"] = scib_me.silhouette_batch(adata, batch_key="tech", label_key="Celltype", embed="X_emb")
print("Done.")

print("Computing PCR batch...")
metrics["PCR batch"] = 1 - scib_me.pcr(adata, covariate="tech", embed="X_emb")
print("Done.")

# print("Computing Graph iLISI...")
# metrics["Graph iLISI"] = scib_me.ilisi_graph(adata, batch_key="tech", type_="embed", use_rep="X_emb")
# print("Done.")

print("Computing Graph connectivity...")
metrics["Graph connectivity"] = scib_me.graph_connectivity(adata, label_key="Celltype")
print("Done.")

# print("Computing Graph cLISI...")
# metrics["Graph cLISI"] = scib_me.clisi_graph(adata, label_key="Celltype", type_="embed", use_rep="X_emb")
# print("Done.")

# print("Computing HVG conservation...")
# metrics["HVG conservation"] = scib_me.hvg_overlap_score(adata, batch_key="tech")
# print("Done.")

# print("Computing Cell cycle conservation...")
# metrics["Cell cycle conservation"] = scib_me.cell_cycle_conservation(adata, batch_key="tech")
# print("Done.")

if "pseudotime" in adata.obs.columns:
    print("Computing Trajectory conservation...")
    metrics["Trajectory conservation"] = scib_me.trajectory_conservation(
        adata, batch_key="tech", pseudotime_key="pseudotime")
    print("Done.")

print("\n===== scIB Benchmark Metrics =====")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

# Compute overall batch, bio, and harmonic mean scores
batch_metrics = [
    metrics.get("Batch ASW"),
    metrics.get("PCR batch")
]
bio_metrics = [
    metrics.get("NMI cluster/label"),
    metrics.get("ARI cluster/label"),
    metrics.get("Label ASW"),
    metrics.get("Isolated label F1"),
    metrics.get("Graph connectivity")
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