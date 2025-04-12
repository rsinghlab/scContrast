import os
import pickle
import scanpy as sc
import torch
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything


from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.autolayout'] = False
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['figure.dpi'] = 600

def visualize_train_adata(model, train_adata, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(f'{output_folder}/train', exist_ok=True)

    with torch.no_grad():
        latent_representations = model.encoder(
            torch.tensor(train_adata.X.toarray(), dtype=torch.float32)
            )
    latent_np = latent_representations.detach().cpu().numpy()
    train_adata.obsm['X_latent'] = latent_np  # Store the latents in the obsm dictionary

    train_adata = train_adata.copy() #NOTE: Apparently AnnData==0.10.8 requires this..?

    sc.pp.neighbors(train_adata, use_rep='X_latent')  # Compute neighbors using latent space
    sc.tl.umap(train_adata)  # Run UMAP

    # UMAP train tissues
    fig = sc.pl.umap(
        train_adata, color='Celltype', show=False, return_fig=True, palette=list(mpl.colors.CSS4_COLORS.values())
    )
    fig.savefig(f"{output_folder}/train/celltype.png")
    plt.close(fig)

    fig = sc.pl.umap(
        train_adata, color='method', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/train/method.png")
    plt.close(fig)

    fig = sc.pl.umap(
        train_adata, color='tissue', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/train/tissue.png")
    plt.close(fig)

    fig = sc.pl.umap(
        train_adata, color='celltype_tech_availability', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/train/celltype_tech_availability.png")
    plt.close(fig)

def visualize_test_adata(model, test_adata, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(f'{output_folder}/train', exist_ok=True)

    with torch.no_grad():
        latent_representations_test = model.encoder(
            torch.tensor(test_adata.X.toarray(), dtype=torch.float32)
            )
    latent_np_test = latent_representations_test.detach().numpy()
    test_adata.obsm['X_latent'] = latent_np_test

    test_adata = test_adata.copy()
        
    sc.pp.neighbors(test_adata, use_rep='X_latent')  # Compute neighbors using latent space
    sc.tl.umap(test_adata)  # Run UMAP

    # UMAP test
    fig = sc.pl.umap(
        test_adata, color='Celltype', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/celltype.png")
    plt.close(fig)

    fig = sc.pl.umap(
        test_adata, color='method', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/method.png")
    plt.close(fig)

    fig = sc.pl.umap(
        test_adata, color='tissue', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/tissue.png")
    plt.close(fig)

    fig = sc.pl.umap(
        test_adata, color='celltype_tech_availability', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/celltype_tech_availability.png")
    plt.close(fig)

    # UMAP individual test tissues, 10x only
    fig = sc.pl.umap(
        test_adata[
            (test_adata.obs['tissue'] == 'Skin') &
            (test_adata.obs['tech'] == '10x')].copy(), color='Celltype', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/celltype_Skin_10x.png")
    plt.close(fig)

    fig = sc.pl.umap(
        test_adata[
            (test_adata.obs['tissue'] == 'Liver') &
            (test_adata.obs['tech'] == '10x')].copy(), color='Celltype', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/celltype_Liver_10x.png")
    plt.close(fig)

    fig = sc.pl.umap(
        test_adata[
            (test_adata.obs['tissue'] == 'Limb_Muscle') &
            (test_adata.obs['tech'] == '10x')].copy(), color='Celltype', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/celltype_Limb_Muscle_10x.png")
    plt.close(fig)

    fig = sc.pl.umap(
        test_adata[
            (test_adata.obs['tissue'] == 'Pancreas') &
            (test_adata.obs['tech'] == '10x')].copy(), color='Celltype', show=False, return_fig=True
    )
    fig.savefig(f"{output_folder}/test/celltype_Pancreas_10x.png")
    plt.close(fig)

def evaluate_tm_adata(model, tm_adata, latent_key, output_folder, figure_name):
    os.makedirs(output_folder, exist_ok=True)
    with torch.no_grad():
        latent_representations = model.encoder(
            torch.tensor(tm_adata.X.toarray(), dtype=torch.float32) #removed .to(device)
            )

    latent_np_encoder = latent_representations.detach().cpu().numpy()
    tm_adata.obsm[latent_key] = latent_np_encoder
    tm_adata.uns.pop('Celltype_colors', None)

    sc.pp.neighbors(tm_adata, use_rep=latent_key)  # Compute neighbors using latent space
    sc.tl.umap(tm_adata)  # Run UMAP

    fig = sc.pl.umap(tm_adata, color='Celltype', legend_loc=None, title='', frameon=False, size=50, return_fig=True)
    fig.savefig(f'{output_folder}/{figure_name}_celltypes.png')

    fig = sc.pl.umap(tm_adata, color='tech', legend_loc=None, title='', frameon=False, size=50, return_fig=True)
    fig.savefig(f'{output_folder}/{figure_name}_techs.png')

    fig = sc.pl.umap(tm_adata, color='celltype_tech_availability', legend_loc=None, title='', frameon=False, size=50, return_fig=True)
    fig.savefig(f'{output_folder}/{figure_name}_availabilities.png')

    return evaluate_clusters(adata=tm_adata, latent_key=latent_key, color_key='Celltype')

def evaluate_clusters(adata, latent_key, color_key='Celltype'):
    # Compute Silhouette
    print('embryo adata, informed FT')
    sil_score_true = silhouette_score(adata.obsm[latent_key], adata.obs[color_key])
    print(f"Silhouette Score ({latent_key}): {sil_score_true:.4f}")

    n_clusters = len(adata.obs[color_key].unique())
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    adata.obs["kmeans_clusters"] = kmeans.fit_predict(adata.obsm[latent_key]).astype(str)
    sc.pp.neighbors(adata, use_rep=latent_key)  # use_rep tells Scanpy to use your embedding
    sc.tl.leiden(adata, resolution=1.0, key_added="leiden_clusters")  # Run Leiden clustering
    true_labels = adata.obs[color_key].values  # True cell types
    predicted_labels = adata.obs["kmeans_clusters"].values  # Use "leiden_clusters" if you chose Leiden

    # Compute ARI
    ari_score = adjusted_rand_score(true_labels, predicted_labels)
    print(f"ARI for {latent_key} Clustering: {ari_score:.4f}")

    return sil_score_true, ari_score

def HiRES_zeroshot_evaluations(model, HiRES_adata, latent_key, output_folder):
    with torch.no_grad():
        latent_representations_embryo = model.encoder(
            torch.tensor(HiRES_adata.X, dtype=torch.float32) # deleted .to(device); needed?
            )
    latent_np_encoder = latent_representations_embryo.cpu().numpy()
    HiRES_adata.obsm[latent_key] = latent_np_encoder

    sc.pp.neighbors(HiRES_adata, use_rep=latent_key)  # Compute neighbors using latent space
    sc.tl.umap(HiRES_adata)  # Run UMAP

    os.makedirs(output_folder, exist_ok=True)

    fig = sc.pl.umap(HiRES_adata, color='Celltype', legend_loc=None, title='', frameon=False, size=50, return_fig=True)
    fig.savefig(f'{output_folder}/zeroshot_celltypes.png')
    plt.close(fig)

    fig = sc.pl.umap(HiRES_adata, color='Stage', legend_loc=None, title='', frameon=False, size=50, return_fig=True)
    fig.savefig(f'{output_folder}/zeroshot_stages.png')
    plt.close(fig)

    fig = sc.pl.umap(HiRES_adata, color='Cellcycle phase', legend_loc=None, title='', frameon=False, size=50, return_fig=True)
    fig.savefig(f'{output_folder}/zeroshot_phases.png')
    plt.close(fig)

    return evaluate_clusters(adata=HiRES_adata, latent_key=latent_key, color_key='Celltype')

def HiRES_finetune_evaluations(model, 
                               HiRES_adata, 
                               HiRES_dataloader, 
                               track_metric, 
                               hyperparameters={}, 
                               precomputed_gene_clusters_path=None, 
                               precomputed_mu_sigma_path=None, 
                               latent_key=None, 
                               augmentations_pipeline=None, 
                               figure_name=None,
                               output_folder=None):
    if precomputed_gene_clusters is not None:
        with open(precomputed_gene_clusters_path, 'rb') as f:
            precomputed_gene_clusters = pickle.load(f)

    with open(precomputed_mu_sigma_path, 'rb') as f:
        precomputed_mu_sigma = pickle.load(f)

    for k, v in precomputed_gene_clusters.items():
        model.k = v

    for k, v in precomputed_mu_sigma.items():
        model.k = v
    
    for k, v in hyperparameters:
        model.k = v
    
    model.augmentations_pipeline = augmentations_pipeline

    early_stop = pl.callbacks.EarlyStopping(track_metric, patience=4)
    tm_trained_hires_finetuned = Trainer(
        max_epochs=50,
        precision='bf16-mixed',
        callbacks=[early_stop],
    )
    tm_trained_hires_finetuned.fit(model, train_dataloaders=HiRES_dataloader)
    model.eval()

    latent_representations = model.encoder(torch.tensor(HiRES_adata.X, dtype=torch.float32))
    latent_np = latent_representations.detach().numpy()
    HiRES_adata.obsm[latent_key] = latent_np  

    sc.pp.neighbors(HiRES_adata, use_rep=latent_key)
    sc.tl.umap(HiRES_adata)  

    os.makedirs(output_folder, exist_ok=True)

    fig = sc.pl.umap(HiRES_adata, color='Celltype', legend_loc=None, title='', frameon=False, size=50, return_fig=True)
    fig.savefig(f'{output_folder}/{figure_name}_celltypes.png')
    plt.close(fig)

    fig = sc.pl.umap(HiRES_adata, color='Stage', legend_loc=None, title='', frameon=False, size=50, return_fig=True)
    fig.savefig(f'{output_folder}/{figure_name}_stages.png')
    plt.close(fig)

    fig = sc.pl.umap(HiRES_adata, color='Cellcycle phase', legend_loc=None, title='', frameon=False, size=50, return_fig=True)
    fig.savefig(f'{output_folder}/{figure_name}_phases.png')
    plt.close(fig)

    return evaluate_clusters(adata=HiRES_adata, latent_key=latent_key, color_key='Celltype')