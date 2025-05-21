import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
import torch.nn.functional as F
from augmentations import *


class scRNASeqEncoder(pl.LightningModule):
    def __init__(self, PARAMETERS):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(PARAMETERS['num_genes'], 256),
            nn.BatchNorm1d(256),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(128, PARAMETERS['latent_dimension'])


            # nn.Linear(PARAMETERS['num_genes'], 256),
            # nn.BatchNorm1d(256),  # Batch Normalization
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.BatchNorm1d(128),  # Batch Normalization
            # nn.ReLU(),
            # nn.Linear(128, PARAMETERS['latent_dimension'])
        )

    def forward(self, x):
        return self.encoder(x)
    
class scRNASeqEncoderLarge(pl.LightningModule):
    def __init__(self, PARAMETERS):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(PARAMETERS['num_genes'], 512),
            nn.BatchNorm1d(512),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(256, PARAMETERS['latent_dimension'])
        )

    def forward(self, x):
        return self.encoder(x)

class scRNASeqDecoder(pl.LightningModule):
    def __init__(self, PARAMETERS):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(PARAMETERS['latent_dimension'], 128),  # Decoder layers
            nn.BatchNorm1d(128),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(256, PARAMETERS['hvgs']),    # Output layer
            nn.Sigmoid()                   # Use Sigmoid for output
        )

    def forward(self, x):
        return self.decoder(x)
    
class scRNASeqProjectionHead(pl.LightningModule):
    def __init__(self, PARAMETERS):
        super().__init__()
        self.projectionHead = nn.Sequential(
            nn.Linear(PARAMETERS['latent_dimension'], PARAMETERS['latent_dimension']),
            nn.ReLU(),
            nn.Linear(PARAMETERS['latent_dimension'], PARAMETERS['latent_dimension']),
        )
    
    def forward(self, x):
        return self.projectionHead(x)

class scRNASeqProjectionHeadExpander(pl.LightningModule):
    def __init__(self, PARAMETERS):
        super().__init__()
        self.projectionHead = nn.Sequential(
            nn.Linear(PARAMETERS['latent_dimension'], 2*PARAMETERS['latent_dimension']),
            nn.BatchNorm1d(2*PARAMETERS['latent_dimension']),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(2*PARAMETERS['latent_dimension'], 2*PARAMETERS['latent_dimension']),
            nn.BatchNorm1d(2*PARAMETERS['latent_dimension']),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(2*PARAMETERS['latent_dimension'], 2*PARAMETERS['latent_dimension']),
        )
    
    def forward(self, x):
        return self.projectionHead(x)
    
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, projections_1, projections_2):
        # Normalize the projections to have unit length
        projections_1 = F.normalize(projections_1, dim=1)
        projections_2 = F.normalize(projections_2, dim=1)

        # Calculate cosine similarity and scale by temperature
        similarities = torch.matmul(projections_1, projections_2.T) / self.temperature

        # Create labels (same indices mean same images/views)
        batch_size = projections_1.size(0)
        contrastive_labels = torch.arange(batch_size, device=projections_1.device)

        # Calculate cross-entropy loss for both directions
        loss_1_2 = F.cross_entropy(similarities, contrastive_labels)
        loss_2_1 = F.cross_entropy(similarities.T, contrastive_labels)

        # Return the mean loss
        return (loss_1_2 + loss_2_1) / 2


def off_diagonal(x):
    # Returns a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n-1, n+1)[:,1:].flatten()

class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambda_offdiag=5e-3):
        """
        lambda_offdiag: the weight for off-diagonal terms in the cross-correlation matrix.
        """
        super().__init__()
        self.lambda_offdiag = lambda_offdiag

    def forward(self, z1, z2):
        """
        z1, z2: [batch_size x embedding_dim] after the projector head.
        Compute cross-correlation matrix between (z1, z2), try to force it to identity.
        """
        # ---- Normalize each feature along the batch dimension ----
        # Subtract mean over batch, divide by std over batch
        z1_norm = (z1 - z1.mean(0)) / z1.std(0)
        z2_norm = (z2 - z2.mean(0)) / z2.std(0)

        # ---- Cross-correlation matrix ----
        batch_size = z1.shape[0]
        c = torch.matmul(z1_norm.T, z2_norm) / batch_size
        # c is [embedding_dim x embedding_dim]

        # ---- On-diagonal: want C[i, i] ~ 1 ----
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()

        # ---- Off-diagonal: want C[i, j] ~ 0 (i != j) ----
        off_diag = off_diagonal(c).pow_(2).sum()

        loss = on_diag + self.lambda_offdiag * off_diag
        return loss

class VICRegLoss(nn.Module):
    """
    Implementation of VICReg (Variance-Invariance-Covariance Regularization)
    Reference: https://arxiv.org/abs/2105.04906
    """
    def __init__(self, sim_weight=25.0, var_weight=25.0, cov_weight=1.0, std_target=1.0, eps=1e-4):
        """
        Args:
            sim_weight: weight for the invariance (similarity) term.
            var_weight: weight for the variance term.
            cov_weight: weight for the covariance term.
            std_target: the target standard deviation in the variance term (typically 1.0).
            eps: small epsilon to avoid numerical issues.
        """
        super().__init__()
        self.sim_weight = sim_weight
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.std_target = std_target
        self.eps = eps

    def forward(self, z1, z2):
        """
        Compute the VICReg loss given two batches of embeddings z1, z2.

        z1, z2 are of shape [batch_size, embedding_dim].
        """

        # ---------------------
        # 1. Invariance (MSE)
        # ---------------------
        invariance_loss = F.mse_loss(z1, z2, reduction='mean')

        # ---------------------
        # 2. Variance
        # ---------------------
        # For each view, compute the standard deviation of each dimension across batch
        z1 = z1 - z1.mean(dim=0, keepdim=True)
        z2 = z2 - z2.mean(dim=0, keepdim=True)

        std_z1 = torch.sqrt(z1.var(dim=0, unbiased=False) + self.eps)
        std_z2 = torch.sqrt(z2.var(dim=0, unbiased=False) + self.eps)

        # Hinge loss on the standard deviation: penalize if std < std_target
        # "relu(std_target - std_z)" ensures that any dimension below std_target is penalized
        var_loss = torch.mean(F.relu(self.std_target - std_z1)) + torch.mean(F.relu(self.std_target - std_z2))

        # ---------------------
        # 3. Covariance
        # ---------------------
        # For each view, compute the cross-covariance across batch
        batch_size, dim = z1.shape

        cov_z1 = (z1.T @ z1) / (batch_size - 1)  # [dim, dim]
        cov_z2 = (z2.T @ z2) / (batch_size - 1)  # [dim, dim]

        # We only penalize off-diagonal terms
        off_diag_z1 = off_diagonal(cov_z1)
        off_diag_z2 = off_diagonal(cov_z2)

        cov_loss = off_diag_z1.pow_(2).sum() / dim + off_diag_z2.pow_(2).sum() / dim

        # ---------------------
        # Combine the terms
        # ---------------------
        loss = (self.sim_weight * invariance_loss
                + self.var_weight * var_loss
                + self.cov_weight * cov_loss)

        return loss
    
# Define the Autoencoder model
class scRNASeqE_simCLR(pl.LightningModule):
    def __init__(self, PARAMETERS, cell_type_mu_sigma, global_mu_sigma, cell_type_msg_mu_sigma,
                 cell_type_lsg_mu_sigma, most_significant_genes_dict, least_significant_genes_dict,
                 gene_networks, gene_names, code_to_celltype, celltype_to_code, gene_name_to_index,
                 index_to_gene_name, gene_dispersions, 
                 temperature=0.5,
                 dropout_rate_DO=0.5,
                 dropout_rate_gSS=0.5,
                 augmentations_pipeline=[]):
        super(scRNASeqE_simCLR, self).__init__()

        self.PARAMETERS = PARAMETERS
        self.encoder = scRNASeqEncoder(PARAMETERS)
        self.projector = scRNASeqProjectionHead(PARAMETERS)
        self.temperature = temperature  # Can be added to PARAMETERS if needed
        self.contrastive_loss_fn = ContrastiveLoss(temperature=self.temperature)

        # Assign precomputed variables
        self.cell_type_mu_sigma = cell_type_mu_sigma
        self.global_mu_sigma = global_mu_sigma
        self.cell_type_msg_mu_sigma = cell_type_msg_mu_sigma
        self.cell_type_lsg_mu_sigma = cell_type_lsg_mu_sigma
        self.most_significant_genes_dict = most_significant_genes_dict
        self.least_significant_genes_dict = least_significant_genes_dict
        self.gene_networks = gene_networks
        self.gene_names = gene_names
        self.code_to_celltype = code_to_celltype
        self.celltype_to_code = celltype_to_code
        self.gene_name_to_index = gene_name_to_index
        self.index_to_gene_name = index_to_gene_name
        self.gene_dispersions = gene_dispersions

        self.dropout_rate_DO = dropout_rate_DO
        self.dropout_rate_gSS = dropout_rate_gSS
        self.augmentations_pipeline = augmentations_pipeline
    
        self.augmentations_used = []
        for step in self.augmentations_pipeline:
            # global augmentations_used
            short_name = AUGMENTATION_SHORT_NAMES[step['fn']]
            self.augmentations_used.append(short_name)

    def forward(self, X):
        latent = self.encoder(X)  # Encoding
        return self.projector(latent)  # Projecting
    
    def augmenter(self, X, y):
        """
        Applies a series of augmentations from self.augmentations_pipeline to X.
        y are cell types, used if augmentation function needs them.
        """
        aug_out = X
        for step in self.augmentations_pipeline:
            fn = step['fn']
            needs_cell_types = step.get('needs_cell_types', False)
            kwargs_config = step.get('kwargs', {})

            if callable(kwargs_config): # This will handle augmentations with lambda kwargs; i.e. those that rely on model's arguments
                fn_kwargs = kwargs_config(self)
            else:
                fn_kwargs = kwargs_config
            
            if needs_cell_types:
                aug_out = fn(expression_matrix=aug_out, cell_types=y, **fn_kwargs)
            else:
                aug_out = fn(expression_matrix=aug_out, **fn_kwargs)
        return aug_out

    def training_step(self, batch, batch_idx):
        X, y = batch
        X = X.to(self.device)
        y = y.to(self.device)
        aug_0 = self.augmenter(X, y)
        aug_1 = self.augmenter(X, y)

        projection_head_latents_0 = self.forward(aug_0)
        projection_head_latents_1 = self.forward(aug_1)

        loss = self.contrastive_loss_fn(projection_head_latents_0, projection_head_latents_1)
        self.log('train_loss_simCLR', loss.item(), on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        aug_0 = self.augmenter(X, y)
        aug_1 = self.augmenter(X, y)

        projection_head_latents_0 = self.forward(aug_0)
        projection_head_latents_1 = self.forward(aug_1)

        loss = self.contrastive_loss_fn(projection_head_latents_0, projection_head_latents_1)
        self.log('val_loss_simCLR', loss.item(), on_step=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())
    
    
class scRNASeqE_Barlow(pl.LightningModule):
    def __init__(
        self,
        PARAMETERS,
        cell_type_mu_sigma,
        global_mu_sigma,
        cell_type_msg_mu_sigma,
        cell_type_lsg_mu_sigma,
        most_significant_genes_dict,
        least_significant_genes_dict,
        gene_networks,
        gene_names,
        code_to_celltype,
        celltype_to_code,
        gene_name_to_index,
        index_to_gene_name,
        gene_dispersions,
        lambda_offdiag=5e-3,
        dropout_rate_DO=0.5,
        dropout_rate_gSS=0.5,
        augmentations_pipeline=[],
    ):
        super().__init__()
        self.PARAMETERS = PARAMETERS
        self.save_hyperparameters(ignore=['augmentations_pipeline'])
        self.PARAMETERS = PARAMETERS

        # Encoder & Projector
        self.encoder = scRNASeqEncoder(PARAMETERS)
        self.projector = scRNASeqProjectionHead(PARAMETERS)

        # Barlow Twins loss
        self.lambda_offdiag = lambda_offdiag
        self.barlow_loss_fn = BarlowTwinsLoss(lambda_offdiag=self.lambda_offdiag)

        # Store precomputed variables
        self.cell_type_mu_sigma = cell_type_mu_sigma
        self.global_mu_sigma = global_mu_sigma
        self.cell_type_msg_mu_sigma = cell_type_msg_mu_sigma
        self.cell_type_lsg_mu_sigma = cell_type_lsg_mu_sigma
        self.most_significant_genes_dict = most_significant_genes_dict
        self.least_significant_genes_dict = least_significant_genes_dict
        self.gene_networks = gene_networks
        self.gene_names = gene_names
        self.code_to_celltype = code_to_celltype
        self.celltype_to_code = celltype_to_code
        self.gene_name_to_index = gene_name_to_index
        self.index_to_gene_name = index_to_gene_name
        self.gene_dispersions = gene_dispersions

        self.dropout_rate_DO = dropout_rate_DO
        self.dropout_rate_gSS = dropout_rate_gSS
        self.augmentations_pipeline = augmentations_pipeline

        # Record augmentations used
        self.augmentations_used = []
        for step in self.augmentations_pipeline:
            short_name = AUGMENTATION_SHORT_NAMES[step['fn']]
            self.augmentations_used.append(short_name)

    def forward(self, X):
        latent = self.encoder(X)
        z = self.projector(latent)
        return z

    def augmenter(self, X, y):
        """
        Applies a series of augmentations from self.augmentations_pipeline to X.
        y are cell types, used if augmentation function needs them.
        """
        aug_out = X
        for step in self.augmentations_pipeline:
            fn = step['fn']
            needs_cell_types = step.get('needs_cell_types', False)
            kwargs_config = step.get('kwargs', {})

            if callable(kwargs_config): # This will handle augmentations with lambda kwargs; i.e. those that rely on model's arguments
                fn_kwargs = kwargs_config(self)
            else:
                fn_kwargs = kwargs_config
            
            if needs_cell_types:
                aug_out = fn(expression_matrix=aug_out, cell_types=y, **fn_kwargs)
            else:
                aug_out = fn(expression_matrix=aug_out, **fn_kwargs)
        return aug_out

    def training_step(self, batch, batch_idx):
        X, y = batch
        X = X.to(self.device)
        y = y.to(self.device)

        # two augmentations
        aug_0 = self.augmenter(X, y)
        aug_1 = self.augmenter(X, y)

        z0 = self.forward(aug_0)
        z1 = self.forward(aug_1)

        loss = self.barlow_loss_fn(z0, z1)
        self.log("train_loss_barlow", loss.item(), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        X = X.to(self.device)
        y = y.to(self.device)

        aug_0 = self.augmenter(X, y)
        aug_1 = self.augmenter(X, y)

        z0 = self.forward(aug_0)
        z1 = self.forward(aug_1)

        loss = self.barlow_loss_fn(z0, z1)
        self.log("val_loss_barlow", loss.item(), on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())

class scRNASeqE_VICReg(pl.LightningModule):
    def __init__(self, PARAMETERS, 
                 cell_type_mu_sigma, global_mu_sigma, 
                 cell_type_msg_mu_sigma, cell_type_lsg_mu_sigma, 
                 most_significant_genes_dict, least_significant_genes_dict,
                 gene_networks, gene_names, code_to_celltype, celltype_to_code, 
                 gene_name_to_index, index_to_gene_name, gene_dispersions,
                 sim_weight=25.0,
                 var_weight=25.0,
                 cov_weight=1.0,
                 std_target=1.0,
                 dropout_rate_DO=0.5,
                 dropout_rate_gSS=0.5,
                 sigma_fill=0.1,
                 eps=1e-4,
                 augmentations_pipeline=[],
                 similarity_matrix=None):
        """
        Replace or adjust default hyperparameters to your liking.
        """
        super(scRNASeqE_VICReg, self).__init__()
        self.save_hyperparameters(ignore=['augmentations_pipeline'])
        self.PARAMETERS = PARAMETERS

        # Define your encoder and projector
        self.encoder = scRNASeqEncoder(PARAMETERS)
        self.projector = scRNASeqProjectionHead(PARAMETERS)

        # # Create an instance of the VICReg loss
        # self.vicreg_loss_fn = VICRegLoss(
        #     sim_weight=sim_weight,
        #     var_weight=var_weight,
        #     cov_weight=cov_weight,
        #     std_target=std_target,
        #     eps=eps
        # )

        ## using similarity loss now!
        self.similarity_loss = FullSimilarityMatrixLoss(target_similarity=similarity_matrix, mode="mse")



        # Assign precomputed variables (used in augmentations)
        self.cell_type_mu_sigma = cell_type_mu_sigma
        self.global_mu_sigma = global_mu_sigma
        self.cell_type_msg_mu_sigma = cell_type_msg_mu_sigma
        self.cell_type_lsg_mu_sigma = cell_type_lsg_mu_sigma
        self.most_significant_genes_dict = most_significant_genes_dict
        self.least_significant_genes_dict = least_significant_genes_dict
        self.gene_networks = gene_networks
        self.gene_names = gene_names
        self.code_to_celltype = code_to_celltype
        self.celltype_to_code = celltype_to_code
        self.gene_name_to_index = gene_name_to_index
        self.index_to_gene_name = index_to_gene_name
        self.gene_dispersions = gene_dispersions

        self.dropout_rate_DO = dropout_rate_DO
        self.dropout_rate_gSS = dropout_rate_gSS
        self.sigma_fill = sigma_fill
        self.augmentations_pipeline = augmentations_pipeline

        self.augmentations_used = []
        for step in self.augmentations_pipeline:
            # global augmentations_used
            short_name = AUGMENTATION_SHORT_NAMES[step['fn']]
            self.augmentations_used.append(short_name)

    def forward(self, X):
        # X -> encoder -> projector -> embeddings (z)
        latent = self.encoder(X)
        z = self.projector(latent)
        return z

    def augmenter(self, X, y):
        """
        Applies a series of augmentations from self.augmentations_pipeline to X.
        y are cell types, used if augmentation function needs them.
        """
        aug_out = X
        for step in self.augmentations_pipeline:
            fn = step['fn']
            needs_cell_types = step.get('needs_cell_types', False)
            kwargs_config = step.get('kwargs', {})

            if callable(kwargs_config): # This will handle augmentations with lambda kwargs; i.e. those that rely on model's arguments
                fn_kwargs = kwargs_config(self)
            else:
                fn_kwargs = kwargs_config
            
            if needs_cell_types:
                aug_out = fn(expression_matrix=aug_out, cell_types=y, **fn_kwargs)
            else:
                aug_out = fn(expression_matrix=aug_out, **fn_kwargs)
        return aug_out

    def training_step(self, batch, batch_idx):
        # X, y = batch
        # X = X.to(self.device)
        # y = y.to(self.device)

        # # Generate two augmentations of the same batch
        # aug_0 = self.augmenter(X, y)
        # aug_1 = self.augmenter(X, y)

        # # Forward pass through encoder + projector
        # z0 = self.forward(aug_0)
        # z1 = self.forward(aug_1)

        # # Compute VICReg loss
        # loss = self.vicreg_loss_fn(z0, z1)

        # self.log("train_loss_vicreg", loss.item(), on_step=True, on_epoch=True)
        # return loss


        X, y = batch
        X = X.to(self.device)
        y = y.to(self.device)

        # Only need one encoding with similarity matrix
        aug = self.augmenter(X, y)
        z = self.forward(aug)

        # Compute similarity matrix loss
        loss = self.similarity_loss(z, y)

        self.log("train_loss_similarity", loss.item(), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # X, y = batch
        # X = X.to(self.device)
        # y = y.to(self.device)

        # # Augment
        # aug_0 = self.augmenter(X, y)
        # aug_1 = self.augmenter(X, y)

        # # Forward pass
        # z0 = self.forward(aug_0)
        # z1 = self.forward(aug_1)

        # # VICReg loss
        # loss = self.vicreg_loss_fn(z0, z1)

        # self.log("val_loss_vicreg", loss.item(), on_step=True, on_epoch=True)
        # return loss
        X, y = batch
        X = X.to(self.device)
        y = y.to(self.device)

        # Only need one encoding with similarity matrix
        aug = self.augmenter(X, y)
        z = self.forward(aug)

        # Compute similarity matrix loss
        loss = self.similarity_loss(z, y)

        self.log("val_loss_similarity", loss.item(), on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())

class scRNASeqE_VICRegLarge(pl.LightningModule):
    def __init__(self, PARAMETERS, 
                 cell_type_mu_sigma, global_mu_sigma, 
                 cell_type_msg_mu_sigma, cell_type_lsg_mu_sigma, 
                 most_significant_genes_dict, least_significant_genes_dict,
                 gene_networks, gene_names, code_to_celltype, celltype_to_code, 
                 gene_name_to_index, index_to_gene_name, gene_dispersions,
                 sim_weight=25.0,
                 var_weight=25.0,
                 cov_weight=1.0,
                 std_target=1.0,
                 dropout_rate_DO=0.5,
                 dropout_rate_gSS=0.5,
                 sigma_fill=0.1,
                 eps=1e-4,
                 augmentations_pipeline=[],
                 similarity_matrix=None):
        """
        Replace or adjust default hyperparameters to your liking.
        """
        super(scRNASeqE_VICRegLarge, self).__init__()
        self.save_hyperparameters(ignore=['augmentations_pipeline'])
        self.PARAMETERS = PARAMETERS

        # Define your encoder and projector
        self.encoder = scRNASeqEncoderLarge(PARAMETERS)
        self.projector = scRNASeqProjectionHead(PARAMETERS)

        # # Create an instance of the VICReg loss
        # self.vicreg_loss_fn = VICRegLoss(
        #     sim_weight=sim_weight,
        #     var_weight=var_weight,
        #     cov_weight=cov_weight,
        #     std_target=std_target,
        #     eps=eps
        # )

        ## using similarity loss now!
        self.similarity_loss = FullSimilarityMatrixLoss(target_similarity=similarity_matrix, mode="mse")



        # Assign precomputed variables (used in augmentations)
        self.cell_type_mu_sigma = cell_type_mu_sigma
        self.global_mu_sigma = global_mu_sigma
        self.cell_type_msg_mu_sigma = cell_type_msg_mu_sigma
        self.cell_type_lsg_mu_sigma = cell_type_lsg_mu_sigma
        self.most_significant_genes_dict = most_significant_genes_dict
        self.least_significant_genes_dict = least_significant_genes_dict
        self.gene_networks = gene_networks
        self.gene_names = gene_names
        self.code_to_celltype = code_to_celltype
        self.celltype_to_code = celltype_to_code
        self.gene_name_to_index = gene_name_to_index
        self.index_to_gene_name = index_to_gene_name
        self.gene_dispersions = gene_dispersions

        self.dropout_rate_DO = dropout_rate_DO
        self.dropout_rate_gSS = dropout_rate_gSS
        self.sigma_fill = sigma_fill
        self.augmentations_pipeline = augmentations_pipeline

        self.augmentations_used = []
        for step in self.augmentations_pipeline:
            # global augmentations_used
            short_name = AUGMENTATION_SHORT_NAMES[step['fn']]
            self.augmentations_used.append(short_name)

    def forward(self, X):
        # X -> encoder -> projector -> embeddings (z)
        latent = self.encoder(X)
        z = self.projector(latent)
        return z

    def augmenter(self, X, y):
        """
        Applies a series of augmentations from self.augmentations_pipeline to X.
        y are cell types, used if augmentation function needs them.
        """
        aug_out = X
        for step in self.augmentations_pipeline:
            fn = step['fn']
            needs_cell_types = step.get('needs_cell_types', False)
            kwargs_config = step.get('kwargs', {})

            if callable(kwargs_config): # This will handle augmentations with lambda kwargs; i.e. those that rely on model's arguments
                fn_kwargs = kwargs_config(self)
            else:
                fn_kwargs = kwargs_config
            
            if needs_cell_types:
                aug_out = fn(expression_matrix=aug_out, cell_types=y, **fn_kwargs)
            else:
                aug_out = fn(expression_matrix=aug_out, **fn_kwargs)
        return aug_out

    def training_step(self, batch, batch_idx):
        # X, y = batch
        # X = X.to(self.device)
        # y = y.to(self.device)

        # # Generate two augmentations of the same batch
        # aug_0 = self.augmenter(X, y)
        # aug_1 = self.augmenter(X, y)

        # # Forward pass through encoder + projector
        # z0 = self.forward(aug_0)
        # z1 = self.forward(aug_1)

        # # Compute VICReg loss
        # loss = self.vicreg_loss_fn(z0, z1)

        # self.log("train_loss_vicreg", loss.item(), on_step=True, on_epoch=True)
        # return loss
        X, y = batch
        X = X.to(self.device)
        y = y.to(self.device)

        # Only need one encoding with similarity matrix
        aug = self.augmenter(X, y)
        z = self.forward(aug)

        # Compute similarity matrix loss
        loss = self.similarity_loss(z, y)

        self.log("train_loss_similarity", loss.item(), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # X, y = batch
        # X = X.to(self.device)
        # y = y.to(self.device)

        # # Augment
        # aug_0 = self.augmenter(X, y)
        # aug_1 = self.augmenter(X, y)

        # # Forward pass
        # z0 = self.forward(aug_0)
        # z1 = self.forward(aug_1)

        # # VICReg loss
        # loss = self.vicreg_loss_fn(z0, z1)

        # self.log("val_loss_vicreg", loss.item(), on_step=True, on_epoch=True)
        # return loss
        X, y = batch
        X = X.to(self.device)
        y = y.to(self.device)

        # Only need one encoding with similarity matrix
        aug = self.augmenter(X, y)
        z = self.forward(aug)

        # Compute similarity matrix loss
        loss = self.similarity_loss(z, y)

        self.log("val_loss_similarity", loss.item(), on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())

class scRNASeqE_VICRegSiam(pl.LightningModule):
    def __init__(self, PARAMETERS, 
                 cell_type_mu_sigma, global_mu_sigma, 
                 cell_type_msg_mu_sigma, cell_type_lsg_mu_sigma, 
                 most_significant_genes_dict, least_significant_genes_dict,
                 gene_networks, gene_names, code_to_celltype, celltype_to_code, 
                 gene_name_to_index, index_to_gene_name, gene_dispersions,
                 sim_weight=25.0,
                 var_weight=25.0,
                 cov_weight=1.0,
                 std_target=1.0,
                 dropout_rate_DO=0.5,
                 dropout_rate_gSS=0.5,
                 sigma_fill=0.1,
                 eps=1e-4,
                 augmentations_pipeline=[],
                 similarity_matrix=None):
        """
        Replace or adjust default hyperparameters to your liking.
        """
        super(scRNASeqE_VICRegSiam, self).__init__()
        self.save_hyperparameters(ignore=['augmentations_pipeline'])
        self.PARAMETERS = PARAMETERS

        # Define your encoder and projector
        self.encoder1 = scRNASeqEncoder(PARAMETERS)
        self.encoder2 = scRNASeqEncoder(PARAMETERS)
        self.projector1 = scRNASeqProjectionHead(PARAMETERS)
        self.projector2 = scRNASeqProjectionHead(PARAMETERS)

        # # Create an instance of the VICReg loss
        # self.vicreg_loss_fn = VICRegLoss(
        #     sim_weight=sim_weight,
        #     var_weight=var_weight,
        #     cov_weight=cov_weight,
        #     std_target=std_target,
        #     eps=eps
        # )

        ## using similarity loss now!
        self.similarity_loss = FullSimilarityMatrixLoss(target_similarity=similarity_matrix, mode="mse")



        # Assign precomputed variables (used in augmentations)
        self.cell_type_mu_sigma = cell_type_mu_sigma
        self.global_mu_sigma = global_mu_sigma
        self.cell_type_msg_mu_sigma = cell_type_msg_mu_sigma
        self.cell_type_lsg_mu_sigma = cell_type_lsg_mu_sigma
        self.most_significant_genes_dict = most_significant_genes_dict
        self.least_significant_genes_dict = least_significant_genes_dict
        self.gene_networks = gene_networks
        self.gene_names = gene_names
        self.code_to_celltype = code_to_celltype
        self.celltype_to_code = celltype_to_code
        self.gene_name_to_index = gene_name_to_index
        self.index_to_gene_name = index_to_gene_name
        self.gene_dispersions = gene_dispersions

        self.dropout_rate_DO = dropout_rate_DO
        self.dropout_rate_gSS = dropout_rate_gSS
        self.sigma_fill = sigma_fill
        self.augmentations_pipeline = augmentations_pipeline

        self.augmentations_used = []
        for step in self.augmentations_pipeline:
            # global augmentations_used
            short_name = AUGMENTATION_SHORT_NAMES[step['fn']]
            self.augmentations_used.append(short_name)

    def forward1(self, X):
        # X -> encoder -> projector -> embeddings (z)
        latent = self.encoder1(X)
        z = self.projector1(latent)
        return z
    
    def forward2(self, X):
        # X -> encoder -> projector -> embeddings (z)
        latent = self.encoder2(X)
        z = self.projector2(latent)
        return z

    def augmenter(self, X, y):
        """
        Applies a series of augmentations from self.augmentations_pipeline to X.
        y are cell types, used if augmentation function needs them.
        """
        aug_out = X
        for step in self.augmentations_pipeline:
            fn = step['fn']
            needs_cell_types = step.get('needs_cell_types', False)
            kwargs_config = step.get('kwargs', {})

            if callable(kwargs_config): # This will handle augmentations with lambda kwargs; i.e. those that rely on model's arguments
                fn_kwargs = kwargs_config(self)
            else:
                fn_kwargs = kwargs_config
            
            if needs_cell_types:
                aug_out = fn(expression_matrix=aug_out, cell_types=y, **fn_kwargs)
            else:
                aug_out = fn(expression_matrix=aug_out, **fn_kwargs)
        return aug_out

    def training_step(self, batch, batch_idx):
        # X, y = batch
        # X = X.to(self.device)
        # y = y.to(self.device)

        # # Generate two augmentations of the same batch
        # aug_0 = self.augmenter(X, y)
        # aug_1 = self.augmenter(X, y)

        # # Forward pass through encoder + projector
        # z0 = self.forward1(aug_0)
        # z1 = self.forward2(aug_1)

        # # Compute VICReg loss
        # loss = self.vicreg_loss_fn(z0, z1)

        # self.log("train_loss_vicreg", loss.item(), on_step=True, on_epoch=True)
        # return loss
        X, y = batch
        X = X.to(self.device)
        y = y.to(self.device)

        # Only need one encoding with similarity matrix
        aug = self.augmenter(X, y)
        z = self.forward(aug)

        # Compute similarity matrix loss
        loss = self.similarity_loss(z, y)

        self.log("train_loss_similarity", loss.item(), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # X, y = batch
        # X = X.to(self.device)
        # y = y.to(self.device)

        # # Augment
        # aug_0 = self.augmenter(X, y)
        # aug_1 = self.augmenter(X, y)

        # # Forward pass
        # z0 = self.forward1(aug_0)
        # z1 = self.forward2(aug_1)

        # # VICReg loss
        # loss = self.vicreg_loss_fn(z0, z1)

        # self.log("val_loss_vicreg", loss.item(), on_step=True, on_epoch=True)
        # return loss
        X, y = batch
        X = X.to(self.device)
        y = y.to(self.device)

        # Only need one encoding with similarity matrix
        aug = self.augmenter(X, y)
        z = self.forward(aug)

        # Compute similarity matrix loss
        loss = self.similarity_loss(z, y)

        self.log("val_loss_similarity", loss.item(), on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())
    
class scRNASeqE_VICRegExpander(pl.LightningModule):
    def __init__(self, PARAMETERS, 
                 cell_type_mu_sigma, global_mu_sigma, 
                 cell_type_msg_mu_sigma, cell_type_lsg_mu_sigma, 
                 most_significant_genes_dict, least_significant_genes_dict,
                 gene_networks, gene_names, code_to_celltype, celltype_to_code, 
                 gene_name_to_index, index_to_gene_name, gene_dispersions,
                 sim_weight=25.0,
                 var_weight=25.0,
                 cov_weight=1.0,
                 std_target=1.0,
                 dropout_rate_DO=0.5,
                 dropout_rate_gSS=0.5,
                 sigma_fill=0.1,
                 eps=1e-4,
                 augmentations_pipeline=[],
                 similarity_matrix=None):
        """
        Replace or adjust default hyperparameters to your liking.
        """
        super(scRNASeqE_VICRegExpander, self).__init__()
        self.save_hyperparameters(ignore=['augmentations_pipeline'])
        self.PARAMETERS = PARAMETERS

        # Define your encoder and projector
        self.encoder = scRNASeqEncoder(PARAMETERS)
        self.projector = scRNASeqProjectionHeadExpander(PARAMETERS)

        # # Create an instance of the VICReg loss
        # self.vicreg_loss_fn = VICRegLoss(
        #     sim_weight=sim_weight,
        #     var_weight=var_weight,
        #     cov_weight=cov_weight,
        #     std_target=std_target,
        #     eps=eps
        # )

        ## using similarity loss now!
        self.similarity_loss = FullSimilarityMatrixLoss(target_similarity=similarity_matrix, mode="mse")




        # Assign precomputed variables (used in augmentations)
        self.cell_type_mu_sigma = cell_type_mu_sigma
        self.global_mu_sigma = global_mu_sigma
        self.cell_type_msg_mu_sigma = cell_type_msg_mu_sigma
        self.cell_type_lsg_mu_sigma = cell_type_lsg_mu_sigma
        self.most_significant_genes_dict = most_significant_genes_dict
        self.least_significant_genes_dict = least_significant_genes_dict
        self.gene_networks = gene_networks
        self.gene_names = gene_names
        self.code_to_celltype = code_to_celltype
        self.celltype_to_code = celltype_to_code
        self.gene_name_to_index = gene_name_to_index
        self.index_to_gene_name = index_to_gene_name
        self.gene_dispersions = gene_dispersions

        self.dropout_rate_DO = dropout_rate_DO
        self.dropout_rate_gSS = dropout_rate_gSS
        self.sigma_fill = sigma_fill
        self.augmentations_pipeline = augmentations_pipeline

        self.augmentations_used = []
        for step in self.augmentations_pipeline:
            # global augmentations_used
            short_name = AUGMENTATION_SHORT_NAMES[step['fn']]
            self.augmentations_used.append(short_name)

    def forward(self, X):
        # X -> encoder -> projector -> embeddings (z)
        latent = self.encoder(X)
        z = self.projector(latent)
        return z

    def augmenter(self, X, y):
        """
        Applies a series of augmentations from self.augmentations_pipeline to X.
        y are cell types, used if augmentation function needs them.
        """
        aug_out = X
        for step in self.augmentations_pipeline:
            fn = step['fn']
            needs_cell_types = step.get('needs_cell_types', False)
            kwargs_config = step.get('kwargs', {})

            if callable(kwargs_config): # This will handle augmentations with lambda kwargs; i.e. those that rely on model's arguments
                fn_kwargs = kwargs_config(self)
            else:
                fn_kwargs = kwargs_config
            
            if needs_cell_types:
                aug_out = fn(expression_matrix=aug_out, cell_types=y, **fn_kwargs)
            else:
                aug_out = fn(expression_matrix=aug_out, **fn_kwargs)
        return aug_out

    def training_step(self, batch, batch_idx):
        # X, y = batch
        # # X = X.to(self.device)
        # # y = y.to(self.device)

        # # Generate two augmentations of the same batch
        # aug_0 = self.augmenter(X, y)
        # aug_1 = self.augmenter(X, y)

        # # Forward pass through encoder + projector
        # z0 = self.forward(aug_0)
        # z1 = self.forward(aug_1)

        # # Compute VICReg loss
        # loss = self.vicreg_loss_fn(z0, z1)

        # self.log("train_loss_vicreg", loss.item(), on_step=True, on_epoch=True)
        # return loss

        X, y = batch
        # X = X.to(self.device)
        # y = y.to(self.device)

        # Only need one encoding with similarity matrix
        aug = self.augmenter(X, y)
        z = self.forward(aug)

        # Compute similarity matrix loss
        loss = self.similarity_loss(z, y)

        self.log("train_loss_similarity", loss.item(), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # X, y = batch
        # # X = X.to(self.device)
        # # y = y.to(self.device)

        # # Augment
        # aug_0 = self.augmenter(X, y)
        # aug_1 = self.augmenter(X, y)

        # # Forward pass
        # z0 = self.forward(aug_0)
        # z1 = self.forward(aug_1)

        # # VICReg loss
        # loss = self.vicreg_loss_fn(z0, z1)

        # self.log("val_loss_vicreg", loss.item(), on_step=True, on_epoch=True)
        # return loss
        X, y = batch
        # X = X.to(self.device)
        # y = y.to(self.device)

        # Only need one encoding with similarity matrix
        aug = self.augmenter(X, y)
        z = self.forward(aug)

        # Compute similarity matrix loss
        loss = self.similarity_loss(z, y)

        self.log("val_loss_similarity", loss.item(), on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())

class scRNASeqE_VICRegExpanderLarge(pl.LightningModule):
    def __init__(self, PARAMETERS, 
                 cell_type_mu_sigma, global_mu_sigma, 
                 cell_type_msg_mu_sigma, cell_type_lsg_mu_sigma, 
                 most_significant_genes_dict, least_significant_genes_dict,
                 gene_networks, gene_names, code_to_celltype, celltype_to_code, 
                 gene_name_to_index, index_to_gene_name, gene_dispersions,
                 sim_weight=25.0,
                 var_weight=25.0,
                 cov_weight=1.0,
                 std_target=1.0,
                 dropout_rate_DO=0.5,
                 dropout_rate_gSS=0.5,
                 sigma_fill=0.1,
                 eps=1e-4,
                 augmentations_pipeline=[],
                 similarity_matrix=None):
        """
        Replace or adjust default hyperparameters to your liking.
        """
        super(scRNASeqE_VICRegExpanderLarge, self).__init__()
        self.save_hyperparameters(ignore=['augmentations_pipeline'])
        self.PARAMETERS = PARAMETERS

        # Define your encoder and projector
        self.encoder = scRNASeqEncoderLarge(PARAMETERS)
        self.projector = scRNASeqProjectionHeadExpander(PARAMETERS)

        # # Create an instance of the VICReg loss
        # self.vicreg_loss_fn = VICRegLoss(
        #     sim_weight=sim_weight,
        #     var_weight=var_weight,
        #     cov_weight=cov_weight,
        #     std_target=std_target,
        #     eps=eps
        # )

        ## using similarity loss now!
        self.similarity_loss = FullSimilarityMatrixLoss(target_similarity=similarity_matrix, mode="mse")



        # Assign precomputed variables (used in augmentations)
        self.cell_type_mu_sigma = cell_type_mu_sigma
        self.global_mu_sigma = global_mu_sigma
        self.cell_type_msg_mu_sigma = cell_type_msg_mu_sigma
        self.cell_type_lsg_mu_sigma = cell_type_lsg_mu_sigma
        self.most_significant_genes_dict = most_significant_genes_dict
        self.least_significant_genes_dict = least_significant_genes_dict
        self.gene_networks = gene_networks
        self.gene_names = gene_names
        self.code_to_celltype = code_to_celltype
        self.celltype_to_code = celltype_to_code
        self.gene_name_to_index = gene_name_to_index
        self.index_to_gene_name = index_to_gene_name
        self.gene_dispersions = gene_dispersions

        self.dropout_rate_DO = dropout_rate_DO
        self.dropout_rate_gSS = dropout_rate_gSS
        self.sigma_fill = sigma_fill
        self.augmentations_pipeline = augmentations_pipeline

        self.augmentations_used = []
        for step in self.augmentations_pipeline:
            # global augmentations_used
            short_name = AUGMENTATION_SHORT_NAMES[step['fn']]
            self.augmentations_used.append(short_name)

    def forward(self, X):
        # X -> encoder -> projector -> embeddings (z)
        latent = self.encoder(X)
        z = self.projector(latent)
        return z

    def augmenter(self, X, y):
        """
        Applies a series of augmentations from self.augmentations_pipeline to X.
        y are cell types, used if augmentation function needs them.
        """
        aug_out = X
        for step in self.augmentations_pipeline:
            fn = step['fn']
            needs_cell_types = step.get('needs_cell_types', False)
            kwargs_config = step.get('kwargs', {})

            if callable(kwargs_config): # This will handle augmentations with lambda kwargs; i.e. those that rely on model's arguments
                fn_kwargs = kwargs_config(self)
            else:
                fn_kwargs = kwargs_config
            
            if needs_cell_types:
                aug_out = fn(expression_matrix=aug_out, cell_types=y, **fn_kwargs)
            else:
                aug_out = fn(expression_matrix=aug_out, **fn_kwargs)
        return aug_out

    def training_step(self, batch, batch_idx):
        # X, y = batch
        # # X = X.to(self.device)
        # # y = y.to(self.device)

        # # Generate two augmentations of the same batch
        # aug_0 = self.augmenter(X, y)
        # aug_1 = self.augmenter(X, y)

        # # Forward pass through encoder + projector
        # z0 = self.forward(aug_0)
        # z1 = self.forward(aug_1)

        # # Compute VICReg loss
        # loss = self.vicreg_loss_fn(z0, z1)

        # self.log("train_loss_vicreg", loss.item(), on_step=True, on_epoch=True)
        # return loss
        X, y = batch
        # X = X.to(self.device)
        # y = y.to(self.device)

        # Only need one encoding with similarity matrix
        aug = self.augmenter(X, y)
        z = self.forward(aug)

        # Compute similarity matrix loss
        loss = self.similarity_loss(z, y)

        self.log("train_loss_similarity", loss.item(), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # X, y = batch
        # # X = X.to(self.device)
        # # y = y.to(self.device)

        # # Augment
        # aug_0 = self.augmenter(X, y)
        # aug_1 = self.augmenter(X, y)

        # # Forward pass
        # z0 = self.forward(aug_0)
        # z1 = self.forward(aug_1)

        # # VICReg loss
        # loss = self.vicreg_loss_fn(z0, z1)

        # self.log("val_loss_vicreg", loss.item(), on_step=True, on_epoch=True)
        # return loss
        X, y = batch
        # X = X.to(self.device)
        # y = y.to(self.device)

        # Only need one encoding with similarity matrix
        aug = self.augmenter(X, y)
        z = self.forward(aug)

        # Compute similarity matrix loss
        loss = self.similarity_loss(z, y)

        self.log("val_loss_similarity", loss.item(), on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters())




## similarity matrix loss
class FullSimilarityMatrixLoss(nn.Module):
    def __init__(self, target_similarity, mode="mse"):
        super().__init__()
        if isinstance(target_similarity, np.ndarray):
            target_similarity = torch.tensor(target_similarity, dtype=torch.float32)
        self.target_similarity = target_similarity  # shape: (num_classes, num_classes)
        assert mode in {"mse", "kl"}, "mode must be 'mse' or 'kl'"
        self.mode = mode

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        """
        z: shape (B, D) - batch of projected embeddings
        y: shape (B,) - class labels (categorical codes)
        """
        z = F.normalize(z, dim=1)
        sim_matrix = torch.matmul(z, z.T)  # shape: (B, B)

        # Move y to CPU for indexing
        y_cpu = y.detach().cpu()
        target = self.target_similarity[y_cpu][:, y_cpu].to(z.device)  # shape: (B, B)

        if self.mode == "mse":
            return F.mse_loss(sim_matrix, target)
        elif self.mode == "kl":
            sim_log_probs = F.log_softmax(sim_matrix, dim=1)
            target_probs = F.softmax(target, dim=1)
            return F.kl_div(sim_log_probs, target_probs, reduction="batchmean")