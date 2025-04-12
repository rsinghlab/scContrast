import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl


class scRNASeqEncoder(pl.LightningModule):
    def __init__(self, num_genes, PARAMETERS):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_genes, 256),
            nn.BatchNorm1d(256),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(128, PARAMETERS['latent_dimension'])
        )

    def forward(self, x):
        return self.encoder(x)

class scRNASeqDecoder(pl.LightningModule):
    def __init__(self, num_genes, PARAMETERS):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(PARAMETERS['latent_dimension'], 128),  # Decoder layers
            nn.BatchNorm1d(128),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(256, num_genes),    # Output layer
            nn.Sigmoid()                   # Use Sigmoid for output
        )

    def forward(self, x):
        return self.decoder(x)

# Define the Autoencoder model
class scRNASeqAE(pl.LightningModule):
    def __init__(self, num_genes, PARAMETERS):
        super(scRNASeqAE, self).__init__()
        self.PARAMETERS = PARAMETERS
        self.encoder = scRNASeqEncoder(num_genes, PARAMETERS)
        self.decoder = scRNASeqDecoder(num_genes, PARAMETERS)
        self.loss_function = nn.MSELoss()
        
        
    def forward(self, x):
        latent = self.encoder(x)        # Encoding
        return self.decoder(latent)     # Decoding

    def encode(self, x):
        return self.encoder(x)
    
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon_x = self.forward(x)
        loss = nn.MSELoss()(recon_x, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recon_x = self.forward(x)
        loss = nn.MSELoss()(recon_x, x)
        self.log('validation_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters())
    