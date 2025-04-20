# Ripped off wholesale from https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple

import json

from utils import batch_data

Tensor = TypeVar('torch.tensor')

# %% Prototype

class VanillaVAE(nn.Module):


    def __init__(self,
                input_dim: int,
                model_dim: int,
                hidden_dims: List = None,
                learning_rate=1e-3,
                weight_decay=1e-5,
                batch_size=1000,
                kld_weight = .001,
        ) -> None:
        argset = {k:v for k,v in locals().items() if k not in {'self','__class__'}}
        # argset.pop('self')
        # argset.pop('__class__') # why is this even here?

        super(VanillaVAE, self).__init__()
        
        self.argset = {k:v for k,v in argset.items()}

        self.model_dim = model_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.kld_weight = kld_weight

        self.data_loss_history = []
        self.validation_loss_history = []

        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64, 32]

        modules = [nn.Linear(input_dim,hidden_dims[0]),]

        # Build Encoder
        for h_dim_1,h_dim_2 in zip(hidden_dims[:-1],hidden_dims[1:]):
            modules.append(
                nn.Sequential(
                    nn.Linear(h_dim_1,h_dim_2),
                    nn.LeakyReLU())
            )

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], model_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], model_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(model_dim, hidden_dims[-1])

        hidden_dims.reverse()

        for h_dim_1,h_dim_2 in zip(hidden_dims[:-1],hidden_dims[1:]):
            modules.append(
                nn.Sequential(
                    nn.Linear(h_dim_1,h_dim_2),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(h_dim_2,input_dim)
                            )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder (batch_size x input_dim)
        :return: (Tensor) List of latent codes (means [batch_size x model_dim], log-variances [batch_size x model_dim])
        """
        result = self.encoder(input)
        # result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def kld_loss(means,log_vars):
        return torch.mean(-0.5 * torch.sum(1 + log_vars - (means ** 2) - log_vars.exp(), dim = 1), dim = 0)

    def loss_function(self,
                      recons,
                      input,
                      mu,
                      log_var,
                      kld_weight=0.00025 
        ) -> dict:

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.model_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    # Interface shim
    def embeddings(self,data):
        [means,log_var] = self.encode(torch.Tensor(data))
        return {"means":means,"log_var":log_var}
    
    @classmethod
    def load(cls,path,device=torch.device('cpu')):
        with open(path + ".params",mode='r') as f:
            params = json.load(f)
            new_model = cls(**params)
            new_model.to(device)
            new_model.load_state_dict(torch.load(path,map_location=new_model.device()))
            return new_model
 
    def save(self,path):
        with open(path+".params",mode='w') as f:
            f.write(json.dumps(self.argset))
        torch.save(self.state_dict(),path)
    
    def device(self):
        return next(self.parameters()).device

    def train(
            self,
            training_set,
            validation_set=None,
            num_iter=100000,
            logging_frequency = 1000,
            device=torch.device('cuda:0'),
        ):


        optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate,weight_decay=self.weight_decay)     

        for i in range(num_iter):

            optimizer.zero_grad()
            
            batch = batch_data(training_set,batch_size=self.batch_size,device=device)
            results = self.forward(batch)
            loss_results = self.loss_function(*results,kld_weight=self.kld_weight)
            loss = loss_results['loss']

            if validation_set is not None:
                validation_batch = batch_data(validation_set,batch_size=self.batch_size,device=device)
                validation_res = self.forward(validation_batch)
                validation_loss_res = self.loss_function(*validation_res,kld_weight=self.kld_weight)
                validation_loss = validation_loss_res['loss']
            
            loss.backward()
            optimizer.step()

            if i%logging_frequency == 0:
            #     # Naming is a little weird here but I don't want to jump to vsc to update it right now, data_loss_hist = training loss hist
                training_report = f"Step:{i},RL:{loss_results['Reconstruction_Loss']},KLD:{loss_results['KLD']}"
                self.data_loss_history.append(training_report)
                print(training_report)
                if validation_set is not None:
                    validation_report = f"Step:{i},VRL:{validation_loss_res['Reconstruction_Loss']}"
                    self.validation_loss_history.append(validation_report)
                    print(validation_report)



    def set_extra_state(self,state):
        if state is None:
            state = {}
        if 'data_loss_history' not in state:
            state['data_loss_history'] = []
        if 'validation_loss_history' not in state:
            state['validation_loss_history'] = []
        if 'argset' not in state:
            state['argset'] = ""

        self.data_loss_history = state['data_loss_history']
        self.validation_loss_history = state['validation_loss_history']
        self.argset = state['argset']

    def get_extra_state(self):
        try:
            extra_state = {
                'data_loss_history' : self.data_loss_history,
                'validation_loss_history' : self.validation_loss_history,
                'argset' : self.argset,
            }
        except Exception as e:
            print(f"History reconstruction exception {e}")
            extra_state = {
                'data_loss_history' : [],
                'validation_loss_history' : [],
                'argset' : "",
            }            
        return extra_state

    # def solve_embedding_prior(
    #         self,
    #         training_set,
    #         means,
    #         log_var,
    #         num_iter=100000,
    #         learning_rate=1e-3,
    #         weight_decay=1e-5,
    #         batch_size=1000,
    #         logging_frequency = 1000,
    #         entropy_weight = 1e-5,
    #         device=torch.device('cuda:0'),
    #         kld_weight = .001
    #     ):


    #     optimizer = torch.optim.Adam(self.parameters(),lr=learning_rate,weight_decay=weight_decay)     

    #     for i in range(num_iter):

    #         optimizer.zero_grad()
            
    #         batch_indices = make_batch_indices(training_set,batch_size=batch_size)
    #         data_batch = batch_data(training_set,batch_indices=batch_indices,device=device)
    #         means_batch = batch_data(means,batch_indices=batch_indices,device=device)
    #         log_var_batch = batch_data(log_var,batch_indices=batch_indices,device=device)

    #         r_data,_,r_means,r_log_var = self.forward(data_batch)
    #         loss_results = self.loss_function(r_data,data_batch,r_means,r_log_var,kld_weight=kld_weight)

    #         prior_means_loss = F.mse_loss(means_batch,r_means)
    #         prior_log_var_loss = F.mse_loss(log_var_batch,r_log_var)

    #         loss = loss_results['loss'] + prior_means_loss + prior_log_var_loss

    #         loss.backward()
    #         optimizer.step()

    #         if i%logging_frequency == 0:
    #             print(f"Step:{i},RL:{loss_results['Reconstruction_Loss']},KLD:{loss_results['KLD']}")


# %%

