import torch
import torch.nn as nn

class VAE(nn.Module):
    """
    Basic Variational Autoencoder module that supports conditional input.
    Used for trajectory generation via latent sampling.
    """
    def __init__(self, in_dim, hidden_dim, latent_dim, conditional=True, condition_dim=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.conditional = conditional
        if self.conditional and condition_dim is not None:
            input_dim = in_dim + condition_dim
            dec_dim = latent_dim + condition_dim
         
        else:
            input_dim = in_dim
            dec_dim = latent_dim
        self.enc_MLP = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU())
        self.linear_means = nn.Linear(hidden_dim, latent_dim)
        self.linear_log_var = nn.Linear(hidden_dim, latent_dim)
        self.dec_MLP = nn.Sequential(
            nn.Linear(dec_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, in_dim))
        
    def forward(self, x, c=None, return_pred=False):
        """
        Forward pass during training.
        Args:
            x: input vector [B, in_dim]
            c: conditional context [B, condition_dim]
            return_pred: if True, returns prediction alongside loss
        Returns:
            Either (recon_loss, KLD) or (recon_x, recon_loss, KLD)
        """
        bs = x.shape[0]
        if self.conditional and c is not None:
            inp = torch.cat((x, c), dim=-1)
        else:
            inp = x
            
        h = self.enc_MLP(inp)
        mean = self.linear_means(h)
        log_var = self.linear_log_var(h)
        z = self.reparameterize(mean, log_var)
        
        
        if self.conditional and c is not None:
            z = torch.cat((z, c), dim=-1)
        recon_x = self.dec_MLP(z) 
        
        recon_loss, KLD = self.loss_fn(recon_x, x, mean, log_var)
        
        if not return_pred:
            return recon_loss, KLD
        else:
            return recon_x, recon_loss, KLD
        
    def loss_fn(self, recon_x, x, mean, log_var):
        """
        Compute reconstruction loss (MSE) and KL divergence
        """
        recon_loss = torch.sum((recon_x - x) ** 2, dim=1)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
        return recon_loss, KLD

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, sigma^2)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def inference(self, z, c=None):
        """
        Inference mode: generate data from sampled z and optional condition
        """
        if self.conditional and c is not None:
            z = torch.cat((z, c), dim=-1)
        recon_x = self.dec_MLP(z)
        return recon_x


class TrajVAE(nn.Module):
    """
    Wrapper module for trajectory prediction using conditional VAE.
    """
    def __init__(self, in_dim, hidden_dim, latent_dim, condition_dim, coord_dim=None,
                 condition_contact=False, z_scale=2.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_contact = condition_contact
        self.z_scale = z_scale
        
        # Internal CVAE module for trajectory modeling
        self.cvae = VAE(in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                        conditional=True, condition_dim=condition_dim)
        

    def forward(self, target_hand, context, contact_point=None, return_pred=True):
        """
        Training-time forward pass.
        Args:
            target_hand: target trajectory (flattened heatmap) [B, in_dim]
            context: condition feature vector [B, condition_dim]
            contact_point: (unused) reserved for future use
            return_pred: whether to return generated output
        Returns:
            Depending on return_pred, either (recon_loss, KLD) or (pred, recon_loss, KLD)
        """
        condition_context = context
        if not return_pred:
            recon_loss, KLD = self.cvae(target_hand, c=condition_context)
        else:
            pred_hand, recon_loss, KLD = self.cvae(target_hand, c=condition_context, return_pred=return_pred)
        if not return_pred:
            return recon_loss, KLD
        else:
            return pred_hand, recon_loss, KLD

    def inference(self, context, contact_point=None):
        """
        Inference-time sampling from prior and decoding
        Args:
            context: condition feature vector [B, condition_dim]
        Returns:
            Reconstructed trajectory from latent sample
        """
        condition_context = context
        z = self.z_scale * torch.randn([context.shape[0], self.latent_dim], device=context.device)
        recon_x = self.cvae.inference(z, c=condition_context)
        return recon_x
    
