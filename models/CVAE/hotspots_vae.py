import torch
import torch.nn as nn

class VAE(nn.Module):   
    """
    Conditional Variational Autoencoder (CVAE).
    Accepts condition vector c and input x, learns latent z ~ q(z|x,c),
    reconstructs x using p(x|z,c).
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
        recon_loss = torch.sum((recon_x - x) ** 2, dim=1)
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
        return recon_loss, KLD

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def inference(self, z, c=None):
        if self.conditional and c is not None:
            z = torch.cat((z, c), dim=-1)
        recon_x = self.dec_MLP(z)
        return recon_x
    
    
class HotspotsCVAE(nn.Module):
   
    """
    CVAE wrapper tailored for 224x224 heatmaps used in interaction hotspots.
    Internally uses the general VAE defined above.
    """
    def __init__(self, in_dim, hidden_dim, latent_dim, condition_dim, coord_dim=None,
                 condition_contact=False, z_scale=2.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_contact = condition_contact
        self.z_scale = z_scale
        
        self.cvae = VAE(in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                        conditional=True, condition_dim=condition_dim)
    
    
    def forward(self, target_hand, context, contact_point=None, return_pred=True):
        """
        Training mode: Learn to reconstruct hotspot map from (z, context).

        Args:
            target_hand (Tensor): Ground truth hotspot heatmap, [B, 224*224]
            context (Tensor): Conditioning vector [B, C]

        Returns:
            pred_hand (Tensor): Reconstructed heatmap [B, 224, 224]
            recon_loss, KLD: Standard VAE losses
        """
        batch_size = context.shape[0]
        condition_context = context
        if not return_pred:
            recon_loss, KLD = self.cvae(target_hand, c=condition_context)
        else:
            pred_hand, recon_loss, KLD = self.cvae(target_hand, c=condition_context, return_pred=return_pred)
        pred_hand = pred_hand.reshape(batch_size, 224, 224)
        if not return_pred:
            return recon_loss, KLD
        else:
            return pred_hand, recon_loss, KLD

    def inference(self, context, contact_point=None):
        """
        Inference mode: Sample z from standard normal, generate hotspot map.

        Args:
            context (Tensor): Conditioning vector [B, C]

        Returns:
            recon_x (Tensor): Sampled heatmap [B, 224, 224]
        """
        condition_context = context
        z = self.z_scale * torch.randn([context.shape[0], self.latent_dim], device=context.device)
        recon_x = self.cvae.inference(z, c=condition_context)
        recon_x = recon_x.reshape(context.shape[0], 224, 224)

        return recon_x
    