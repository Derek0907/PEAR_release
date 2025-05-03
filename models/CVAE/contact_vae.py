import torch
import torch.nn as nn

class VAE(nn.Module):
    """
    Conditional VAE module for predicting hand contact area.

    Supports BCE loss, Sigmoid activation, and latent sampling.
    """
    def __init__(self, in_dim, hidden_dim, latent_dim, conditional=True, condition_dim=None, black_points_path=None):

        super().__init__()

        self.latent_dim = latent_dim
        self.conditional = conditional
        self.black_points_path = black_points_path
        
        if self.conditional and condition_dim is not None:
            input_dim = in_dim + condition_dim
            dec_dim = latent_dim + condition_dim 
        else:
            input_dim = in_dim
            dec_dim = latent_dim
            
        self.enc_MLP = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU()
        )
        self.linear_means = nn.Linear(hidden_dim, latent_dim)
        self.linear_log_var = nn.Linear(hidden_dim, latent_dim)
        self.dec_MLP = nn.Sequential(
            nn.Linear(dec_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, in_dim))
        self.bceloss = nn.BCELoss(reduction='sum')
        self.sigmoid = nn.Sigmoid()

    def binalize_area(self, area):
        """
        Convert area values to binary using a 0.5 threshold.
        """
        area = (area > 0.5).float()
        return area
    
    def forward(self, x, c=None, return_pred=False):
        """
        Forward pass for training.

        Args:
            x: GT contact map [B, 778]
            c: condition vector [B, 768]
        
        Returns:
            pred or loss depending on return_pred
        """
        if self.conditional and c is not None:
            inp = torch.cat((x, c), dim=-1)
        else:
            inp = x
                
        # Forward through encoder
        h = self.enc_MLP(inp)
        mean = self.linear_means(h)
        log_var = self.linear_log_var(h)
        
        # Latent sampling
        z = self.reparameterize(mean, log_var)
        
        # Decode from z + condition
        if self.conditional and c is not None:
            z = torch.cat((z, c), dim=-1)
            
        recon_x = self.dec_MLP(z)
        recon_x = self.sigmoid(recon_x)
        recon_x = recon_x.float()
        x = x.float()
        
        recon_loss, KLD = self.loss_fn(recon_x, x, mean, log_var)
        if not return_pred:
            return recon_loss, KLD
        else:
            return recon_x, recon_loss, KLD

    def loss_fn(self, recon_x, x, mean, log_var):
        """
        Compute BCE + KL Divergence loss.
        """
        bs = x.shape[0]
        loss_list = []
        for i in range(bs):
            b_loss = self.bceloss(recon_x[i], x[i])
            loss_list.append(b_loss)
        recon_loss = torch.stack(loss_list)
        recon_loss = recon_loss.float()
        KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
        return recon_loss, KLD

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def inference(self, z, c=None):
        """
        Sampling-based inference: randomly sample z ~ N(0,1), decode with condition.
        """
        list = []
        with open(self.black_points_path, 'r') as f:
            for line in f:
                list.append(int(line.strip())) # When reasoning, set the valid contact area on the back of the hand to invalid
        if self.conditional and c is not None:
            z = torch.cat((z, c), dim=-1)
        recon_x = self.dec_MLP(z)
        recon_x = self.sigmoid(recon_x)
        bs = recon_x.shape[0]
        for i in range(bs):
            for j in range(778):
                if j in list:
                    recon_x[i,j] = 0    
        return recon_x
    
class Contact_VAE(nn.Module):
    """
    Wrapper around VAE for contact area prediction. Includes interface for training and inference.
    """
    def __init__(self, in_dim, hidden_dim, latent_dim, condition_dim, coord_dim=None,
                 condition_contact=False, z_scale=2.0, black_points_path = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_contact = condition_contact
        self.z_scale = z_scale
        self.cvae = VAE(in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                        conditional=True, condition_dim=condition_dim, black_points_path=black_points_path)

    def forward(self,  target_hand, context, contact_point=None, return_pred=True):
        """
        Forward pass of the CVAE for contact area.

        Args:
            target_hand: Ground truth contact [B, 778]
            context: Condition [B, 768]
        
        Returns:
            Either loss or prediction depending on return_pred
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
        Inference mode: sample latent z and decode with condition.
        """
        condition_context = context
        z = self.z_scale * torch.randn([context.shape[0], self.latent_dim], device=context.device)
        recon_x = self.cvae.inference(z, c=condition_context)
        return recon_x
