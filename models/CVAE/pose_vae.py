import torch
import torch.nn as nn
from ..MANO.utils.geometry import rot6d_to_rotmat
import numpy as np
class VAE(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) that reconstructs hand poses 
    with optional conditioning on context vectors.
    """
    
    def __init__(self, in_dim, hidden_dim, latent_dim, conditional=True, condition_dim=None, pose_mean_path=None):
        """
        Initializes the VAE module.

        Args:
            in_dim (int): Input dimensionality (e.g., 154 for pose).
            hidden_dim (int): Hidden layer size.
            latent_dim (int): Latent vector size.
            conditional (bool): Whether to use conditional VAE.
            condition_dim (int): Dimensionality of the condition vector.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.conditional = conditional
        self.mean_path = pose_mean_path
        
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
        
        self.sigmoid = nn.Sigmoid()
        mean_params = np.load(self.mean_path)
        init_hand_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        self.register_buffer('init_hand_pose', init_hand_pose)
        self.register_buffer('init_betas', init_betas)
        
        

    def forward(self, x, c=None, return_pred=False):
        """
        Full forward pass including encoding, sampling, decoding, and loss computation.

        Args:
            x (Tensor): Input tensor [B, in_dim].
            c (Tensor): Optional condition tensor [B, condition_dim].
            return_pred (bool): Whether to return the reconstructed pose.

        Returns:
            recon_x or loss terms depending on return_pred.
        """
        bs = x.shape[0]
        init_hand_pose = self.init_hand_pose.expand(bs, -1)
        init_betas = self.init_betas.expand(bs, -1)
        
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
        # Restore full MANO representation
        re_pose = recon_x[:, :96] + init_hand_pose
        re_beta = recon_x[:, 96:106] + init_betas
        pred_hand_pose = rot6d_to_rotmat(re_pose).view(bs, 144)
        pred_betas = re_beta.view(bs, 10)
        recon_x = torch.cat((pred_hand_pose, pred_betas), dim=-1)
        
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
        """
        Inference mode: samples from latent prior and decodes to pose.

        Args:
            z (Tensor): Latent vector [B, latent_dim]
            c (Tensor): Optional condition vector

        Returns:
            Tensor: Reconstructed pose vector [B, out_dim]
        """
        if self.conditional and c is not None:
            z = torch.cat((z, c), dim=-1)
            
        bs = z.shape[0]
        init_hand_pose = self.init_hand_pose.expand(bs, -1)
        init_betas = self.init_betas.expand(bs, -1)  
          
        recon_x = self.dec_MLP(z)
        re_pose = recon_x[:, : 96] + init_hand_pose
        re_beta = recon_x[:, 96:106] + init_betas
        pred_hand_pose = rot6d_to_rotmat(re_pose).view(bs, 144)
        pred_betas = re_beta.view(bs, 10)
        recon_x = torch.cat((pred_hand_pose, pred_betas), dim=-1)
        return recon_x
    
class PoseCVAE(nn.Module):
    """
    Wrapper class for pose CVAE that abstracts training and inference logic.
    """
    
    def __init__(self, in_dim, hidden_dim, latent_dim, condition_dim, coord_dim=None,
                 condition_contact=False, z_scale=2.0, pose_mean_path=None):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_contact = condition_contact
        self.z_scale = z_scale
        self.pose_mean_path = pose_mean_path
        self.cvae = VAE(in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                        conditional=True, condition_dim=condition_dim, pose_mean_path=pose_mean_path)

    def forward(self, target_hand, context, contact_point=None, return_pred=True):
        """
        Forward pass for CVAE training.

        Args:
            target_hand (Tensor): Ground-truth hand pose vector.
            context (Tensor): Contextual condition vector.
            return_pred (bool): Whether to return prediction or just loss.

        Returns:
            Tuple or loss values.
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
        Inference mode: samples latent z and generates pose prediction.

        Args:
            context (Tensor): Contextual feature.

        Returns:
            Tensor: Predicted hand pose.
        """
        condition_context = context
        z = self.z_scale * torch.randn([context.shape[0], self.latent_dim], device=context.device)
        recon_x = self.cvae.inference(z, c=condition_context)
        return recon_x
    