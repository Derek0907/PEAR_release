import torch.nn as nn
from .CVAE.contact_vae import Contact_VAE

class Area_decoder(nn.Module):
    """
    Decoder for contact area prediction based on a conditional VAE.

    Takes the feature vector (e.g., 768) as condition input and 
    reconstructs a 778-dimensional contact area prediction.
    """
    def __init__(self, cfg, input_dim = 778, out_dim=778):
        super(Area_decoder, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        black_points_path = cfg.BLIP.BLACKLIST_PATH
        self.contact_vae = Contact_VAE(in_dim = 778, hidden_dim = 512, latent_dim = 32, condition_dim = 768, condition_contact = True, black_points_path=black_points_path)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def binalize_area(self, area):
        """
        Threshold contact area output into binary labels: values > 0.5 set to 1, else 0.
        """
        area = (area > 0.5).float()
        return area


    def forward(self, gt, condition):
        """
        Forward pass during training.

        Args:
            gt (Tensor): Ground truth contact area, shape [B, 778]
            condition (Tensor): Contextual feature, shape [B, 768]
        
        Returns:
            pred_area: Reconstructed area prediction
            recon_loss: BCE reconstruction loss
            KLD: KL divergence loss
        """
        pred_area, recon_loss, KLD = self.contact_vae(gt, condition) 
      

        return pred_area, recon_loss, KLD   

    def inference(self, condition):
        """
        Inference mode: sample from latent space to generate contact map.
        """
        pred_area = self.contact_vae.inference(condition) 
        pred_area = self.binalize_area(pred_area)
        return pred_area
        

    






