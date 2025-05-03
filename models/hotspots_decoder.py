import torch.nn as nn
from .CVAE.hotspots_vae import HotspotsCVAE

class Hotspots_decoder(nn.Module):
    """
    Wrapper around HotspotsCVAE to generate and reconstruct hand-object interaction hotspots.
    """

    def __init__(self, input_dim, condition_dim):
        super(Hotspots_decoder, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim

        # Instantiate CVAE with latent conditioning
        self.cvae = HotspotsCVAE(in_dim=input_dim, hidden_dim=768, latent_dim=180, condition_dim=condition_dim)
        self.relu = nn.ReLU()
        for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        # nn.init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, gt, condition):
        """
        Forward method used during training.

        Args:
            gt (Tensor): Ground truth heatmap, shape [B, 224, 224]
            condition (Tensor): Conditioning feature, shape [B, 224, 224]

        Returns:
            pred_hotspots: CVAE reconstruction from latent z
            recon_loss: MSE reconstruction loss
            KLD: KL-divergence loss
        """
        bs = gt.shape[0]
        condition = condition.reshape(bs, -1)
        gt = gt.reshape(bs, -1)
        pred_hotspots, recon_loss, KLD = self.cvae(gt, condition)
        return pred_hotspots, recon_loss, KLD

    
    def inference(self, condition):
        """
        Inference mode for CVAE - generate from sampled z and condition.

        Args:
            condition (Tensor): Conditioning feature, shape [B, 224, 224]

        Returns:
            pred_hotspots (Tensor): Generated heatmap, shape [B, 224, 224]
        """
        bs = condition.shape[0]
        condition = condition.reshape(bs, -1)
        pred_hotspots = self.cvae.inference(condition)
        return pred_hotspots
