import torch
import torch.nn as nn
import torch.nn.functional as F
from .CVAE.traj_vae import TrajVAE
from .fusion.traj_condition_fusion import Traj_Hmap_Fusion_small

class Traj_decoder_l2(nn.Module):
    """
    Trajectory decoder L2 stage.
    This module predicts manipulation trajectories using three chained CVAEs.
    """
    def __init__(self):
        super(Traj_decoder_l2, self).__init__()

        self.vae_1 = TrajVAE(in_dim = 32 * 32, hidden_dim= 512, latent_dim= 64, condition_dim = 196)
        self.vae_2 = TrajVAE(in_dim = 32 * 32, hidden_dim= 512, latent_dim= 64, condition_dim = 196)
        self.vae_3 = TrajVAE(in_dim = 32 * 32, hidden_dim= 512, latent_dim= 64, condition_dim = 196)
        
        self.traj_fusion_1 = Traj_Hmap_Fusion_small()
        self.traj_fusion_2 = Traj_Hmap_Fusion_small()
        self.traj_fusion_3 = Traj_Hmap_Fusion_small()

        for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        self.apply(_init_vit_weights)
        
    def forward(self, gt, traj_feature, pred_hotspots):
        """
        Training-time forward pass for multi-step manipulation trajectory generation.
        
        Args:
            gt (Tensor): Ground truth 3-step trajectory heatmaps [B, 3, 32, 32]
            traj_feature (Tensor): Semantic feature vector [B, 768]
            pred_hotspots (Tensor): Predicted hotspot heatmap [B, 224, 224]

        Returns:
            pred_traj (Tensor): Predicted trajectory sequence [B, 3, 32, 32]
            recon_loss (Tensor): Total reconstruction loss across steps
            KLD (Tensor): Total KL-divergence loss across steps
        """
        
        # Preprocess predicted hotspots to match CVAE resolution
        pred_hotspots= F.interpolate(pred_hotspots.unsqueeze(1), size=(32, 32), mode='bilinear', align_corners=False).squeeze(1) #[B, 32, 32]
        bs = gt.shape[0]
        
        # ----- Step 1 -----
        traj_condition_1 = self.traj_fusion_1(pred_hotspots, traj_feature)
        traj_condition_1 = traj_condition_1.reshape(bs, -1) 
        gt_1 = gt[:, 0, :, :].reshape(bs, -1)
        pred_t1, recon_1, KLD_1 = self.vae_1(gt_1, traj_condition_1)
        pred_t1 = pred_t1.reshape(bs, 32, 32)
        
        # ----- Step 2 -----
        traj_condition_2 = self.traj_fusion_2(pred_t1, traj_feature)
        traj_condition_2 = traj_condition_2.reshape(bs, -1) 
        gt_2 = gt[:, 1, :, :].reshape(bs, -1)
        pred_t2, recon_2, KLD_2 = self.vae_2(gt_2, traj_condition_2)
        pred_t2 = pred_t2.reshape(bs, 32, 32)
        
        # ----- Step 3 -----
        traj_condition_3 = self.traj_fusion_3(pred_t2, traj_feature)
        traj_condition_3 = traj_condition_3.reshape(bs, -1) 
        gt_3 = gt[:, 2, :, :].reshape(bs, -1)
        pred_t3, recon_3, KLD_3 = self.vae_3(gt_3, traj_condition_3)
        pred_t3 = pred_t3.reshape(bs, 32, 32)
        
        # Stack and compute losses
        pred_t1 = pred_t1.unsqueeze(1)
        pred_t2 = pred_t2.unsqueeze(1)
        pred_t3 = pred_t3.unsqueeze(1)
        pred_traj = torch.cat([pred_t1, pred_t2, pred_t3], dim=1)
        recon_loss = recon_1 + recon_2 + recon_3
        KLD = KLD_1 + KLD_2 + KLD_3
        return pred_traj, recon_loss, KLD
    

    def inference(self, traj_feature, pred_hotspots):
        """
        Inference-time trajectory generation (3-step autoregressive decoding).

        Args:
            traj_feature (Tensor): Semantic feature vector [B, 768]
            pred_hotspots (Tensor): Predicted interaction heatmap [B, 224, 224]

        Returns:
            pred_traj (Tensor): Predicted 3-stage trajectory [B, 3, 32, 32]
        """
        pred_hotspots= F.interpolate(pred_hotspots.unsqueeze(1), size=(32,32), mode='bilinear', align_corners=False).squeeze(1) #[B, 32, 32]
        bs = traj_feature.shape[0]
        
        # ----- Step 1 -----
        traj_condition_1 = self.traj_fusion_1(pred_hotspots, traj_feature)
        traj_condition_1 = traj_condition_1.reshape(bs, -1) 
        pred_t1 = self.vae_1.inference(traj_condition_1) 
        pred_t1 = pred_t1.reshape(bs, 32,32)
       
        # ----- Step 2 -----
        traj_condition_2 = self.traj_fusion_2(pred_t1, traj_feature)
        traj_condition_2 = traj_condition_2.reshape(bs, -1) 
        pred_t2 = self.vae_2.inference(traj_condition_2)    
        pred_t2 = pred_t2.reshape(bs, 32,32)
       
        # ----- Step 3 -----
        traj_condition_3 = self.traj_fusion_3(pred_t2, traj_feature)
        traj_condition_3 = traj_condition_3.reshape(bs, -1) 
        pred_t3 = self.vae_3.inference(traj_condition_3)
        pred_t3 = pred_t3.reshape(bs, 32,32)

        # Concatenate
        pred_t1 = pred_t1.unsqueeze(1)
        pred_t2 = pred_t2.unsqueeze(1)
        pred_t3 = pred_t3.unsqueeze(1)
        pred_traj = torch.cat([pred_t1, pred_t2, pred_t3], dim=1) #[B, 3, 32, 32]
        return pred_traj

def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)  
