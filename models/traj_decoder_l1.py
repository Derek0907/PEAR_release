import torch
import torch.nn as nn
import torch.nn.functional as F
from .CVAE.traj_vae import TrajVAE
from .fusion.traj_condition_fusion import Traj_Hmap_Fusion_small

class Traj_decoder_l1(nn.Module):
    """
    Trajectory decoder L1 stage.
    This module predicts the initial trajectory in a two-stage manner using two CVAE blocks.
    """
    def __init__(self):
        super(Traj_decoder_l1, self).__init__()

        # Two CVAE blocks for sequential prediction
        self.vae_1 = TrajVAE(in_dim = 32 * 32, hidden_dim= 512, latent_dim= 64, condition_dim = 196)
        self.vae_2 = TrajVAE(in_dim = 32 * 32, hidden_dim= 512, latent_dim= 64, condition_dim = 196)
        
        # Heatmap and feature fusion modules for each CVAE stage
        self.traj_fusion_1 = Traj_Hmap_Fusion_small()
        self.traj_fusion_2 = Traj_Hmap_Fusion_small()

        for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        self.apply(_init_vit_weights)
        
    def normalize_hmap(self, hmap):
        """
        Normalize a heatmap to [0, 1] range.
        """
        max_val = torch.max(hmap)
        min_val = torch.min(hmap)
        hmap = (hmap - min_val) / (max_val - min_val)
        return hmap
        
    def forward(self, gt, hmap_init, traj_feature, pred_hotspots):
        """
        Forward pass for training.
        Args:
            gt: Ground truth trajectory [B, 3, 32, 32]
            hmap_init: Initial heatmap input [B, 32, 32]
            traj_feature: Feature condition [B, 768]
            pred_hotspots: Predicted hotspot map [B, 224, 224]
        Returns:
            pred_traj: Predicted trajectory [B, 3, 32, 32]
            recon_loss, KLD: Total reconstruction loss and KL divergence
        """
        # Downsample predicted hotspots to match CVAE resolution
        pred_hotspots= F.interpolate(pred_hotspots.unsqueeze(1), size=(32, 32), mode='bilinear', align_corners=False).squeeze(1) 
        bs = gt.shape[0]
        
        # Stage 1: Predict first-step trajectory from initial heatmap and global feature
        traj_condition_1 = self.traj_fusion_1(hmap_init, traj_feature)
        traj_condition_1 = traj_condition_1.reshape(bs, -1) 
        gt_1 = gt[:, 0, :, :].reshape(bs, -1)
        pred_t1, recon_1, KLD_1 = self.vae_1(gt_1, traj_condition_1)
        pred_t1 = pred_t1.reshape(bs, 32, 32)
        
        # Stage 2: Predict second-step trajectory conditioned on predicted t1 and predicted hotspots
        # The fused condition vector encodes global trajectory semantics and hotspot cues
        t2_condition = pred_t1 + pred_hotspots 
        t2_condition = self.normalize_hmap(t2_condition) 
        traj_condition_2 = self.traj_fusion_2(pred_t1, traj_feature)
        traj_condition_2 = traj_condition_2.reshape(bs, -1) 
        gt_2 = gt[:, 1, :, :].reshape(bs, -1)
        pred_t2, recon_2, KLD_2 = self.vae_2(gt_2, traj_condition_2)
        pred_t2 = pred_t2.reshape(bs, 32, 32)
        
        # Concatenate predictions for loss and output
        pred_hotspots = pred_hotspots.unsqueeze(1)
        pred_t1 = pred_t1.unsqueeze(1)
        pred_t2 = pred_t2.unsqueeze(1)
     
        pred_traj = torch.cat([pred_t1, pred_t2, pred_hotspots], dim=1) #[B, 3, 32, 32]
        recon_loss = recon_1 + recon_2 
        KLD = KLD_1 + KLD_2 
        return pred_traj, recon_loss, KLD
    

    def inference(self, hmap_init, traj_feature, pred_hotspots):
        """
        Inference mode for L1 trajectory decoder (no GT used).
        """
        pred_hotspots= F.interpolate(pred_hotspots.unsqueeze(1), size=(32, 32), mode='bilinear', align_corners=False).squeeze(1) 
        bs = hmap_init.shape[0]
        
        # Stage 1
        traj_condition_1 = self.traj_fusion_1(hmap_init, traj_feature)
        traj_condition_1 = traj_condition_1.reshape(bs, -1) 
        pred_t1 = self.vae_1.inference(traj_condition_1)
        pred_t1 = pred_t1.reshape(bs, 32, 32)
        
        # Stage 2
        t2_condition = pred_t1 + pred_hotspots 
        t2_condition = self.normalize_hmap(t2_condition) 
        traj_condition_2 = self.traj_fusion_2(t2_condition, traj_feature)
        traj_condition_2 = traj_condition_2.reshape(bs, -1)
        pred_t2 = self.vae_2.inference(traj_condition_2)    
        pred_t2 = pred_t2.reshape(bs, 32, 32)
        
        # Concatenate
        pred_t1 = pred_t1.unsqueeze(1)
        pred_t2 = pred_t2.unsqueeze(1)
        pred_hotspots = pred_hotspots.unsqueeze(1)
        
        pred_traj = torch.cat([pred_t1, pred_t2, pred_hotspots], dim=1) #[B, 3, 32, 32]
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
