import torch.nn as nn
import torch
from .MANO.mano_wrapper import MANO
from yacs.config import CfgNode
from .MANO.utils import SkeletonRenderer, MeshRenderer
from .CVAE.pose_vae import PoseCVAE

class Pose_decoder(nn.Module):
    """
    A pose decoder module that uses a conditional variational autoencoder (CVAE) 
    to reconstruct hand pose parameters and convert them to 3D joint coordinates using MANO.
    """
    
    def __init__(self, input_dim, cfg: CfgNode, init_renderer: bool = True, out_dim=154, depth=2):
        """
        Initializes the PoseDecoder module.

        Args:
            input_dim (int): Dimension of the pose vector input.
            cfg (CfgNode): Configuration object for MANO and rendering.
            init_renderer (bool): Whether to initialize mesh renderers for visualization.
            out_dim (int): Output dimension of pose parameters.
            depth (int): Depth setting, not actively used here.
        """
        super(Pose_decoder, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.depth = depth
        self.pose_vae = PoseCVAE(in_dim = 154, hidden_dim = 256, latent_dim =32, condition_dim = 768, condition_contact=True, pose_mean_path=cfg.BLIP.POSE_MEAN_PATH)
        self.cfg = cfg
        self.init_renderer = init_renderer
        self.linear = nn.Linear(768, 1024)
        self.relu = nn.ReLU()
        
        mano_cfg = {k.lower(): v for k,v in dict(cfg.MANO).items()}
        self.mano = MANO(**mano_cfg)
        self.register_buffer('initialized', torch.tensor(False))

        if init_renderer:
            self.renderer = SkeletonRenderer(self.cfg)
            self.mesh_renderer = MeshRenderer(self.cfg, faces=self.mano.faces)
        else:
            self.renderer = None
            self.mesh_renderer = None

        # Kaiming initialization for Conv2D/BatchNorm layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
      
    
    def transfer(self, x):
        """
        Converts the CVAE output vector into MANO input and computes 3D joints/mesh.

        Args:
            x (Tensor): Tensor of shape (B, 154) representing MANO parameters.

        Returns:
            dict: Contains 'pred_mano_params', 'pred_keypoints_3d', and 'pred_vertices'.
        """
        B = x.shape[0]
        x = x.reshape(B, 1, self.out_dim)
        glo= x[:, :,:9].reshape(B, -1, 3, 3)
        hand = x[:, :,9:144].reshape(B ,-1, 3, 3)
        betas= x[:, :,144:154].reshape(B, 10)

        pred_mano_params = {'global_orient': glo, 'hand_pose': hand, 'betas': betas}
        mano = self.mano(**{k: v.float() for k,v in pred_mano_params.items()}, pose2rot=False)
        output = {}

        output['pred_mano_params'] = {k: v.clone() for k,v in pred_mano_params.items()}
        keypoints_3d = mano.joints
        vertices = mano.vertices
        output['pred_keypoints_3d'] = keypoints_3d
        output['pred_vertices'] = vertices

        return output


    def forward(self, gt_pose, condition):
        """
        Forward pass during training: generates pose prediction and reconstruction loss.

        Args:
            gt_pose (Tensor): Ground-truth pose parameters [B, 154].
            condition (Tensor): Context feature [B, 768].

        Returns:
            Tuple: pose dict, reconstruction loss, KL divergence, and raw pose tensor.
        """
        bs = condition.shape[0]
        condition = condition.reshape(bs, -1)
        pred_pose, recon_loss, KLD = self.pose_vae(gt_pose, condition)  # VAE forward
        pred_pose = pred_pose.reshape(-1, 154)
        
        # Convert predicted pose into MANO parameters and generate 3D joints & vertices
        trans_pose = self.transfer(pred_pose)
        return trans_pose, recon_loss, KLD, pred_pose

    def inference(self, condition):
        """
        Inference without ground-truth: samples latent z from prior and decodes pose.

        Args:
            condition (Tensor): Contextual condition feature [B, 768].

        Returns:
            dict: Converted MANO pose outputs including 3D joints and vertices.
        """
        bs = condition.shape[0]
        condition = condition.reshape(bs, -1)
        pred_pose = self.pose_vae.inference(condition)
        pred_pose = pred_pose.reshape(-1, 154)
        trans_pose = self.transfer(pred_pose)
        return trans_pose






