import torch
import pytorch_lightning as pl
from typing import Any, Dict, Mapping, Tuple
import pickle
from yacs.config import CfgNode
import sys
from .utils import SkeletonRenderer, MeshRenderer
from .utils.geometry import aa_to_rotmat, perspective_projection
from .utils.pylogger import get_pylogger
from .mano_wrapper import MANO

log = get_pylogger(__name__)

class MANOHandModel(pl.LightningModule):

    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        """
        Setup HAMER model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])

        self.cfg = cfg

        mano_cfg = {k.lower(): v for k,v in dict(cfg.MANO).items()}

        self.mano = MANO(**mano_cfg)
        

        if init_renderer:
            self.renderer = SkeletonRenderer(self.cfg)
            self.mesh_renderer = MeshRenderer(self.cfg, faces=self.mano.faces)
        else:
            self.renderer = None
            self.mesh_renderer = None



    def forward(self, batch, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """

        # Use RGB image as input
        device =  "cuda:0"
        batch_size = batch['global_orient'].shape[0]
        
        # Store useful regression outputs to the output dict
        pred_mano_params = {}
        output = {}
        # Compute model vertices, joints and the projected joints
        pred_mano_params['global_orient'] = batch['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['hand_pose'] = batch['hand_pose'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['betas'] = batch['betas'].reshape(batch_size, -1)
    
        mano_output = self.mano(**{k: v.float() for k,v in pred_mano_params.items()}, pose2rot=False)
        
        pred_keypoints_3d = mano_output.joints
        pred_vertices = mano_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        # print("mano_out:")
        # print(mano_output)
        return output, mano_output
    



