import torch
import torch.nn as nn
import torch.nn.functional as F
from yacs.config import CfgNode
from .pose_decoder import Pose_decoder
from .blip.blip import blip_feature_extractor
from .segformer.segformer import SegFormer
from .hotspots_decoder import Hotspots_decoder
from .area_decoder import Area_decoder
from .fusion.hot_pose import Fusion_hot_pose
from .fusion.norm_feat import Norm_Feature_free
from .fusion.same_uni import fusion_same_uni
from .DEQfusion import DEQFusion
from .traj_decoder_l1 import Traj_decoder_l1
from .traj_decoder_l2 import Traj_decoder_l2
from .fusion.weight_sum import Sum_weights
from .fusion.attentionwithsplit import Att_split
from .fusion.traj_int import fusion_traj_int
from .fusion.text_enhance import text_image_cross_align

class Net(nn.Module):
    def __init__(self, cfg: CfgNode):
        super(Net, self).__init__()

        # =================== Config and backbones =================== #
        self.img_size = cfg.BLIP.IMAGE_SIZE
        self.blip_model = blip_feature_extractor(
            pretrained= cfg.BLIP.WEIGHTS,
            image_size=self.img_size,
            vit='base'
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = cfg
        self.channel_dim = 512

        # =================== Multi-branch modules =================== #
        self.att_split = Att_split(input_dim=768, output_dim1=768, output_dim2=768)

        # Hotspots decoders
        self.hotspots_decoder_l = Hotspots_decoder(input_dim=224 * 224, condition_dim=224 * 224)
        self.hotspots_decoder_r = Hotspots_decoder(input_dim=224 * 224, condition_dim=224 * 224)

        # Pose decoders
        self.pose_decoder_l = Pose_decoder(self.channel_dim, cfg=self.cfg)
        self.pose_decoder_r = Pose_decoder(self.channel_dim, cfg=self.cfg)

        # Area decoders
        self.area_decoder_l = Area_decoder(cfg, input_dim=778, out_dim=778)
        self.area_decoder_r = Area_decoder(cfg, input_dim=778, out_dim=778)

        # Unused branch
        self.fus_hot_traj = Fusion_hot_pose()  # no use in this version
        self.weight_sum = Sum_weights() # no use in this version
        
        # Feature fusion modules
        self.Norm_feat = Norm_Feature_free(768)
        self.deq = DEQFusion(channel_dim=768, num_modals=2)

        # Trajectory decoders (Hand motion trend -> l1 & Manipulation trajecotry -> l2)
        self.traj_decoder_l1 = Traj_decoder_l1()
        self.traj_decoder_l2 = Traj_decoder_l2()
        self.traj_decoder_r1 = Traj_decoder_l1()
        self.traj_decoder_r2 = Traj_decoder_l2()

        # Vision encoder
        self.vision_back = SegFormer(num_classes=1, phi='b2', pretrained=True, weights=self.cfg.BLIP.SEG_WEIGHTS)

        # Cross-modality fusion
        self.same_uni_hot = fusion_same_uni()
        self.fus_traj_int = fusion_traj_int()
        self.text_img_enhance = text_image_cross_align()

    def forward(self, image, raw_img, verb,
                gt_hotspots_l, gt_hotspots_r, gt_pose,
                gt_area_l, gt_area_r, hmap_l, hmap_r,
                gt_traj_l, gt_traj_r, gt_mani_l, gt_mani_r):
        """
        Forward training mode: all modules conditioned on ground truth.
        """
        mask_img = image
        feature_list = []
        B = image.shape[0]

        mask_feat = self.vision_back(mask_img)  # [bs, 1, 224, 224]
        verb_feat = self.blip_model(raw_img, verb, mode='multimodal')  # [bs, 32, 768]
        hot_feat, traj_feat, verb_feat = self.text_img_enhance(mask_feat, verb_feat)  # [bs, 1, 224, 224]
        hot_feat, traj_feat = self.Norm_feat(hot_feat, traj_feat)

        feature_list.append(hot_feat)
        feature_list.append(traj_feat)

        int_feat, jac_loss, _ = self.deq(feature_list)  # [bs, 768]
        pose_feat, area_feat, mani_feat = self.att_split(int_feat)  # [bs, 768]

        int_feat = int_feat.unsqueeze(1)  # [bs, 1, 768]

        # -------- Pose Prediction -------- #
        gt_pose_l = gt_pose[:, 0, :]
        gt_pose_r = gt_pose[:, 1, :]
        pred_handpose_l, recon_pose_l, KLD_pose_l, pose_l = self.pose_decoder_l(gt_pose_l, pose_feat)
        pred_handpose_r, recon_pose_r, KLD_pose_r, pose_r = self.pose_decoder_r(gt_pose_r, pose_feat)

        # -------- Area Prediction -------- #
        pred_area_l, recon_area_l, KLD_area_l = self.area_decoder_l(gt_area_l, area_feat)
        pred_area_r, recon_area_r, KLD_area_r = self.area_decoder_r(gt_area_r, area_feat)

        # -------- Hotspots Prediction -------- #
        hot_condition = self.same_uni_hot(mask_feat, int_feat).reshape(B, 224, 224)
        pred_hotspots_l, recon_hotspots_l, KLD_hotspots_l = self.hotspots_decoder_l(gt_hotspots_l, hot_condition)
        pred_hotspots_r, recon_hotspots_r, KLD_hotspots_r = self.hotspots_decoder_r(gt_hotspots_r, hot_condition)

        # -------- Hand Motion Trend Prediction -------- #
        traj_condition = self.fus_traj_int(verb_feat, int_feat)
        gt_traj_l = gt_traj_l.float()
        gt_traj_r = gt_traj_r.float()

        pred_traj_l, recon_traj_l, KLD_traj_l = self.traj_decoder_l1(gt_traj_l, hmap_l, traj_condition, pred_hotspots_l)
        pred_traj_r, recon_traj_r, KLD_traj_r = self.traj_decoder_r1(gt_traj_r, hmap_r, traj_condition, pred_hotspots_r)

        # -------- Manipulation Trajectory Prediction -------- #
        pred_mani_l, recon_mani_l, KLD_mani_l = self.traj_decoder_l2(gt_mani_l, mani_feat, pred_hotspots_l)
        pred_mani_r, recon_mani_r, KLD_mani_r = self.traj_decoder_r2(gt_mani_r, mani_feat, pred_hotspots_r)

        return pred_hotspots_l, pred_hotspots_r, recon_hotspots_l, recon_hotspots_r, \
               KLD_hotspots_l, KLD_hotspots_r, \
               pred_handpose_l, pred_handpose_r, recon_pose_l, recon_pose_r, \
               KLD_pose_l, KLD_pose_r, \
               pred_area_l, pred_area_r, recon_area_l, recon_area_r, \
               KLD_area_l, KLD_area_r, \
               pred_traj_l, recon_traj_l, KLD_traj_l, \
               pred_traj_r, recon_traj_r, KLD_traj_r, \
               pred_mani_l, recon_mani_l, KLD_mani_l, \
               pred_mani_r, recon_mani_r, KLD_mani_r

    def inference(self, image, raw_img, verb, hmap_l, hmap_r):
        """
        Inference mode: no ground truth used, sampling-based prediction.
        """
        mask_img = image
        feature_list = []
        B = image.shape[0]
        mask_feat = self.vision_back(mask_img)
        verb_feat = self.blip_model(raw_img, verb, mode='multimodal')
        hot_feat, traj_feat, verb_feat = self.text_img_enhance(mask_feat, verb_feat)
        hot_feat, traj_feat = self.Norm_feat(hot_feat, traj_feat)
        feature_list.append(hot_feat)
        feature_list.append(traj_feat)

        int_feat, jac_loss, _ = self.deq(feature_list)
        pose_feat, area_feat, mani_feat = self.att_split(int_feat)
        int_feat = int_feat.unsqueeze(1)
        pred_handpose_l = self.pose_decoder_l.inference(pose_feat)
        pred_handpose_r = self.pose_decoder_r.inference(pose_feat)

        pred_area_l = self.area_decoder_l.inference(area_feat)
        pred_area_r = self.area_decoder_r.inference(area_feat)

        hot_condition = self.same_uni_hot(mask_feat, int_feat).reshape(B, 224, 224)
        pred_hotspots_l = self.hotspots_decoder_l.inference(hot_condition)
        pred_hotspots_r = self.hotspots_decoder_r.inference(hot_condition)
        
        traj_condition = self.fus_traj_int(verb_feat, int_feat)
        pred_traj_l = self.traj_decoder_l1.inference(hmap_l, traj_condition, pred_hotspots_l)
        pred_traj_r = self.traj_decoder_r1.inference(hmap_r, traj_condition, pred_hotspots_r)

        pred_mani_l = self.traj_decoder_l2.inference(mani_feat, pred_hotspots_l)
        pred_mani_r = self.traj_decoder_r2.inference(mani_feat, pred_hotspots_r)

        pred_mani_l_small = F.interpolate(pred_mani_l, size=(32, 32), mode='bilinear')
        pred_mani_r_small = F.interpolate(pred_mani_r, size=(32, 32), mode='bilinear')

        pred_traj_all_l = torch.cat([pred_traj_l, pred_mani_l_small], dim=1)
        pred_traj_all_r = torch.cat([pred_traj_r, pred_mani_r_small], dim=1)

        return pred_hotspots_l, pred_hotspots_r, \
               pred_handpose_l, pred_handpose_r, \
               pred_area_l, pred_area_r, \
               pred_traj_l, pred_traj_r, \
               pred_mani_l, pred_mani_r, \
               pred_traj_all_l, pred_traj_all_r
