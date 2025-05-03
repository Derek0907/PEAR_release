from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from .heatmap_utils import compute_heatmap
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
import torch

class Ego4dDataset(Dataset):
    """
    Custom Dataset for Ego4D-based hand-object interaction anticipation.
    This dataset loads egocentric images, masks, text descriptions, 3D hand poses,
    trajectories, contact areas, and affordance maps. It supports tasks including
    pose prediction, trajectory forecasting, interaction hotspots prediction, 
    and hand contact area prediction.
    """
    def __init__(self, uid, img_path, mask_path, text, verb, noun, flag, para, pred_cam, 
                 keypoints_3d, gt_pose, aff_l, aff_r, area_l, area_r, f_area, 
                 f_traj, l_traj, r_traj, l_traj_pred, r_traj_pred, f_pose, transform=None):
        '''
        uid: index of the sample
        img_path: raw image of the sample
        mask_path: mask generated from LISA (using text and raw image)
        text: interaction prompt 
        verb: verb extracted from raw sentences
        noun: noun extracted from raw sentences
        flag: valid condition of interaction hotspots. 0 -> left | 1 -> right | 2 -> both
        para: MANO parameters generated from HaMeR
        pred_cam: pred_cam generated from HaMeR
        keypoints_3d: GT 3d_hand_points generated from HaMeR
        gt_pose: GT pose generated from HaMeR
        aff_l: GT left interaction hotspots
        aff_r: GT right interaction hotspots
        area_l: GT left hand contact
        area_r: GT right hand contact
        f_area: valid condition of hand contact || 1 -> valid | 0 -> not valid
        f_traj: valid condition of all hand trajectory (both pre and post contact) || 1 -> valid | 0 -> not valid
        l_traj: GT all left hand trajectory, including 7 points 
        r_traj: GT all right hand trajectory, including 7 points
        l_traj_pred: GT left hand trajectory, including 5 points, without initial hand position in the raw image & hand contact position (the last point of hand motion trend)
        r_traj_pred: GT left hand trajectory, including 5 points, without initial hand position in the raw image & hand contact position (the last point of hand motion trend) 
        f_pose: valid condition of hand pose || 0 -> left | 1 -> right | 2 -> both | -1 -> not valid
        transform: not used in this repo
        '''
        self.img_path = img_path
        self.mask_path = mask_path
        self.text = text
        self.verb = verb
        self.noun = noun
        self.transform = transform
        self.para = para
        self.pred_cam = pred_cam
        self.flag = flag
        self.aff_l = aff_l
        self.aff_r = aff_r
        self.uid = uid
        self.keypoints_3d = keypoints_3d
        self.gt_pose = gt_pose
        self.area_l = area_l
        self.area_r = area_r
        self.f_area = f_area
        self.f_traj = f_traj
        self.l_traj = l_traj
        self.r_traj = r_traj
        self.l_traj_pred = l_traj_pred
        self.r_traj_pred = r_traj_pred
        self.f_pose = f_pose

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.img_path)
    
    def load_demo_image(self, image_size, img_path):
        raw_image = Image.open(img_path).convert('RGB')   
        ori_shape = raw_image.size # (W, H)
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size),interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]) 
        image = transform(raw_image)
        return image, np.array(ori_shape)
    
    def process_mask(self, img, mask_path):
        """
        Apply a binary mask to the image, keeping only foreground regions.

        Args:
            img (Tensor): The input RGB image [3, H, W].
            mask_path (str): Path to the binary mask image.

        Returns:
            img_masked (Tensor): Image with background zeroed out.
        """
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (224, 224))
        mask = (mask > 0).astype(np.float32) # Assuming mask is grayscale: foreground is non-black (>0), background is black
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(img.device) 
        img_masked = img * mask_tensor  
        return img_masked
        
    def __getitem__(self, idx):
        """
        Load and process a single sample for model input.

        Returns:
            A tuple of image, heatmaps, trajectories, text, pose, and other metadata.
        """
        uid = self.uid[idx]
        # Load and preprocess image and mask
        img, ori_shape = self.load_demo_image(224, self.img_path[idx])
        raw_img = img.clone()
        img = self.process_mask(img, self.mask_path[idx])
        img = np.asarray(img, dtype=np.float32)
        
        small_shape = (ori_shape[0]//4, ori_shape[1]//4) # trajectory labels are generated from small shape images.
        text = self.text[idx]
        verb = self.verb[idx]
        noun = self.noun[idx]
        
        # flag read of each element
        f_pose = self.f_pose[idx]
        f_traj = self.f_traj[idx]
        f_area = self.f_area[idx]
        flag = self.flag[idx] 
        flag = np.array(flag, dtype=np.float32)     
        
        # GT pose
        para = self.para[idx]
        pred_cam = self.pred_cam[idx] 
        gt_pose = self.gt_pose[idx]
        keypoints_3d = self.keypoints_3d[idx]
        
        # GT hotpots and trajecotry
        l_traj = self.l_traj[idx]
        r_traj = self.r_traj[idx]
        l_traj_pred = self.l_traj_pred[idx]
        r_traj_pred = self.r_traj_pred[idx]

        l_traj_pred = l_traj_pred / small_shape
        r_traj_pred = r_traj_pred / small_shape
        l_traj = l_traj / small_shape
        r_traj = r_traj / small_shape

        l_init = l_traj[0, :]
        r_init = r_traj[0, :]
        
        l_init = np.expand_dims(l_init, axis=0)
        r_init = np.expand_dims(r_init, axis=0)
        
        hmap_l_init = compute_heatmap(l_init, (32, 32))
        hmap_r_init = compute_heatmap(r_init, (32, 32))
        
        # heatmap for hotspots
        hmap_l = compute_heatmap(self.aff_l[idx], (224, 224)) 
        hmap_r = compute_heatmap(self.aff_r[idx], (224, 224))
        
        #heatmap for the last point of hand motion trend
        hmap_l_small = compute_heatmap(self.aff_l[idx], (32, 32))
        hmap_r_small = compute_heatmap(self.aff_r[idx], (32, 32))
        
        #heatmap for the last point of hand motion trend
        hpos_last_l = self.aff_l[idx]
        hpos_last_r = self.aff_r[idx]
        
        # GT and predicted heatmap sequences
        l_traj_map, r_traj_map = [], []
        l_mani_map, r_mani_map = [], []
        
        # Trajectory heatmaps (first 2 GT points + contact point)
        for i in range(2):
            ll = l_traj_pred[i, :]
            rr = r_traj_pred[i, :]
            ll = np.expand_dims(ll, axis=0)
            rr = np.expand_dims(rr, axis=0)
            l_map = compute_heatmap(ll, (32, 32))
            r_map = compute_heatmap(rr, (32, 32))
            l_traj_map.append(l_map)
            r_traj_map.append(r_map)
            
        # Add final contact heatmap (from GT affordance point)
        l_traj_map.append(hmap_l_small)
        r_traj_map.append(hmap_r_small)
        
        # Manipulation phase (last 3 predicted points)
        for i in range(2, 5):
            ll = l_traj_pred[i, :]
            rr = r_traj_pred[i, :]
            ll = np.expand_dims(ll, axis=0)
            rr = np.expand_dims(rr, axis=0)
            l_map = compute_heatmap(ll, (32, 32))
            r_map = compute_heatmap(rr, (32, 32))
            l_mani_map.append(l_map)
            r_mani_map.append(r_map)
            
        # Final formatting    
        l_traj_map = np.array(l_traj_map)
        r_traj_map = np.array(r_traj_map)
        l_mani_map = np.array(l_mani_map, dtype=np.float32)
        r_mani_map = np.array(r_mani_map, dtype=np.float32)

        # GT hand contact area
        area_l = self.area_l[idx]
        area_r = self.area_r[idx]
        
        return (
        uid, flag, img, raw_img, text, verb, noun, para, pred_cam, keypoints_3d, gt_pose,
        hmap_l, hmap_r, hpos_last_l, hpos_last_r, area_l, area_r, ori_shape, f_area, f_traj,
        l_traj, r_traj, hmap_l_init, hmap_r_init, 
        l_traj_map, r_traj_map,
        l_mani_map, r_mani_map, f_pose
        )