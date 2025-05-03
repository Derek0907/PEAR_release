import numpy as np
import torch.nn as nn
    
class Hmaptraj_Metric(nn.Module):
    """
    Metric class for evaluating predicted hand trajectory heatmaps using
    Average Displacement Error (ADE) and Final Displacement Error (FDE).
    This evaluates the initial 3-step trajectory by converting heatmaps
    to coordinates and comparing against ground truth positions.
    """
    def __init__(self):
        super(Hmaptraj_Metric, self).__init__()

    def ade(self, pred_traj, gt_traj):
        
        """
        Compute Average Displacement Error (ADE).

        Args:
            pred_traj (np.ndarray): Predicted trajectory points [T, 2].
            gt_traj (np.ndarray): Ground truth trajectory points [T, 2].

        Returns:
            float: Mean Euclidean distance across all timesteps.
        """
        
        diff = pred_traj - gt_traj    # shape [timestep, 2]
        ade = np.linalg.norm(diff, axis=1).mean()
        return ade

    
    def fde(self, pred_traj, gt_traj):

        """
        Compute Final Displacement Error (FDE).

        Args:
            pred_traj (np.ndarray): Predicted trajectory [T, 2].
            gt_traj (np.ndarray): Ground truth trajectory [T, 2].

        Returns:
            float: Euclidean distance at final timestep.
        """

        p = pred_traj[-1]
        g = gt_traj[-1]
        fde = np.linalg.norm(p - g)
        return fde
    
    def find_max_value_coordinate(self, heatmap):
        
        """
        Extract the normalized coordinate (x, y) of the max value in a heatmap.

        Args:
            heatmap (torch.Tensor): Heatmap [H, W].

        Returns:
            np.ndarray: Normalized coordinate [x/w, y/h].
        """
        
        heatmap = heatmap.detach().cpu().numpy()
        h, w = heatmap.shape
        y, x = np.where(heatmap == np.max(heatmap))
        max_loc = (x[0] / w, y[0] / h)
        return np.array(max_loc)
    

       
    def forward(self, pred_traj_l, pred_traj_r, gt_hpos_last_l, gt_hpos_last_r, flag_traj, lr, gt_l_pos, gt_r_pos):
        
        """
        Evaluate predicted initial hand motion trend heatmaps against ground truth keypoints.
        Three steps are evaluated: two from ground truth positions and one from the hotspot point.

        Args:
            pred_traj_l (Tensor): Predicted left-hand trajectory heatmaps [B, 3, 32, 32].
            pred_traj_r (Tensor): Predicted right-hand trajectory heatmaps [B, 3, 32, 32].
            gt_hpos_last_l (Tensor): Ground truth left-hand contact position [B, 2].
            gt_hpos_last_r (Tensor): Ground truth right-hand contact position [B, 2].
            flag_traj (Tensor): Indicates whether trajectory prediction is valid for each sample [B].
            lr (Tensor): Indicates which hand is used (0=left, 1=right, 2=both) [B].
            gt_l_pos (Tensor): Full ground truth left-hand trajectory [B, 7, 2].
            gt_r_pos (Tensor): Full ground truth right-hand trajectory [B, 7, 2].

        Returns:
            Tuple: (total_ADE, total_FDE, total_valid_sample_count)
        """
        
        bs = pred_traj_l.shape[0]
        ade_all, fde_all, count = 0, 0, 0
        
        for i in range(bs):
            if flag_traj[i] == 1:
                l_traj_p, r_traj_p = [], []
                l_traj_g, r_traj_g = [], []
                
                for j in range(3):  # 3 timesteps
                    l_traj_p.append(self.find_max_value_coordinate(pred_traj_l[i][j]))
                    r_traj_p.append(self.find_max_value_coordinate(pred_traj_r[i][j]))

                l_traj_g_first_two = gt_l_pos[i, 1:3, :].detach().cpu().numpy() 
                r_traj_g_first_two = gt_r_pos[i, 1:3, :].detach().cpu().numpy()
                l_traj_g_third = gt_hpos_last_l[i].detach().cpu().numpy()
                r_traj_g_third = gt_hpos_last_r[i].detach().cpu().numpy() 
                l_traj_g = np.vstack([l_traj_g_first_two, l_traj_g_third]) 
                r_traj_g = np.vstack([r_traj_g_first_two, r_traj_g_third])

                l_traj_p, r_traj_p = np.array(l_traj_p), np.array(r_traj_p)


                ade_l, ade_r = self.ade(l_traj_p, l_traj_g), self.ade(r_traj_p, r_traj_g)
                fde_l, fde_r = self.fde(l_traj_p, l_traj_g), self.fde(r_traj_p, r_traj_g)

                if lr[i] == 0:
                    ade_all += ade_l
                    fde_all += fde_l
                    count += 1
                elif lr[i] == 1:
                    ade_all += ade_r
                    fde_all += fde_r
                    count += 1
                elif lr[i] == 2:
                    ade_all += ade_l + ade_r
                    fde_all += fde_l + fde_r
                    count += 2

        return ade_all, fde_all, count


class Hmapmani_Metric(nn.Module):
    """
    Metric class for evaluating manipulation (post-contact) trajectory predictions
    using ADE and FDE, based on heatmap outputs for each timestep.
    """
    def __init__(self):
        super(Hmapmani_Metric, self).__init__()

    def ade(self, pred_traj, gt_traj):
        diff = pred_traj - gt_traj    # shape [timestep, 2]
        ade = np.linalg.norm(diff, axis=1).mean()
        return ade

    
    def fde(self, pred_traj, gt_traj):
        p, g = pred_traj[-1], gt_traj[-1]
        return np.linalg.norm(p - g)
    
    def find_max_value_coordinate(self, heatmap):
        heatmap = heatmap.detach().cpu().numpy()
        h, w = heatmap.shape
        y, x = np.where(heatmap == np.max(heatmap))
        return np.array([x[0] / w, y[0] / h])

       
    def forward(self, pred_traj_l, pred_traj_r, gt_traj_l, gt_traj_r, flag_traj, lr):
        """
        Evaluate post-contact (manipulation) trajectory predictions.

        For each sample, extract the maximum point from each predicted heatmap (3 steps),
        and compare to ground truth trajectory points at timesteps 4, 5, 6 (post-contact).

        Args:
            pred_traj_l (Tensor): Left hand predicted manipulation heatmaps [B, 3, 32, 32].
            pred_traj_r (Tensor): Right hand predicted manipulation heatmaps [B, 3, 32, 32].
            gt_traj_l (Tensor): Ground truth left-hand full trajectory [B, 7, 2].
            gt_traj_r (Tensor): Ground truth right-hand full trajectory [B, 7, 2].
            flag_traj (Tensor): Indicates whether trajectory is valid for each sample [B].
            lr (Tensor): Indicates which hand is involved (0=left, 1=right, 2=both) [B].

        Returns:
            Tuple: (total_ADE, total_FDE, total_valid_sample_count)
        """
        
        bs = pred_traj_l.shape[0]
        ade_all, fde_all, count = 0, 0, 0

        for i in range(bs):
            if flag_traj[i] == 1:
                l_traj_p, r_traj_p = [], []
                l_traj_g, r_traj_g = [], []

                for j in range(3):  # Manipulation frames
                    l_traj_p.append(self.find_max_value_coordinate(pred_traj_l[i][j]))
                    r_traj_p.append(self.find_max_value_coordinate(pred_traj_r[i][j]))

                l_traj_p, r_traj_p = np.array(l_traj_p), np.array(r_traj_p)
                l_traj_g = gt_traj_l[i, 4:7, :].detach().cpu().numpy()
                r_traj_g = gt_traj_r[i, 4:7, :].detach().cpu().numpy()
                ade_l, ade_r = self.ade(l_traj_p, l_traj_g), self.ade(r_traj_p, r_traj_g)
                fde_l, fde_r = self.fde(l_traj_p, l_traj_g), self.fde(r_traj_p, r_traj_g)

                if lr[i] == 0:
                    ade_all += ade_l
                    fde_all += fde_l
                    count += 1
                elif lr[i] == 1:
                    ade_all += ade_r
                    fde_all += fde_r
                    count += 1
                elif lr[i] == 2:
                    ade_all += ade_l + ade_r
                    fde_all += fde_l + fde_r
                    count += 2

        return ade_all, fde_all, count