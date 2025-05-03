import torch
import torch.nn as nn

class Keypoint3DLoss(nn.Module):
    def __init__(self, loss_type: str = 'l1'):
        """
        Loss for 3D hand keypoints regression.
        Supports both L1 and L2 loss.

        Args:
            loss_type (str): 'l1' for L1 loss, 'l2' for MSE loss.
        """
        super().__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='mean')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError

    def forward(self, pred_left, pred_right, gt, lr, f_pose):
        """
        Compute 3D keypoint loss for left and right hands.

        Args:
            pred_left/right: dict with key 'pred_keypoints_3d' of shape [B, 21, 3]
            gt: [B, 2, 21, 3], ground truth keypoints (left, right)
            lr: hand flag per sample (0: left, 1: right, 2: both)
            f_pose: validity flag per sample (-1: invalid)

        Returns:
            loss_all: aggregated 3D joint loss
            count: number of valid hand predictions used
        """
        joint_left = pred_left['pred_keypoints_3d']
        joint_right = pred_right['pred_keypoints_3d']
        point_loss = self.loss_fn
        bs = joint_left.shape[0]
        count = 0
        loss_all = torch.tensor(0.0, device=joint_left.device)

        for idx in range(bs):
            p = int(lr[idx])
            f_p = f_pose[idx]
            if f_p != -1:
                if p == 2:
                    count += 2
                    loss_joint = point_loss(joint_left[idx], gt[idx][0]) + \
                                 point_loss(joint_right[idx], gt[idx][1])
                elif p == 1:
                    count += 1
                    loss_joint = point_loss(joint_right[idx], gt[idx][1])
                elif p == 0:
                    count += 1
                    loss_joint = point_loss(joint_left[idx], gt[idx][0])
                loss_all += loss_joint

        return loss_all, count

    
class Pose_Loss(nn.Module):
    def __init__(self, alpha=1):
        """
        VAE-style loss for pose parameter reconstruction.
        Includes reconstruction loss (sum over 154-dim MANO parameters) and KL divergence.
        """
        super(Pose_Loss, self).__init__()
        self.alpha = alpha

    def forward(self, recon_l, recon_r, KLD_l, KLD_r, lr, f_pose):
        """
        Args:
            recon_l/r: reconstruction loss tensors per hand
            KLD_l/r: KL divergence tensors per hand
            lr: left/right flag (0/1/2)
            f_pose: valid flag (-1 if invalid)

        Returns:
            normalized reconstruction loss, KL loss, valid sample count
        """
        recon_loss_all = torch.tensor(0.0, device=recon_l.device)
        KLD_loss_all = torch.tensor(0.0, device=recon_l.device)
        count = 0

        for idx in range(recon_l.shape[0]):
            p = int(lr[idx])
            f_p = f_pose[idx]
            if f_p != -1:
                if p == 2:
                    recon_loss_all += recon_l[idx].sum() + recon_r[idx].sum()
                    KLD_loss_all += KLD_l[idx].sum() + KLD_r[idx].sum()
                    count += 2
                elif p == 1:
                    recon_loss_all += recon_r[idx].sum()
                    KLD_loss_all += KLD_r[idx].sum()
                    count += 1
                elif p == 0:
                    recon_loss_all += recon_l[idx].sum()
                    KLD_loss_all += KLD_l[idx].sum()
                    count += 1

        return recon_loss_all / 154, KLD_loss_all, count
    
    
class Area_Loss(nn.Module):
    def __init__(self, alpha=1):
        """
        Binary contact area loss for 778 hand vertices.
        Computes VAE-style loss (reconstruction + KL) over contact maps.
        """
        super(Area_Loss, self).__init__()
        self.alpha = alpha

    def forward(self, recon_l, recon_r, KLD_l, KLD_r, lr, f_area):
        recon_loss_all = torch.tensor(0.0, device=recon_l.device)
        KLD_loss_all = torch.tensor(0.0, device=recon_l.device)
        count = 0

        for idx in range(recon_l.shape[0]):
            p = int(lr[idx])
            f_a = f_area[idx]
            if f_a == 1:
                if p == 2:
                    recon_loss_all += recon_l[idx].sum() + recon_r[idx].sum()
                    KLD_loss_all += KLD_l[idx].sum() + KLD_r[idx].sum()
                    count += 2
                elif p == 1:
                    recon_loss_all += recon_r[idx].sum()
                    KLD_loss_all += KLD_r[idx].sum()
                    count += 1
                elif p == 0:
                    recon_loss_all += recon_l[idx].sum()
                    KLD_loss_all += KLD_l[idx].sum()
                    count += 1

        return recon_loss_all / 778, KLD_loss_all, count
        

class Hot_Loss(nn.Module):
    def __init__(self, alpha=1):
        """
        VAE-style heatmap loss for hotspot prediction.
        Assumes output map size is 224x224.
        """
        super(Hot_Loss, self).__init__()
        self.alpha = alpha

    def forward(self, recon_l, recon_r, KLD_l, KLD_r, lr):
        recon_loss_all = torch.tensor(0.0, device=recon_l.device)
        KLD_loss_all = torch.tensor(0.0, device=recon_l.device)
        count = 0

        for idx in range(recon_l.shape[0]):
            p = int(lr[idx])
            if p == 2:
                recon_loss_all += recon_l[idx].sum() + recon_r[idx].sum()
                KLD_loss_all += KLD_l[idx].sum() + KLD_r[idx].sum()
                count += 2
            elif p == 1:
                recon_loss_all += recon_r[idx].sum()
                KLD_loss_all += KLD_r[idx].sum()
                count += 1
            elif p == 0:
                recon_loss_all += recon_l[idx].sum()
                KLD_loss_all += KLD_l[idx].sum()
                count += 1

        return recon_loss_all / (224 * 224), KLD_loss_all, count
    
class Traj_loss(nn.Module):
    def __init__(self, alpha=1):
        """
        VAE-style heatmap loss for hotspot prediction.
        Assumes output map size is 224x224.
        """
        super(Traj_loss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, recon_l, recon_r, KLD_l, KLD_r, flag, lr):
        recon_loss_all = torch.tensor(0.0, device=recon_l.device)
        KLD_loss_all = torch.tensor(0.0, device=recon_l.device)
        count = 0

        for i in range(recon_l.shape[0]):
            ll = lr[i]
            ff = flag[i]
            if ff == 1:
                if ll == 0:
                    recon_loss_all += recon_l[i].sum()
                    KLD_loss_all += KLD_l[i].sum()
                    count += 1
                elif ll == 1:
                    recon_loss_all += recon_r[i].sum()
                    KLD_loss_all += KLD_r[i].sum()
                    count += 1
                elif ll == 2:
                    recon_loss_all += recon_l[i].sum() + recon_r[i].sum()
                    KLD_loss_all += KLD_l[i].sum() + KLD_r[i].sum()
                    count += 2

        return recon_loss_all / (32 * 32), KLD_loss_all, count
    
class Mani_loss(nn.Module):
    def __init__(self, alpha=1):
        """
        VAE-style loss for manipulation trajectory maps.
        Also assumes 32x32 resolution.
        """
        super(Mani_loss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, recon_l, recon_r, KLD_l, KLD_r, flag, lr):
        recon_loss_all = torch.tensor(0.0, device=recon_l.device)
        KLD_loss_all = torch.tensor(0.0, device=recon_l.device)
        count = 0

        for i in range(recon_l.shape[0]):
            ll = lr[i]
            ff = flag[i]
            if ff == 1:
                if ll == 0:
                    recon_loss_all += recon_l[i].sum()
                    KLD_loss_all += KLD_l[i].sum()
                    count += 1
                elif ll == 1:
                    recon_loss_all += recon_r[i].sum()
                    KLD_loss_all += KLD_r[i].sum()
                    count += 1
                elif ll == 2:
                    recon_loss_all += recon_l[i].sum() + recon_r[i].sum()
                    KLD_loss_all += KLD_l[i].sum() + KLD_r[i].sum()
                    count += 2

        return recon_loss_all / (32 * 32), KLD_loss_all, count



        