import torch
import numpy as np
from typing import Optional, Dict, List, Tuple
import torch.nn as nn


def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """

    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1**2).sum(dim=(1,2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1, X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=U.device).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale*torch.matmul(R, mu1)

    # 7. Error:
    S1_hat = scale*torch.matmul(R, S1) + t

    return S1_hat.permute(0, 2, 1)

def reconstruction_error(S1, S2) -> np.array:
    """
    Computes mean per-joint error (PA-MPJPE) after Procrustes alignment.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (np.array): Reconstruction error.
    """
    S1_hat = compute_similarity_transform(S1, S2)
    re = torch.sqrt( ((S1_hat - S2)** 2).sum(dim=-1)).mean(dim=-1)
    return re

def eval_pose(pred_joints, gt_joints) -> Tuple[np.array, np.array]:
    """
    Compute joint errors in mm before and after Procrustes alignment.
    Args:
        pred_joints (torch.Tensor): Predicted 3D joints of shape (B, N, 3).
        gt_joints (torch.Tensor): Ground truth 3D joints of shape (B, N, 3).
    Returns:
        Tuple[np.array, np.array]: Joint errors in mm before and after alignment.
    """
    # Absolute error (MPJPE)
    mpjpe = torch.sqrt(((pred_joints - gt_joints) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()

    # Reconstruction_error
    r_error = reconstruction_error(pred_joints, gt_joints).cpu().numpy()
    return 1000 * mpjpe, 1000 * r_error

class Evaluator_pose(nn.Module):
    """
    Module to evaluate 3D hand pose predictions with MPJPE and PA-MPJPE metrics.
    """
    def __init__(self):
        super(Evaluator_pose, self).__init__()

    def forward(self, out_left, out_right, gt, lr, f_pose):
        """
        Evaluate batch of predicted 3D poses against ground truth.

        Args:
            out_left (dict): Contains 'pred_keypoints_3d' for left hand, shape [B, 21, 3]
            out_right (dict): Contains 'pred_keypoints_3d' for right hand, shape [B, 21, 3]
            gt (torch.Tensor): Ground truth keypoints, shape [B, 2, 21, 3]
            lr (torch.Tensor): Hand type indicator per sample: 0=left, 1=right, 2=both
            f_pose (torch.Tensor): Validity flag: -1 = invalid, else = valid

        Returns:
            Tuple of (total MPJPE in mm, total PA-MPJPE in mm, valid sample count)
        """
        joint_left = out_left['pred_keypoints_3d']  # [B, 21, 3]
        joint_right = out_right['pred_keypoints_3d']  # [B, 21, 3]
        bs = joint_left.shape[0]
        re_all = 0
        mpjpe_all = 0
        count = 0
        
        for idx in range(bs):
            hand_type = int(lr[idx])          # 0 = left, 1 = right, 2 = both
            is_valid = f_pose[idx] != -1      # Only evaluate if pose is valid
            if is_valid:
                joint_gt_left = gt[idx][0]
                joint_gt_right = gt[idx][1]
                if hand_type == 2:
                    mpjpe_left, re_left = eval_pose(joint_left[idx].reshape(-1,21,3), joint_gt_left.reshape(-1,21,3))
                    mpjpe_right, re_right = eval_pose(joint_right[idx].reshape(-1,21,3), joint_gt_right.reshape(-1,21,3))
                    mpjpe = (mpjpe_left + mpjpe_right)
                    re = (re_left + re_right) 
                    count += 2
                elif hand_type == 1:
                    mpjpe, re = eval_pose(joint_right[idx].reshape(-1,21,3), joint_gt_right.reshape(-1,21,3))
                    count += 1
                elif hand_type == 0:
                    mpjpe, re = eval_pose(joint_left[idx].reshape(-1,21,3), joint_gt_left.reshape(-1,21,3))
                    count += 1
                mpjpe_all += mpjpe
                re_all += re
        return mpjpe_all, re_all, count




