from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class Evaluator_area(nn.Module):
    
    """
    Evaluator for hand contact area predictions.
    Calculates standard classification metrics between predicted and ground truth contact regions.
    """
    
    def __init__(self):
        super(Evaluator_area, self).__init__()

    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
        
        """
        Compute classification metrics: accuracy, precision, recall, F1 score.

        Args:
            y_true (np.ndarray): Ground truth binary contact area.
            y_pred (np.ndarray): Predicted binary contact area.

        Returns:
            Tuple: (accuracy, precision, recall, F1 score)
        """
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return accuracy, precision, recall, f1

    def binalize_area(self, area: torch.Tensor) -> torch.Tensor:

        """
        Binarize the area tensor using a threshold of 0.5.

        Args:
            area (torch.Tensor): Continuous-valued area tensor.

        Returns:
            torch.Tensor: Binarized tensor (0 or 1).
        """
        
        return (area > 0.5).float()
    
    def forward(self, out_left: torch.Tensor, out_right: torch.Tensor,
                gt_l: torch.Tensor, gt_r: torch.Tensor,
                lr: torch.Tensor, f_area: torch.Tensor) -> Tuple[float, float, float, float, int]: 
        
        """
        Evaluate batch of predicted contact areas against ground truth.

        Args:
            out_left (torch.Tensor): Predicted contact areas for left hand (B, ...).
            out_right (torch.Tensor): Predicted contact areas for right hand (B, ...).
            gt_l (torch.Tensor): Ground truth contact areas for left hand (B, ...).
            gt_r (torch.Tensor): Ground truth contact areas for right hand (B, ...).
            lr (torch.Tensor): Hand flag indicator (0: left, 1: right, 2: both).
            f_area (torch.Tensor): Validity of area labels (1: valid, 0: invalid).

        Returns:
            Tuple: (total accuracy, precision, recall, F1, valid sample count)
        """
        
        bs = out_left.shape[0]
        
        # Accumulate metrics
        acc_all, pre_all, rec_all, f1_all, valid_count = 0, 0, 0, 0, 0
        
        for idx in range(bs):
            hand_flag = int(lr[idx])
            p_l = self.binalize_area(out_left[idx]).cpu().numpy()
            p_r = self.binalize_area(out_right[idx]).cpu().numpy()
            g_l = gt_l[idx].cpu().numpy()
            g_r = gt_r[idx].cpu().numpy()
            is_valid = f_area[idx]

            if is_valid == 1:
                if hand_flag == 0:  # Left hand
                    acc, pre, rec, f1 = self.compute_metrics(g_l, p_l)
                    valid_count += 1
                elif hand_flag == 1:  # Right hand
                    acc, pre, rec, f1 = self.compute_metrics(g_r, p_r)
                    valid_count += 1
                else:  # Both hands
                    acc_l, pre_l, rec_l, f1_l = self.compute_metrics(g_l, p_l)
                    acc_r, pre_r, rec_r, f1_r = self.compute_metrics(g_r, p_r)
                    acc = acc_l + acc_r
                    pre = pre_l + pre_r
                    rec = rec_l + rec_r
                    f1 = f1_l + f1_r
                    valid_count += 2
            else:
                acc = pre = rec = f1 = 0  # Invalid sample contributes zero

            acc_all += acc
            pre_all += pre
            rec_all += rec
            f1_all += f1
        
        return acc_all, pre_all, rec_all, f1_all, valid_count
