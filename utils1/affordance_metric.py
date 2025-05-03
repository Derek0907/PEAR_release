import torch.nn as nn
import numpy as np
import cv2

def compute_heatmap(points, image_size, k_ratio=3, transpose=False):
    
    """
    Generate a Gaussian-smoothed heatmap from a list of 2D normalized points.

    Args:
        points (list or np.ndarray): List of normalized (x, y) coordinates.
        image_size (tuple): Size of the heatmap (height, width).
        k_ratio (int): Ratio to control Gaussian kernel size.
        transpose (bool): Whether to transpose the output heatmap.

    Returns:
        np.ndarray: Heatmap of shape (height, width).
    """
    
    points = np.asarray(points)
    height, width = image_size
    heatmap = np.zeros((height, width), dtype=np.float32)
    n_points = points.shape[0]

    for i in range(n_points):
        x = points[i, 0] * width
        y = points[i, 1] * height
        col = int(x)
        row = int(y)
        try:
            heatmap[row, col] += 1.0
        except:
            col = min(max(col, 0), width - 1)
            row = min(max(row, 0), height - 1)
            heatmap[row, col] += 1.0

    k_size = int(np.sqrt(height * width) / k_ratio)
    if k_size % 2 == 0:
        k_size += 1
    heatmap = cv2.GaussianBlur(heatmap, (k_size, k_size), 0)

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    if transpose:
        heatmap = heatmap.transpose()

    return heatmap

def SIM(map1, map2, eps=1e-12):
    
    """
    Similarity metric based on intersection over normalized maps.

    Args:
        map1, map2 (np.ndarray): Heatmaps to compare.
        eps (float): Small epsilon to avoid division by zero.

    Returns:
        float: Similarity score.
    """
    
    map1, map2 = map1 / (map1.sum() + eps), map2 / (map2.sum() + eps)
    intersection = np.minimum(map1, map2)
    return np.sum(intersection)

def AUC_Judd(saliency_map, fixation_map, jitter=True):
    
    """
    AUC-Judd metric for saliency evaluation.

    Args:
        saliency_map (np.ndarray): Predicted saliency map.
        fixation_map (np.ndarray): Binary ground truth fixation map.
        jitter (bool): Add noise to break ties.

    Returns:
        float: AUC score.
    """
    
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5
    if not np.any(fixation_map):
        return np.nan
    if saliency_map.shape != fixation_map.shape:
        saliency_map = cv2.resize(saliency_map, fixation_map.shape)
    if jitter:
        saliency_map += np.random.rand(*saliency_map.shape) * 1e-7
    saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map) + 1e-12)

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F]
    n_fix = len(S_fix)
    n_pixels = len(S)
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds) + 2)
    fp = np.zeros(len(thresholds) + 2)
    tp[0] = 0; tp[-1] = 1
    fp[0] = 0; fp[-1] = 1
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh)
        tp[k+1] = (k + 1) / float(n_fix)
        fp[k+1] = (above_th - k - 1) / float(n_pixels - n_fix)
    return np.trapz(tp, fp)

def NSS(saliency_map, fixation_map):
    
    """
    Normalized Scanpath Saliency (NSS) metric.

    Args:
        saliency_map (np.ndarray): Predicted saliency map.
        fixation_map (np.ndarray): Binary ground truth fixation map.

    Returns:
        float: NSS score.
    """
    
    MAP = (saliency_map - saliency_map.mean()) / (saliency_map.std() + 1e-12)
    mask = fixation_map.astype(np.bool_)
    if np.sum(mask) == 0:
        return np.nan
    score = MAP[mask].mean()
    return score

def find_max_value_location(heatmap):
    
    """
    Find normalized location of maximum value in the heatmap.

    Args:
        heatmap (np.ndarray): Input heatmap.

    Returns:
        np.ndarray: Normalized (x, y) location of max value.
    """
    
    max_value = np.max(heatmap)
    h, w = heatmap.shape
    y, x = np.where(heatmap == max_value)
    max_loc = (x[0] / w, y[0] / h)
    return np.array(max_loc)

class Evaluator_affordance(nn.Module):
    
    """
    Evaluator for interaction hotspots heatmaps
    """
    
    def __init__(self, mode, shape=(32, 32)):
        super(Evaluator_affordance, self).__init__()
        self.shape = shape
        self.mode = mode

    def score(self, pred, gt):
        
        """
        Compute SIM, AUC-Judd, NSS scores between predicted and ground truth maps.

        Args:
            pred (np.ndarray): Predicted heatmap.
            gt (np.ndarray): Ground truth heatmap.

        Returns:
            tuple: SIM, AUC-Judd, NSS scores.
        """
        
        pred = pred / (pred.max() + 1e-12)

        gt_real = np.array(gt)
        if gt_real.sum() == 0:
            gt_real = np.ones(gt_real.shape) / np.product(gt_real.shape)

        score_sim = SIM(pred, gt_real)

        gt_binary = np.array(gt)
        gt_binary = (gt_binary / (gt_binary.max() + 1e-12)) if gt_binary.max() > 0 else gt_binary
        gt_binary = np.where(gt_binary > 0.5, 1, 0)
        score_auc = AUC_Judd(pred, gt_binary)
        score_nss = NSS(pred, gt_binary)

        return score_sim, score_auc, score_nss

    def forward(self, pred_l, gt_l, pred_r, gt_r, flag):
        
        """
        Evaluate batch of predicted affordance maps.

        Args:
            pred_l, pred_r (torch.Tensor): Predicted affordance maps for left/right hands.
            gt_l, gt_r (torch.Tensor): Ground truth affordance maps for left/right hands.
            flag (torch.Tensor): Indicator of which hand(s) to evaluate.

        Returns:
            tuple: Aggregated SIM, AUC, NSS scores, and valid count.
        """
        
        sim_all = auc_all = nss_all = count = 0

        pred_l = pred_l.detach().cpu().numpy()
        gt_l = gt_l.detach().cpu().numpy()
        pred_r = pred_r.detach().cpu().numpy()
        gt_r = gt_r.detach().cpu().numpy()

        bs = pred_l.shape[0]

        for i in range(bs):
            p_l, g_l = pred_l[i], gt_l[i]
            p_r, g_r = pred_r[i], gt_r[i]
            ff = flag[i]

            if self.mode == 'map2point':
                coord_l = find_max_value_location(p_l).reshape(-1, 2)
                coord_r = find_max_value_location(p_r).reshape(-1, 2)
                coord_l_gt = find_max_value_location(g_l).reshape(-1, 2)
                coord_r_gt = find_max_value_location(g_r).reshape(-1, 2)
                
                p_l = compute_heatmap(coord_l, self.shape)
                p_r = compute_heatmap(coord_r, self.shape)
                g_l = compute_heatmap(coord_l_gt, self.shape)
                g_r = compute_heatmap(coord_r_gt, self.shape)

            sim_l, auc_l, nss_l = self.score(p_l, g_l)
            sim_r, auc_r, nss_r = self.score(p_r, g_r)

            if ff == 0:
                sim_all += sim_l
                auc_all += auc_l
                nss_all += nss_l
                count += 1
            elif ff == 1:
                sim_all += sim_r
                auc_all += auc_r
                nss_all += nss_r
                count += 1
            elif ff == 2:
                sim_all += (sim_l + sim_r)
                auc_all += (auc_l + auc_r)
                nss_all += (nss_l + nss_r)
                count += 2

        return sim_all, auc_all, nss_all, count
