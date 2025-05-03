import sys
import torch
from tqdm import tqdm
from .losses import Keypoint3DLoss, Pose_Loss, Area_Loss, Hot_Loss, Traj_loss, Mani_loss
from .utils import recursive_to
from multi_train_utils.distributed_utils import reduce_value, is_main_process

def train_one_epoch(model, optimizer, data_loader, device, epoch, args):
    
    """
    Train the multi-task model for one epoch.

    This function performs training over multiple tasks including:
    - 3D hand pose regression (with KLD + reconstruction + joint loss)
    - Binary contact area classification
    - Interaction hotspot localization
    - Pre-contact trajectory forecasting
    - Post-contact manipulation trajectory forecasting

    Args:
        model (nn.Module): The wrapped DDP model.
        optimizer (Optimizer): Optimizer instance.
        data_loader (DataLoader): Training set loader.
        device (torch.device): CUDA or CPU device.
        epoch (int): Current epoch index (for logging).
        args (Namespace): Runtime arguments.

    Returns:
        float: The average training loss for this epoch.
    """
    
    model.train()
    accu_loss = torch.zeros(1).to(device)
    loss_map = torch.zeros(1).to(device)  # For hotspot loss tracking
    loss_point = torch.zeros(1).to(device)  # For 3D keypoint loss tracking

    if is_main_process():
        data_loader = tqdm(data_loader)  # Only show progress bar on main process

    # Initialize loss functions
    point_loss_fn = Keypoint3DLoss().to(device)
    pose_loss_fn = Pose_Loss().to(device)
    area_loss_fn = Area_Loss().to(device)
    hot_loss_fn = Hot_Loss().to(device)
    traj_loss_fn = Traj_loss().to(device)
    mani_loss_fn = Mani_loss().to(device)

    # Initialize loss accumulators
    loss_po = torch.zeros(1).to(device)
    loss_ar = torch.zeros(1).to(device)
    loss_mani = torch.zeros(1).to(device)
    loss_traj = torch.zeros(1).to(device)
    
    # Initialize valid sample counters for each task
    count_traj = count_area = count_hot = count_pose = count_mani = 0
    
    for step,data in enumerate(data_loader):

       # Unpack batch data
        uid, flag, img, raw_img, text, verb, noun, para, pred_cam, key_points_3d, gt_pose, aff_gt_l, aff_gt_r, gt_hpos_last_l, gt_hpos_last_r, gt_area_l, gt_area_r, \
        ori_shape, f_area, f_traj, l_traj_gt, r_traj_gt, hmap_l_init, hmap_r_init, l_traj_map_gt, \
        r_traj_map_gt, l_mani_map_gt, r_mani_map_gt, f_pose = data
        B = img.shape[0]  # Batch size
        
        # Move inputs to device
        img = img.to(device)
        raw_img = raw_img.to(device)
        pred_cam = pred_cam.to(device)
        key_points_3d = key_points_3d.to(device)
        flag = flag.to(device)
        para = recursive_to(para, device)
        aff_gt_r = aff_gt_r.to(device)
        aff_gt_l = aff_gt_l.to(device)
        gt_pose = gt_pose.to(device)
        gt_area_l = gt_area_l.to(device)
        gt_area_r = gt_area_r.to(device)
        l_traj_gt = l_traj_gt.to(device)
        r_traj_gt = r_traj_gt.to(device)
        
        # Forward pass with all ground truth signals
        outputs = model(
            img, raw_img, verb, 
            aff_gt_l, aff_gt_r, gt_pose, gt_area_l, gt_area_r,
            hmap_l_init, hmap_r_init,
            l_traj_map_gt, r_traj_map_gt,
            l_mani_map_gt, r_mani_map_gt
        )
        
        # Unpack outputs (VAE reconstructions + KLDs + predictions)
        (pred_hotspots_l, pred_hotspots_r, recon_hotspots_l, recon_hotspots_r, KLD_hotspots_l, KLD_hotspots_r,
         pred_pose_l, pred_pose_r, recon_pose_l, recon_pose_r, KLD_pose_l, KLD_pose_r,
         pred_area_l, pred_area_r, recon_area_l, recon_area_r, KLD_area_l, KLD_area_r,
         pred_traj_l, recon_traj_l, KLD_traj_l, pred_traj_r, recon_traj_r, KLD_traj_r,
         pred_mani_l, recon_mani_l, KLD_mani_l, pred_mani_r, recon_mani_r, KLD_mani_r) = outputs
        
        # Compute individual task losses
        recon_pose, kld_pose, flag_pose = pose_loss_fn(recon_pose_l, recon_pose_r, KLD_pose_l, KLD_pose_r, flag, f_pose)
        recon_area, kld_area, flag_area = area_loss_fn(recon_area_l, recon_area_r, KLD_area_l, KLD_area_r, flag, f_area)
        recon_traj, kld_traj, flag_traj = traj_loss_fn(recon_traj_l, recon_traj_r, KLD_traj_l, KLD_traj_r, f_traj, flag)
        recon_hot, kld_hot, flag_hot = hot_loss_fn(recon_hotspots_l, recon_hotspots_r, KLD_hotspots_l, KLD_hotspots_r, flag)
        recon_mani, kld_mani, flag_mani = mani_loss_fn(recon_mani_l, recon_mani_r, KLD_mani_l, KLD_mani_r, f_traj, flag)

        
        # Update valid sample counters
        count_traj += flag_traj
        count_area += flag_area
        count_pose += flag_pose
        count_hot += flag_hot
        count_mani += flag_mani
        
        # Prevent division by zero in loss calculation
        if count_traj == 0:
            count_traj = 1
        if count_mani == 0:
            count_mani = 1
        if count_area == 0:
            count_area = 1
        if count_pose == 0:
            count_pose = 1  
        if count_hot == 0:
            count_hot = 1
        
        
        # Compute 3D keypoint loss (joint loss)
        joint_loss_3d, _ = point_loss_fn(pred_pose_l, pred_pose_r, key_points_3d, flag, f_pose)

        # Final task-specific losses
        pose_loss = recon_pose + 0.005 * kld_pose + joint_loss_3d
        area_loss = recon_area + 0.005 * kld_area
        traj_loss = recon_traj + 0.005 * kld_traj
        mani_loss = recon_mani + 0.005 * kld_mani
        hot_loss = recon_hot + 0.001 * kld_hot
        
        # Total loss (scaled by valid flags and batch size)
        if flag_pose == 0:
            flag_pose = 1
        if flag_area == 0:
            flag_area = 1
        if flag_traj == 0:
            flag_traj = 1
        if flag_mani == 0:
            flag_mani = 1
        if flag_hot == 0:
            flag_hot = 1
        
        # Compute total loss (scale each task based on number of valid samples) 
        loss = pose_loss/flag_pose *B + area_loss/flag_area  * B + hot_loss/flag_hot * B  + traj_loss/flag_traj * B + mani_loss/flag_mani * B
        
        # loss = reduce_value(loss, average=True)
        loss.backward()
        loss_to_log = reduce_value(loss.detach().clone(), average=True)
        
        # Accumulate losses for logging
        accu_loss += loss_to_log.item()
        loss_map += recon_hot.item()
        loss_point += joint_loss_3d.item()
        loss_traj += recon_traj.item()
        loss_po += recon_pose.item()
        loss_ar += recon_area.item()
        loss_mani += recon_mani.item()
        
        # Update progress bar (only on main process)
        if is_main_process():
            data_loader.desc = (
                f"[train epoch {epoch + 1}][{step}/{len(data_loader)}] "
                f"Loss: {accu_loss.item() / (step + 1):.4f} "
                f"Loss_hot: {loss_map.item() / count_hot:.4f} "
                f"Loss_point: {loss_point.item() / count_pose:.4f} "
                f"Loss_pose: {loss_po.item() / count_pose:.4f} "
                f"Loss_area: {loss_ar.item() / count_area:.4f} "
                f"Loss_traj: {loss_traj.item() / count_traj:.4f} "
                f"Loss_mani: {loss_mani.item() / count_mani:.4f}"
            )
            
        # Check for NaNs or infinite losses    
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
            
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
        
    # Ensure all CUDA operations are finished
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)
        
    return accu_loss.item() / (step + 1)
