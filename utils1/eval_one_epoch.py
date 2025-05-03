import torch
from tqdm import tqdm
from .utils import recursive_to
from .pose_metric import Evaluator_pose
from .affordance_metric import Evaluator_affordance
from .area_metric import Evaluator_area
from .traj_metric import Hmaptraj_Metric, Hmapmani_Metric
from multi_train_utils.distributed_utils import is_main_process

@torch.no_grad()
def evaluate(model, data_loader, device, epoch, args, save_root):
    
    """
    Run single-GPU evaluation on the validation set for one epoch.
    This function is expected to be executed only on rank 0 (main process) 
    during multi-GPU distributed training.

    Args:
        model (torch.nn.Module): The model wrapped with DistributedDataParallel.
        data_loader (torch.utils.data.DataLoader): Validation dataloader.
        device (torch.device): CUDA or CPU device for inference.
        epoch (int): Current epoch number (used for logging).
        args (argparse.Namespace): Runtime arguments.
        save_root (str): Path to save visualizations and logs.

    Returns:
        tuple: (PA-MPJPE, SIM, AUC, NSS, Precision, Recall, F1, ADE, FDE, ADE_mani, FDE_mani)
    """
    
    model.eval()
    
    # Initialize metric accumulators
    r = torch.zeros(1).to(device) #PA-MPJPE (pose)
    sim_mean = torch.zeros(1).to(device)
    auc_mean = torch.zeros(1).to(device)
    nss_mean = torch.zeros(1).to(device)
    ade_mean = torch.zeros(1).to(device)
    fde_mean = torch.zeros(1).to(device)
    ade_mani = torch.zeros(1).to(device)
    fde_mani = torch.zeros(1).to(device)
    accuracy_mean = torch.zeros(1).to(device)
    precision_mean = torch.zeros(1).to(device)
    recall_mean = torch.zeros(1).to(device)
    f1_mean = torch.zeros(1).to(device)
    
    # Progress bar for main process
    if is_main_process():
        data_loader = tqdm(data_loader)
    
    # Initialize visualizers and evaluators
    pose_eval = Evaluator_pose()
    area_eval = Evaluator_area()
    traj_eval = Hmaptraj_Metric()
    mani_eval = Hmapmani_Metric()
    aff_eval = Evaluator_affordance(mode='map2point')

    
    # Sample counters for valid data
    hot_num = area_num = pose_num = traj_num = 0
    

    for step, data in enumerate(data_loader):
        # Unpack and move data to device
        (uid, flag, img, raw_img, text, verb, noun, para, pred_cam, key_points_3d, gt_pose,
         aff_gt_l, aff_gt_r, gt_hpos_last_l, gt_hpos_last_r, gt_area_l, gt_area_r, ori_shape, f_area, f_traj,
         l_traj_gt, r_traj_gt, hmap_l_init, hmap_r_init,
         l_traj_map_gt, r_traj_map_gt,
         l_mani_map_gt, r_mani_map_gt, f_pose) = data

        # Move inputs to device
        img = img.to(device)
        pred_cam = pred_cam.to(device)
        raw_img = raw_img.to(device)
        key_points_3d = key_points_3d.to(device)
        flag = flag.to(device)
        para = recursive_to(para, device)
        aff_gt_r = aff_gt_r.to(device)
        aff_gt_l = aff_gt_l.to(device)
        gt_hpos_last_l = gt_hpos_last_l.to(device)
        gt_hpos_last_r = gt_hpos_last_r.to(device)
        gt_area_l = gt_area_l.to(device)
        gt_area_r = gt_area_r.to(device)
        hmap_l_init = hmap_l_init.to(device)
        hmap_r_init = hmap_r_init.to(device)
        l_traj_gt = l_traj_gt.to(device)
        r_traj_gt = r_traj_gt.to(device)
        gt_pose = gt_pose.to(device)
        # Inference through the model
        (pred_hotspots_l, pred_hotspots_r, pred_pose_l, pred_pose_r, pred_area_l, pred_area_r,
         pred_traj_l, pred_traj_r, pred_mani_l, pred_mani_r,
         pred_traj_all_l, pred_traj_all_r) = model.module.inference(
            img, raw_img, verb, hmap_l_init, hmap_r_init)

            
        # Compute evaluation metrics    
        mpjpe, re, n_pose = pose_eval(pred_pose_l, pred_pose_r, key_points_3d, flag, f_pose)
        sim_all, auc_all, nss_all, n_hot = aff_eval(pred_hotspots_l,aff_gt_l, pred_hotspots_r, aff_gt_r, flag)
        acc, pre, rec, f1, n_area = area_eval(pred_area_l, pred_area_r, gt_area_l, gt_area_r, flag, f_area)
        ade, fde, n_traj = traj_eval(pred_traj_l, pred_traj_r, gt_hpos_last_l, gt_hpos_last_r, f_traj, flag, l_traj_gt, r_traj_gt)
        ade_2, fde_2, n_mani = mani_eval(pred_mani_l, pred_mani_r, l_traj_gt, r_traj_gt, f_traj, flag)
        
        # Accumulate results
        if n_pose != 0:
            r += re.item()
        sim_mean += sim_all.item()
        auc_mean += auc_all.item()
        nss_mean += nss_all.item()
        accuracy_mean += acc
        precision_mean += pre
        recall_mean += rec
        f1_mean += f1
        ade_mean += ade
        fde_mean += fde
        ade_mani += ade_2
        fde_mani += fde_2

        # Update counters
        pose_num += n_pose
        hot_num += n_hot
        area_num += n_area
        traj_num += n_traj
        
        # Prevent division by zero
        pose_num = max(pose_num, 1)
        hot_num = max(hot_num, 1)
        area_num = max(area_num, 1)
        traj_num = max(traj_num, 1)

        # Display metrics on progress bar (main process only)
        if is_main_process():
            data_loader.desc = (
                f"[validation epoch {epoch+1}] re: {r.item() / pose_num:.3f} sim: {sim_mean.item() / hot_num:.3f} "
                f"auc: {auc_mean.item() / hot_num:.3f} nss: {nss_mean.item() / hot_num:.3f} "
                f"accuracy: {accuracy_mean.item() / area_num:.3f} precision: {precision_mean.item() / area_num:.3f} "
                f"recall: {recall_mean.item() / area_num:.3f} f1: {f1_mean.item() / area_num:.3f} "
                f"ade: {ade_mean.item() / traj_num:.3f} fde: {fde_mean.item() / traj_num:.3f} "
                f"ade_mani: {ade_mani.item() / traj_num:.3f} fde_mani: {fde_mani.item() / traj_num:.3f}"
            )

    # Ensure all CUDA ops are done
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    # Return average metrics across valid samples
    return (
        r.item() / pose_num,
        sim_mean.item() / hot_num,
        auc_mean.item() / hot_num,
        nss_mean.item() / hot_num,
        precision_mean.item() / area_num,
        recall_mean.item() / area_num,
        f1_mean.item() / area_num,
        ade_mean.item() / traj_num,
        fde_mean.item() / traj_num,
        ade_mani.item() / traj_num,
        fde_mani.item() / traj_num
    )