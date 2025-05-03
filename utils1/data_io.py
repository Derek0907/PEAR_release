import csv
import os
import numpy as np
import pickle
import json
import open3d as o3d

def hotspots_read(hotspots_path, flag):
    
    """
    Read interaction hotspots from a JSON file.

    Args:
        hotspots_path (str): Path to the JSON file containing hotspots.
        flag (int): Indicates which hand(s) are valid (0: left, 1: right, 2: both).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Normalized left and right hotspot points.
    """
    
    if os.path.exists(hotspots_path):
        with open(hotspots_path, 'r') as f:
            data = json.load(f)

        shapes = data['shapes']
        height = data['imageHeight']
        width = data['imageWidth']
        left_point_list, right_point_list = [], []
        
        for idx in range(len(shapes)):
            lr = shapes[idx]['label']
            points = shapes[idx]['points'][0]
            points[0] = points[0] / width
            points[1] = points[1] / height
            if lr in 'left' and len(left_point_list) == 0:
                left_point_list.append(points)  
            elif lr in 'right' and len(right_point_list) == 0:
                right_point_list.append(points)          
                
        if flag == 0:
            left_point_list = np.array(left_point_list)
            right_point_list = np.array([[0.5, 0.5]]) # Placeholder
            
        elif flag == 1: 
            left_point_list = np.array([[0.5, 0.5]]) # Placeholder
            right_point_list = np.array(right_point_list)
            
        else:
            left_point_list = np.array(left_point_list)
            right_point_list = np.array(right_point_list)
            
        return left_point_list, right_point_list
                
                
def pose_read(pose_path, pose_flag, pose_path_invalid):
    
    """
    Read 3D hand pose data from a pickle file.

    Args:
        pose_path (str): Path to the pose pickle file.
        pose_flag (int): Indicates whether pose is valid (-1: invalid/default).

    Returns:
        Tuple: (MANO parameters, GT pose [2,154], camera parameters [2,3], keypoints_3d).
    """
    
    if pose_flag == -1:
        # Default pose data fallback
        pose_path = pose_path_invalid # any pkl file, and it won't be applied to train/eval process (pose_flag = -1)

    with open(pose_path, 'rb') as f:
        pkl_data = pickle.load(f)

    pred_cam = pkl_data['pred_cam']
    para = pkl_data['pred_mano_params']
    points_3d = pkl_data['pred_keypoints_3d']

    gt_hp = para['hand_pose'].reshape(-1, 135)
    gt_betas = para['betas'].reshape(-1, 10)
    gt_global_orient = para['global_orient'].reshape(-1, 9)
    gt_pose = np.concatenate([gt_global_orient, gt_hp, gt_betas], axis=1)

    return para, gt_pose, pred_cam, points_3d


def contact_area_read(ply_path, flag, uid):
    """
    Read binary contact area from colored .ply files.

    Args:
        ply_path (str): Root path containing 'left' and 'right' .ply folders.
        flag (int): Contact hand indicator (0: left only, 1: right only, 2: both).
        uid (str): Unique sample ID.

    Returns:
        Tuple[np.ndarray, np.ndarray, int]:
            - left_list: binary array of shape (778,) for left hand contact.
            - right_list: binary array of shape (778,) for right hand contact.
            - f_area: flag indicating whether any file exists (1) or not (0).
    """
    left_file = os.path.join(ply_path, 'left', f'{uid}.ply')
    right_file = os.path.join(ply_path, 'right', f'{uid}.ply')

    left_list = [0] * 778
    right_list = [0] * 778
    f_area = 1

    def extract_contact(ply_file):
        """
        Extract contact indices from black-colored points.
        """
        pcd = o3d.io.read_point_cloud(ply_file)
        colors = np.asarray(pcd.colors)
        black_indices = set(np.where(np.all(colors == [0, 0, 0], axis=1))[0])
        # Assume 778 MANO vertices; must match contact region index space.
        contact = [1 if i in black_indices else 0 for i in range(778)] # The contact vertices are black in GT plys
        return contact

    if flag in [0, 2]:
        if os.path.exists(left_file):
            left_list = extract_contact(left_file)
        else:
            f_area = 0

    if flag in [1, 2]:
        if os.path.exists(right_file):
            right_list = extract_contact(right_file)
        else:
            f_area = 0

    return np.array(left_list), np.array(right_list), f_area


def read_trajectory(trajectory_path, flag, lr):
    
    """
    Read full hand trajectory from pickle.

    Args:
        trajectory_path (str): Path to trajectory pickle file.
        flag (int): Validity flag for trajectory.
        lr (int): Hand flag (0: left, 1: right, 2: both).

    Returns:
        Tuple: (left hand trajectory [7,2], right hand trajectory [7,2])
    """
    
    if flag == 1 and os.path.exists(trajectory_path):
        with open(trajectory_path, 'rb') as f:
            data = pickle.load(f)
        hand_trajs = data['hand_trajs']
        l_traj = hand_trajs.get('LEFT', {'traj': np.zeros([7, 2])})['traj'] if lr in [0, 2] else np.zeros([7, 2])
        r_traj = hand_trajs.get('RIGHT', {'traj': np.zeros([7, 2])})['traj'] if lr in [1, 2] else np.zeros([7, 2])
    else:
        l_traj = r_traj = np.zeros([7, 2])
    return l_traj, r_traj


def read_trajectory_pred(trajectory_path, flag, lr):
    
    """
    Read predicted trajectory excluding first (hand position in the image) and fourth (using GT hotspots instead) points.

    Args:
        trajectory_path (str): Path to trajectory pickle file.
        flag (int): Validity flag for trajectory.
        lr (int): Hand flag (0: left, 1: right, 2: both).

    Returns:
        Tuple: (left hand predicted trajectory [5,2], right hand predicted trajectory [5,2])
    """
    
    new_l_traj, new_r_traj = [], []
    if flag == 1 and os.path.exists(trajectory_path):
        with open(trajectory_path, 'rb') as f:
            data = pickle.load(f)
        hand_trajs = data['hand_trajs']
        if lr in [0, 2]:
            l_traj = hand_trajs['LEFT']['traj']
            new_l_traj = [l_traj[i] for i in range(7) if i not in [0, 3]]
        else:
            new_l_traj = np.zeros([5, 2])
        if lr in [1, 2]:
            r_traj = hand_trajs['RIGHT']['traj']
            new_r_traj = [r_traj[i] for i in range(7) if i not in [0, 3]]
        else:
            new_r_traj = np.zeros([5, 2])
    else:
        new_l_traj = np.zeros([5, 2])
        new_r_traj = np.zeros([5, 2])
    return np.array(new_l_traj), np.array(new_r_traj)


def read_data(mode, yaml_data):
    """
    Load dataset metadata and labels from CSV and various annotation files.

    Args:
        mode (str): 'train' or 'val' mode.
        yaml_data (dict): YAML config with paths.

    Returns:
        Tuple: Lists of data paths, labels, and annotations for dataset loading.
        
    CSV header: 
    clip_id | video_id | start_frame | end_frame | text_description | contact_frame | uid | hotspots_flag | verb | trajectory_flag | noun | pose_flag
    """
    csv_path = yaml_data['ANNOTATIONS']['TRAIN'] if mode == "train" else yaml_data['ANNOTATIONS']['VAL']
        
    # Lists to collect data
    uid_list, img_path_list, mask_list = [], [], []
    text_list, verb_list, noun_list, flag_list = [], [], [], []
    para_list, pred_cam_list, keypoints_3d_list, gt_pose_list = [], [], [], []
    affordance_list_l, affordance_list_r = [], []
    area_left_list, area_right_list, f_area_list = [], [], []
    l_traj_list, r_traj_list, l_traj_pred_list, r_traj_pred_list = [], [], [], []
    f_traj_list, f_pose_list = [], []
    
    # Roots
    img_root = yaml_data['DATA']['IMAGE']
    hotspots_root = yaml_data['DATA']['HOTSPOTS']
    pose_root = yaml_data['DATA']['POSE']
    contact_area_root = yaml_data['DATA']['CONTACT_AREA']
    
    # The train trajectory labels are generated by following steps: [raw point position -> heatmap -> extract point from heatmaps] to avoid boundary effects.
    trajectory_root_train = yaml_data['DATA']['TRAJECTORY_TRAIN'] 
    trajectory_root_val = yaml_data['DATA']['TRAJECTORY_VAL']
    mask_root = yaml_data['DATA']['MASK']
    pose_path_invalid = yaml_data['DATA']['POSE_INVALID'] # Any label can be applied (just for format alighment)

    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            text, uid = row[4], row[-6]
            flag, verb, traj_flag, noun, pose_flag = int(row[-5]), row[-4], int(row[-3]), row[-2], int(row[-1])
            f_traj_list.append(traj_flag)
            uid_list.append(uid)
            text_list.append(text)
            flag_list.append(flag)
            verb_list.append(verb)
            noun_list.append(noun)
            
            # Paths
            img_path_list.append(os.path.join(img_root, f'{uid}.jpg'))
            mask_list.append(os.path.join(mask_root, f'{uid}.jpg'))
            hotspots_path = os.path.join(hotspots_root, f'{uid}_text.json')
            pose_path = os.path.join(pose_root, f'{uid}.pkl')
            if mode == "train":
                traj_path = os.path.join(trajectory_root_train, f'label_{uid}.pkl')
            else:
                traj_path = os.path.join(trajectory_root_val, f'label_{uid}.pkl')
            
            # Read data
            left_point, right_point = hotspots_read(hotspots_path, flag)
            affordance_list_l.append(left_point)
            affordance_list_r.append(right_point)

            para, gt_pose, pred_cam, points_3d = pose_read(pose_path, pose_flag, pose_path_invalid)
            para_list.append(para)
            pred_cam_list.append(pred_cam)
            keypoints_3d_list.append(points_3d)
            gt_pose_list.append(gt_pose)
            f_pose_list.append(pose_flag)

            left_area, right_area, f_area = contact_area_read(contact_area_root, flag, uid)
            area_left_list.append(left_area)
            area_right_list.append(right_area)
            f_area_list.append(f_area)

            l_traj, r_traj = read_trajectory(traj_path, traj_flag, flag)
            l_traj_pred, r_traj_pred = read_trajectory_pred(traj_path, traj_flag, flag)
            l_traj_list.append(l_traj)
            r_traj_list.append(r_traj)
            l_traj_pred_list.append(l_traj_pred)
            r_traj_pred_list.append(r_traj_pred)

    return (uid_list, img_path_list, mask_list, text_list, verb_list, noun_list,
            flag_list, para_list, pred_cam_list, keypoints_3d_list, gt_pose_list,
            affordance_list_l, affordance_list_r, area_left_list, area_right_list,
            f_area_list, f_traj_list, l_traj_list, r_traj_list,
            l_traj_pred_list, r_traj_pred_list, f_pose_list)     

