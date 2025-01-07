# Training Script with Lazy-Loading Dataset
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PoseBridge.data.end2end_dataloader_lazyloading import GRAB_DataLoader as MotionLoader
from PoseBridge.data.preprocessed_dataloader import PreprocessedMotionLoader
from tqdm import tqdm
import wandb
import argparse
import logging
import numpy as np
from human_body_prior.tools.model_loader import load_vposer
from PoseBridge.models.models import EndToEndModel
from TemporalTransformer.models.models import TemporalTransformer
from LiftUpTransformer.models.models import LiftUpTransformer
from MotionFill.models.LocalMotionFill import Motion_CNN_CVAE
from MotionFill.models.TrajFill import Traj_MLP_CVAE
import pickle
import scipy.ndimage.filters as filters
from utils.como.como_utils import *

# Hyperparameters and Settings
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLIP_SECONDS = 2
CLIP_FPS = 30
MARKERS_TYPE = 'f15_p22'
MODE = 'local_markers_3dv'
SMPLX_MODEL_PATH = 'body_utils/body_models'
NUM_JOINTS = 22
NUM_MARKERS = 143
VALIDATE_EVERY = 5
PELVIS_LOSS_WEIGHT = 5.0
LEG_RECONSTRUCTION_WEIGHT = 2.0
RIGHT_HAND_WEIGHTS = 2.0

FINAL_RECONSTRUCTION_LOSS_WEIGHT = 0.6
FINAL_VELOCITY_LOSS_WEIGHT = 0.1
FINAL_ACCELERATION_LOSS_WEIGHT = 0.05
FINAL_PELVIS_LOSS_WEIGHT = 0.10
FINAL_FOOT_SKATING_LOSS_WEIGHT = 0.15

from utils.utils_body import (gen_body_mesh_v1, get_body_mesh, get_body_model,
                              get_global_pose, get_markers_ids,
                              get_object_mesh)

"""
	FullGraspMotion Pipeline:
		- Generate ending pose (either load from saved results or load saved model and implement the optimization)
		- Set initial frames (in markers)
		- Generate trajectories (in position) -> maybe we need some post-optimization here
		- Feed into the Motion-CVAE network
"""


def get_forward_joint(joint_start):
	""" Joint_start: [B, N, 3] in xyz """
	x_axis = joint_start[:, 2, :] - joint_start[:, 1, :]
	x_axis[:, -1] = 0
	x_axis = x_axis / torch.norm(x_axis, dim=-1).unsqueeze(1)
	z_axis = torch.tensor([0, 0, 1]).float().unsqueeze(0).repeat(len(x_axis), 1).to(DEVICE)
	y_axis = torch.cross(z_axis, x_axis)
	y_axis = y_axis / torch.norm(y_axis, dim=-1).unsqueeze(1)
	transf_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=1)
	return y_axis, transf_rotmat

def prepare_traj_input(joint_start, joint_end, traj_stats):
	""" Joints: [B, N, 3] in xyz """
	B, N, _ = joint_start.shape
	T = 62
	joint_sr_input = torch.ones(B, 4, T)  # [B, xyr, T]
	y_axis, transf_rotmat = get_forward_joint(joint_start)
	joint_start_new = joint_start.clone()
	joint_end_new = joint_end.clone()  # to check whether original joints change or not
	joint_start_new = torch.matmul(joint_start - joint_start[:, 0:1], transf_rotmat)
	joint_end_new = torch.matmul(joint_end - joint_start[:, 0:1], transf_rotmat)

	# start_forward, _ = get_forward_joint(joint_start_new)
	start_forward = torch.tensor([0, 1, 0]).unsqueeze(0)
	end_forward, _ = get_forward_joint(joint_end_new)

	joint_sr_input[:, :2, 0] = joint_start_new[:, 0, :2]  # xy
	joint_sr_input[:, :2, -1] = joint_end_new[:, 0, :2]   # xy
	joint_sr_input[:, 2:, 0] = start_forward[:, :2]  # r
	joint_sr_input[:, 2:, -1] = end_forward[:, :2]  # r


	# normalize
	traj_mean = torch.tensor(traj_stats['traj_Xmean']).unsqueeze(0).unsqueeze(2)
	traj_std = torch.tensor(traj_stats['traj_Xstd']).unsqueeze(0).unsqueeze(2)

	joint_sr_input_normed = (joint_sr_input - traj_mean) / traj_std
	for t in range(joint_sr_input_normed.size(-1)):
		joint_sr_input_normed[:, :, t] = joint_sr_input_normed[:, :, 0] + (joint_sr_input_normed[:, :, -1] - joint_sr_input_normed[:, :, 0])*t/(joint_sr_input_normed.size(-1)-1)
		joint_sr_input_normed[:, -2:, t] = joint_sr_input_normed[:, -2:, t] / torch.norm(joint_sr_input_normed[:, -2:, t], dim=1).unsqueeze(1)

	for t in range(joint_sr_input.size(-1)):
		joint_sr_input[:, :, t] = joint_sr_input[:, :, 0] + (joint_sr_input[:, :, -1] - joint_sr_input[:, :, 0])*t/(joint_sr_input.size(-1)-1)
		joint_sr_input[:, -2:, t] = joint_sr_input[:, -2:, t] / torch.norm(joint_sr_input[:, -2:, t], dim=1).unsqueeze(1)

	# linear interpolation

	return joint_sr_input_normed.float().to(DEVICE), joint_sr_input.float().to(DEVICE), transf_rotmat, joint_start_new, joint_end_new

def prepare_clip_img_input(marker_start, marker_end, joint_start, joint_end, joint_start_new, joint_end_new, transf_rotmat, traj_samples_unnormed_best, traj_sr_unnormed, markers_stats, traj_smoothed=True):
	B, n_markers, _ = marker_start.shape
	_, n_joints, _ = joint_start.shape
	markers = torch.rand(B, 61, n_markers, 3)  # [B, T, N ,3]
	joints = torch.rand(B, 61, n_joints, 3)  # [B, T, N ,3]

	marker_start_new = torch.matmul(marker_start - joint_start[:, 0:1], transf_rotmat)
	marker_end_new = torch.matmul(marker_end - joint_start[:, 0:1], transf_rotmat)  

	z_transl_to_floor_start = torch.min(marker_start_new[:, :, -1], dim=-1)[0]# - 0.03
	z_transl_to_floor_end = torch.min(marker_end_new[:, :, -1], dim=-1)[0]# - 0.03

	marker_start_new[:, :, -1] -= z_transl_to_floor_start.unsqueeze(1)
	marker_end_new[:, :, -1] -= z_transl_to_floor_end.unsqueeze(1)
	joint_start_new[:, :, -1] -= z_transl_to_floor_start.unsqueeze(1)
	joint_end_new[:, :, -1] -= z_transl_to_floor_end.unsqueeze(1)

	markers[:, 0] = marker_start_new
	markers[:, -1] = marker_end_new
	joints[:, 0] = joint_start_new
	joints[:, -1] = joint_end_new

	cur_body = torch.cat([joints[:, :, 0:1], markers], dim=2)

	cur_body[:, :, :, [1, 2]] = cur_body[:, :, :, [2, 1]]  # => xyz -> xzy

	reference = cur_body[:, :, 0] * torch.tensor([1, 0, 1])  # => the xy of pelvis joint?
	cur_body = torch.cat([reference.unsqueeze(2), cur_body], dim=2)   # [B, T, 1(reference)+1(pelvis)+N, 3]

	# position to local frame
	cur_body[:, :, :, 0] = cur_body[:, :, :, 0] - cur_body[:, :, 0:1, 0]
	cur_body[:, :, :, -1] = cur_body[:, :, :, -1] - cur_body[:, :, 0:1, -1]

	forward = np.zeros((B, 62, 3))
	forward[:, :, :2] = traj_samples_unnormed_best[:, 2:].transpose(0, 2, 1)
	forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]
	forward[:, :, [1, 2]] = forward[:, :, [2, 1]]

	if traj_smoothed:
		direction_filterwidth = 20
		forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=1, mode='nearest')
		traj_samples_unnormed_best[:, 2] = forward[:, :, 0]
		traj_samples_unnormed_best[:, 3] = forward[:, :, -1]
    
	target = np.array([[0, 0, 1]])#.repeat(len(forward), axis=0)

	rotation = Quaternions.between(forward, target)[:, :, np.newaxis]  # [B, T, 1, 4]

	cur_body = rotation[:, :-1] * cur_body.detach().cpu().numpy()  # [B, T, 1+1+N, xzy]
	cur_body[:, 1:-1] = 0
	cur_body[:, :, :, [1, 2]] = cur_body[:, :, :, [2, 1]]  # xzy => xyz
	cur_body = cur_body[:, :, 1:, :]
	cur_body = cur_body.reshape(cur_body.shape[0], cur_body.shape[1], -1)  # [B, T, N*3]

	velocity = np.zeros((B, 3, 61))
	velocity[:, 0, :] = traj_samples_unnormed_best[:, 0, 1:] - traj_samples_unnormed_best[:, 0, 0:-1]  # [B, 2, 60] on Joint frame
	velocity[:, -1, :] = traj_samples_unnormed_best[:, 1, 1:] - traj_samples_unnormed_best[:, 1, 0:-1]  # [B, 2, 60] on Joint frame


	velocity = rotation[:, 1:] * velocity.transpose(0, 2, 1).reshape(B, 61, 1, 3)
	rvelocity = Pivots.from_quaternions(rotation[:, 1:] * -rotation[:, :-1]).ps   # [B, T-1, 1]
	rot_0_pivot = Pivots.from_quaternions(rotation[:, 0]).ps


	global_x = velocity[:, :, 0, 0]
	global_y = velocity[:, :, 0, 2]
	contact_lbls = np.zeros((B, 61, 4))

	channel_local = np.concatenate([cur_body, contact_lbls], axis=-1)[:, np.newaxis, :, :]  # [B, 1, T-1, d=N*3+4]
	T, d = channel_local.shape[-2], channel_local.shape[-1]
	channel_global_x = np.repeat(global_x, d).reshape(-1, 1, T, d)  # [B, 1, T-1, d]
	channel_global_y = np.repeat(global_y, d).reshape(-1, 1, T, d)  # [B, 1, T-1, d]
	channel_global_r = np.repeat(rvelocity, d).reshape(-1, 1, T, d)  # [B, 1, T-1, d]

	cur_body = np.concatenate([channel_local, channel_global_x, channel_global_y, channel_global_r], axis=1)  # [B, 4, T-1, d]

	cur_body[:, 0] = (cur_body[:, 0] - markers_stats['Xmean_local']) / markers_stats['Xstd_local']
	cur_body[:, 1:3] = (cur_body[:, 1:3] - markers_stats['Xmean_global_xy']) / markers_stats['Xstd_global_xy']
	cur_body[:, 3] = (cur_body[:, 3] - markers_stats['Xmean_global_r']) / markers_stats['Xstd_global_r']


	# mask cur_body
	cur_body = cur_body.transpose(0, 1, 3, 2)  # [B, 4, D, T-1]
	mask_t_1 = [0, 60]
	mask_t_0 = list(set(range(60+1)) - set(mask_t_1))
	cur_body[:, 0, 2:, mask_t_0] = 0.
	cur_body[:, 0, -4:, :] = 0.
	# print('Mask the markers in the following frames: ', mask_t_0)

	# for key in end_body_smplx.keys():
	# 	print('processing:', key, end_body_smplx[key].shape)

	return cur_body, rot_0_pivot, marker_start_new, marker_end_new, traj_samples_unnormed_best

def motion_infilling_inference(model, clip_img_input_new):
	with torch.no_grad():
		z_rand = torch.randn((clip_img_input_new.size(0), 512)).cuda()
		clip_img_rec, _, _ = model(input=clip_img_input_new, is_train=False, z=z_rand)

	contact_lbl_rec = F.sigmoid(clip_img_rec[:, 0, -4:, :].permute(0, 2, 1))  # [B, T, 4]
	contact_lbl_rec[contact_lbl_rec > 0.5] = 1.0
	contact_lbl_rec[contact_lbl_rec <= 0.5] = 0.0

	return clip_img_rec, contact_lbl_rec

def compute_foot_skating_loss(foot_positions):
    """
    Quantify foot skating artifacts during motion.

    Args:
        foot_positions (torch.Tensor): Tensor of foot positions [B, T, F, 3], 
                                       where F is the number of foot joints.

    Returns:
        avg_skating_ratio (float): Average skating ratio across the batch.
        avg_skating_speed (float): Average skating speed during skating frames across the batch.
    """
    # Extract Z-axis (height) and calculate velocity
    heel_positions_z = foot_positions[:, :, :, 2]  # Z-axis (height)
    foot_velocity = foot_positions[:, 1:, :, :] - foot_positions[:, :-1, :, :]  # Velocity
    foot_speed = torch.norm(foot_velocity, dim=-1)  # Speed (Euclidean norm)

    # Conditions for skating
    close_to_ground = (heel_positions_z[:, :-1, :] <= 0.1)  # Height ≤ 5 cm
    high_speed = (foot_speed > 0.075)  # Speed > 75 mm/s

    # Combine conditions
    skating_frames = close_to_ground & high_speed

    # Calculate skating ratio and speed for each sample in the batch
    batch_skating_ratios = []
    batch_avg_skating_speeds = []

    for i in range(foot_positions.size(0)):  # Iterate over the batch
        skating_count = skating_frames[i].sum().item()
        total_frames = foot_positions.size(1) - 1

        # Skating ratio for this sample
        skating_ratio = skating_count / total_frames
        batch_skating_ratios.append(skating_ratio)

        # Average skating speed for this sample
        skating_speeds = foot_speed[i][skating_frames[i]]
        avg_skating_speed = skating_speeds.mean().item() if skating_speeds.numel() > 0 else 0
        batch_avg_skating_speeds.append(avg_skating_speed)

    # Compute averages across the batch
    avg_skating_ratio = sum(batch_skating_ratios) / len(batch_skating_ratios)
    avg_skating_speed = sum(batch_avg_skating_speeds) / len(batch_avg_skating_speeds)

    # Print metrics for debugging
    print(f"Batch size: {foot_positions.size(0)}")
    print(f"Total frames per sequence: {total_frames}")
    print(f"Average skating ratio across batch: {avg_skating_ratio}")
    print(f"Average skating speed across batch: {avg_skating_speed}")

    return avg_skating_ratio, avg_skating_speed

def save_reconstruction_npz(
    slerp_img, temp_filled_joints, temp_original_joints, lift_predicted_markers,
    lift_original_markers, traj_gt, rot_0_pivot, transf_matrix_smplx, joint_start,
    save_dir, epoch, exp_name, batch
):
    """
    Save reconstruction data for the entire batch.

    Args:
        slerp_img: Tensor of shape (B, T, J, C) for the entire batch.
        temp_filled_joints: Tensor of shape (B, T, J, C) for the reconstructed joints.
        temp_original_joints: Tensor of shape (B, T, J, C) for the ground truth joints.
        lift_predicted_markers: Tensor of shape (B, T, M, C) for the predicted markers.
        lift_original_markers: Tensor of shape (B, T, M, C) for the ground truth markers.
        save_dir: Directory to save the file.
        epoch: Current epoch number.
        exp_name: Experiment name.
    """
    save_path = os.path.join(save_dir, exp_name)
    os.makedirs(save_path, exist_ok=True)

    # Construct a single file name for the entire batch
    file_name = f"batch_{batch+1}_evaluation_posebridge_{exp_name}.npz"

    # Save all data in a single file
    np.savez_compressed(
        os.path.join(save_path, file_name),
        slerp_img=slerp_img.cpu().numpy(),
        temp_filled_joints=temp_filled_joints.cpu().numpy(),
        temp_original_joints=temp_original_joints.cpu().numpy(),
        lift_predicted_markers=lift_predicted_markers.cpu().numpy(),
        lift_original_markers=lift_original_markers.cpu().numpy(),
        traj_gt=traj_gt.cpu().numpy() if traj_gt is not None else None,
        rot_0_pivot=rot_0_pivot.cpu().numpy() if rot_0_pivot is not None else None,
        transf_matrix_smplx=transf_matrix_smplx.cpu().numpy() if transf_matrix_smplx is not None else None,
        joint_start=joint_start.cpu().numpy() if joint_start is not None else None,
    )
    print(f"Reconstruction saved at {os.path.join(save_path, file_name)}")

def validate_combined(model, dataloader, DEVICE, traj_stats, markers_stats, save_dir=None, epoch=None, exp_name="default"):
    model.eval()
    val_loss = 0.0
    velocity_loss_total = 0.0
    acceleration_loss_total = 0.0
    leg_loss_total = 0.0
    pelvis_loss_total = 0.0
    foot_skating_loss_total = 0.0
    num_batches = len(dataloader)
    save_first_batch = True

    # Define leg and hand indices
    leg_joint_indices = [1, 2, 4, 5, 7, 8, 10, 11]
    right_hand_indices = torch.cat([torch.arange(64, 79), torch.arange(121, 143)]).to(DEVICE)

    with torch.no_grad():
        for i, (clip_img_joints, clip_img_markers, slerp_img, traj, joint_start_global, joint_end_global, marker_start_global, marker_end_global,  *_) in enumerate(tqdm(dataloader, desc="Validating")):
            slerp_img = slerp_img.to(DEVICE, dtype=torch.float32)
            temp_original_joints = clip_img_joints.to(DEVICE, dtype=torch.float32)
            lift_original_markers = clip_img_markers.to(DEVICE, dtype=torch.float32)
            marker_start = marker_start_global.to(DEVICE)
            marker_end = marker_end_global.to(DEVICE)
            joint_start = joint_start_global.to(DEVICE)
            joint_end = joint_end_global.to(DEVICE)
            bs = slerp_img.size(0)
            
            """ Also conduct validation on SAGA Model"""
            
            mano_fname = './body_utils/smplx_mano_flame_correspondences/MANO_SMPLX_vertex_ids.pkl'
            with open(mano_fname, 'rb') as f:
                idxs_data = pickle.load(f)
                rhand_verts = idxs_data['right_hand']
                lhand_verts = idxs_data['left_hand']
            # markers setup
            markers_ids = get_markers_ids('f15_p22')  # different from grasppose training where we have dense markers on the hand, for motion infilling, we only use 5 markers on each palm.
            markers_ids_143 = get_markers_ids('f15_p22')
            
            
            """ 3. Generate in-between trajectories and local motions """
            ### prepare models
            traj_model = Traj_MLP_CVAE(nz=512, feature_dim=4, T=62, residual='True', load_path=args.traj_ckpt_path).to(DEVICE)
            motion_model = Motion_CNN_CVAE(nz=512, downsample='True', in_channel=4, kernel=3, clip_seconds=2).to(DEVICE)
            ## todo: integrate checkpoint loading into motion model
            motion_model.load_state_dict(torch.load(args.motion_ckpt_path)['model_dict'])
            traj_model.eval()
            motion_model.eval()
            vposer_model_path = './body_utils/body_models/vposer_v1_0'
            vposer_model, _ = load_vposer(vposer_model_path, vp_model='snapshot')
            vposer_model = vposer_model.cuda()

            # prepare statistics
            traj_stats = np.load(args.traj_stats_path)
            markers_stats = np.load(args.markers_stats_dir)

            # generate in-between trajectories
            traj_sr_input, traj_sr_unnormed, transf_rotmat, joint_start_new, joint_end_new = prepare_traj_input(joint_start, joint_end, traj_stats)  # Note: this is the joint forward
            traj_samples = traj_model.sample(traj_sr_input.view(bs, -1))
            traj_mean = torch.tensor(traj_stats['traj_Xmean']).unsqueeze(0).unsqueeze(2).to(DEVICE)
            traj_std = torch.tensor(traj_stats['traj_Xstd']).unsqueeze(0).unsqueeze(2).to(DEVICE)
            traj_samples_unnormed = (traj_samples * traj_std + traj_mean).detach().cpu().numpy()
            
            # generate in-between local motions
            clip_img_input, rot_0_pivot, marker_start_new, marker_end_new, traj_input = prepare_clip_img_input(marker_start, marker_end, joint_start, joint_end, joint_start_new, joint_end_new, transf_rotmat, traj_samples_unnormed, traj_sr_unnormed, markers_stats)
            clip_img_input_new = torch.tensor(clip_img_input).to(DEVICE).float()  # [B, 4, D, T]
            clip_img_rec, contact_lbl_rec = motion_infilling_inference(motion_model, clip_img_input_new)
            
            # Align dimensions
            temp_original_joints = temp_original_joints.permute(0, 3, 2, 1)  # [B, C, J, T]
            lift_original_markers = lift_original_markers.permute(0, 3, 2, 1)  # [B, C, M, T]
            print("lift_original_markers shape: ", lift_original_markers.shape)
            
            
            traj = traj[:, :2, :].to(DEVICE)  # Take only x and y, shape [batch_size, 2, frames]
            joint_start_global = joint_start_global.to(DEVICE)
            joint_end_global = joint_end_global.to(DEVICE)

            pelvis_start = joint_start_global[:, 0, :2]  # Extract pelvis (x, y) from start
            pelvis_end = joint_end_global[:, 0, :2]      # Extract pelvis (x, y) from end

            frames = 61  # Number of frames (matching original_clip and slerp_img)
            interp_weights = torch.linspace(0, 1, frames).view(1, -1, 1).to(pelvis_start.device)


            # Interpolated pelvis trajectory
            interp_pelvis = (1 - interp_weights) * pelvis_start.unsqueeze(1) + interp_weights * pelvis_end.unsqueeze(1)  # Shape: [batch_size, frames, 2]

            # Transform traj to have shape [batch_size, frames, 1, 3]
            traj = traj.permute(0, 2, 1)  # Change shape to [batch_size, frames, 2]
            traj = traj[:, :frames, :]  # Adjust traj to match the number of frames
            traj = traj.unsqueeze(2)  # Add the singleton dimension for [batch_size, frames, 1, 2]
            traj = torch.cat([traj, torch.zeros(traj.shape[0], traj.shape[1], 1, 1, device=DEVICE)], dim=-1)  # Add z dimension
            interp_pelvis = interp_pelvis.unsqueeze(2)  # Add the singleton dimension for [batch_size, frames, 1, 2]
            interp_pelvis = torch.cat([interp_pelvis, torch.zeros(interp_pelvis.shape[0], interp_pelvis.shape[1], 1, 1, device=DEVICE)], dim=-1)
            temp_original_joints = torch.cat([traj, temp_original_joints], dim=2)  # Concatenate without unsqueezing
            lift_original_markers = torch.cat([traj, lift_original_markers], dim=2)  # Concatenate 
            print("lift_original_markers shape after concat with traj: ", lift_original_markers.shape)

            # Prepend interp_pelvis to slerp_img
            slerp_img = torch.cat([interp_pelvis, slerp_img], dim=2)  # Concatenate without unsqueezing
            print("slerp_img shape: ", slerp_img.shape)
            # Forward pass
            temp_filled_joints, lift_predicted_markers = model(slerp_img)

            ############################ Temporal Transformer Loss ############################
            pelvis_output = temp_filled_joints[:, :, 0, :]
            pelvis_original = temp_original_joints[:, :, 0, :]
            pelvis_loss = ((pelvis_output - pelvis_original) ** 2).mean()
            pelvis_loss_weighted = PELVIS_LOSS_WEIGHT * pelvis_loss

            # Define leg indices
            leg_indices = [1, 2, 4, 5, 7, 8, 10, 11, 18, 19, 20, 21]

            # Create weight tensor (double the weight for leg joints)
            weights = torch.ones_like(temp_filled_joints)
            weights[:, :, leg_indices, :] *= LEG_RECONSTRUCTION_WEIGHT

            # Weighted reconstruction loss
            weighted_rec_loss = ((temp_filled_joints - temp_original_joints) ** 2 * weights).sum() / weights.sum()

            # Compute velocity (1st derivative)
            original_velocity = temp_original_joints[:, :, :, 1:] - temp_original_joints[:, :, :, :-1]
            reconstructed_velocity = temp_filled_joints[:, :, :, 1:] - temp_filled_joints[:, :, :, :-1]
            velocity_diff = (original_velocity - reconstructed_velocity) ** 2 * weights[:, :, :, 1:]
            weighted_velocity_loss = velocity_diff.sum() / weights[:, :, :, 1:].sum()

            # Compute acceleration (2nd derivative)
            original_acceleration = original_velocity[:, :, :, 1:] - original_velocity[:, :, :, :-1]
            reconstructed_acceleration = reconstructed_velocity[:, :, :, 1:] - reconstructed_velocity[:, :, :, :-1]
            acceleration_diff = (original_acceleration - reconstructed_acceleration) ** 2 * weights[:, :, :, 2:]
            weighted_acceleration_loss = acceleration_diff.sum() / weights[:, :, :, 2:].sum()

            # Leg-specific reconstruction loss (for logging)
            leg_loss = ((temp_filled_joints[:, :, leg_indices, :] - temp_original_joints[:, :, leg_indices, :]) ** 2).mean()
            leg_loss_total += leg_loss.item()

            # Step 2: Global context restoration for foot skating loss
            global_translation = temp_filled_joints[:, :, 0:1, :]

            # local_joints: shape (B, T, J-1, F)
            local_joints = temp_filled_joints[:, :, 1:, :]
            restored_joints = local_joints + global_translation  # Restore global context

            # Step 3: Compute foot skating loss
            # Compute foot skating loss
            feet_indices = [7, 8, 10, 11]
            foot_positions = restored_joints[:, :, feet_indices, :]  # Use restored joints only
            skating_ratio, _ = compute_foot_skating_loss(foot_positions)
            print(f"Foot Skating Ratio: {skating_ratio:.4f}")
            foot_velocity = foot_positions[:, 1:, :] - foot_positions[:, :-1, :]
            foot_skating_loss = (foot_velocity ** 2).sum() / foot_velocity.numel()

            pelvis_loss_total += pelvis_loss_weighted.item()
            foot_skating_loss_total += foot_skating_loss.item()
            velocity_loss_total += weighted_velocity_loss.item()
            acceleration_loss_total += weighted_acceleration_loss.item()
            
            # feet_lifted_markers = [40, 43]
            # feet_lifted_positions = lift_predicted_markers[:, :, feet_lifted_markers, :]
            # skating_ratio, _ = compute_foot_skating_loss(feet_lifted_positions)
            # print(f"Foot Skating Ratio (Lifted): {skating_ratio:.4f}")

            # Combine losses
            temporal_loss = (
                FINAL_RECONSTRUCTION_LOSS_WEIGHT * weighted_rec_loss +
                FINAL_VELOCITY_LOSS_WEIGHT * weighted_velocity_loss +
                FINAL_ACCELERATION_LOSS_WEIGHT * weighted_acceleration_loss +
                FINAL_FOOT_SKATING_LOSS_WEIGHT * foot_skating_loss +
                FINAL_PELVIS_LOSS_WEIGHT * pelvis_loss_weighted
            )

            ############################ Lift-Up Transformer Loss ############################
            weights = torch.ones_like(lift_predicted_markers).to(DEVICE)
            weights[:, :, right_hand_indices, :] *= RIGHT_HAND_WEIGHTS
            lift_loss = ((lift_predicted_markers - lift_original_markers) ** 2 * weights).sum() / weights.sum()

            # Total Loss
            total_loss = 0.8 * temporal_loss + 0.2 * lift_loss
            val_loss += total_loss.item()

            # Save reconstruction for all sequences in the batch
            if save_dir is not None:
                batch_save_dir = os.path.join(save_dir, f"batch_{i}")
                os.makedirs(batch_save_dir, exist_ok=True)  # Create a directory for each batch
                save_reconstruction_npz(
                    slerp_img=slerp_img,  # Save the entire batch
                    temp_filled_joints=temp_filled_joints,
                    temp_original_joints=temp_original_joints,
                    lift_predicted_markers=lift_predicted_markers,
                    lift_original_markers=lift_original_markers,
                    traj_gt=traj,
                    rot_0_pivot=None,  # Optional, set to None
                    transf_matrix_smplx=None,  # Optional, set to None
                    joint_start=None,  # Optional, set to None
                    save_dir=batch_save_dir,
                    epoch=epoch,
                    exp_name=exp_name,
                    batch=i
                )


    # Compute average validation loss
    avg_val_loss = val_loss / num_batches

    print(f"Validation - Total Loss: {avg_val_loss:.4f}")
    return avg_val_loss

def main(exp_name):
    # Logging Setup
    SAVE_DIR = os.path.join('posebridge_eval_log', exp_name)
    os.makedirs(SAVE_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[logging.FileHandler(os.path.join(SAVE_DIR, f"{exp_name}_inference.log")), logging.StreamHandler()],
    )
    logger = logging.getLogger()

    # Dataset paths
    grab_dir = '../../../data/edwarde/dataset/testingforSAGA'
    test_datasets = ['s9', 's10']
    val_dataset = PreprocessedMotionLoader(grab_dir, test_datasets)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Load Models
    temporal_transformer = TemporalTransformer(
        dim_in=3,
        dim_out=3,
        dim_feat=128,
        depth=5,
        num_heads=8,
        num_joints=26,
        maxlen=CLIP_SECONDS * CLIP_FPS + 1,
    ).to(DEVICE)

    liftup_transformer = LiftUpTransformer(
        input_dim=3,
        embed_dim=64,
        num_joints=NUM_JOINTS,
        num_markers=NUM_MARKERS,
        num_layers=6,
        num_heads=4,
    ).to(DEVICE)

    temporal_checkpoint_path = 'finetune_temporal_log/withpelvis_fromscratch/epoch_100.pth'
    liftup_checkpoint_path = 'finetune_liftup_log/test4_fromscratch/epoch_55.pth'
    traj_stats = np.load(args.traj_stats_path)
    markers_stats = np.load(args.markers_stats_dir)

    if os.path.exists(temporal_checkpoint_path):
        logger.info(f"Loading TemporalTransformer from checkpoint: {temporal_checkpoint_path}")
        temporal_checkpoint = torch.load(temporal_checkpoint_path, map_location=DEVICE)
        temporal_transformer.load_state_dict(temporal_checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError(f"TemporalTransformer checkpoint not found at {temporal_checkpoint_path}.")

    if os.path.exists(liftup_checkpoint_path):
        logger.info(f"Loading LiftUpTransformer from checkpoint: {liftup_checkpoint_path}")
        liftup_checkpoint = torch.load(liftup_checkpoint_path, map_location=DEVICE)
        liftup_transformer.load_state_dict(liftup_checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError(f"LiftUpTransformer checkpoint not found at {liftup_checkpoint_path}.")

    # Combine Models
    model = EndToEndModel(temporal_transformer, liftup_transformer).to(DEVICE)

    # Evaluate Model
    logger.info("Starting Evaluation...")
    validate_combined(model, val_loader, DEVICE, traj_stats, markers_stats, save_dir=SAVE_DIR, exp_name=exp_name)
    logger.info("Evaluation Complete.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoseBridge Inference and Evaluation Script")
    parser.add_argument("--exp_name", required=True, help="Experiment name")
    """Config for GraspMotion"""
    parser.add_argument('--GraspPose_exp_name', default=None, type=str, help='Loaded GraspPose training experiment name')
    parser.add_argument('--object', default=None, type=str, help='object name')
    parser.add_argument('--gender', default=None, type=str, help='gender')
    parser.add_argument('--traj_ckpt_path', default='./pretrained_model/TrajFill_model.pkl', type=str, help='traj_infilling checkpoint path')
    parser.add_argument('--motion_ckpt_path', default='./pretrained_model/LocalMotionFill_model.pkl', type=str, help='traj_infilling checkpoint path')
    parser.add_argument('--traj_stats_path', default='./pretrained_model/prestats_GRAB_traj.npz', type=str, help='traj statistics')
    parser.add_argument('--markers_stats_dir', default='./pretrained_model/prestats_GRAB_contact_given_global_withHand_local_markers_3dv_4chan.npz', type=str, help='markers statistics')

    parser.add_argument('--stage1_weight_loss_rec_markers', type=float, default=1.0)
    parser.add_argument('--stage1_weight_loss_vposer', type=float, default=0.02)
    parser.add_argument('--stage1_weight_loss_shape', type=float, default=0.01)
    parser.add_argument('--stage1_weight_loss_hand', type=float, default=0.01)

    parser.add_argument('--stage2_weight_loss_rec_markers', type=float, default=0.1)
    parser.add_argument('--stage2_weight_loss_vposer', type=float, default=0.02)
    parser.add_argument('--stage2_weight_loss_shape', type=float, default=0.02)
    parser.add_argument('--stage2_weight_loss_hand', type=float, default=0.02)
    parser.add_argument('--stage2_weight_loss_skating', type=float, default=0.05)
    parser.add_argument('--stage2_weight_loss_smooth', type=float, default=2e7)  # 2e7
    parser.add_argument('--stage2_weight_loss_hand_smooth',
                        type=float, default=1)  # 1
    parser.add_argument('--stage2_weight_loss_hand_angle',
                        type=float, default=1)  # 1
    parser.add_argument('--stage2_weight_loss_contact',
                        type=float, default=60)  # 60
    parser.add_argument('--stage2_weight_loss_collision', type=float, default=10)  # 10
    parser.add_argument('--stage2_weight_loss_end_markers_fit',
                        type=float, default=10)  # 0.1
    args = parser.parse_args()
    main(args.exp_name)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    traj_stats = np.load(args.traj_stats_path)
    markers_stats = np.load(args.markers_stats_dir)
