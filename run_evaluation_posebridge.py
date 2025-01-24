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

from PoseBridge.models.models import EndToEndModel
from TemporalTransformer.models.models import TemporalTransformer
from LiftUpTransformer.models.models import LiftUpTransformer

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


grab_dir = '../../../data/edwarde/dataset/include_global_traj'
train_datasets = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
test_datasets = ['s9', 's10']

#save the infilled joints, lifted markers, original joints, original markers, slerp input in an npz file
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

def train_combined(model, optimizer, dataloader, epoch, logger, DEVICE):
    model.train()
    epoch_loss = 0.0
    velocity_loss_total = 0.0
    acceleration_loss_total = 0.0  # Track acceleration loss for the epoch
    leg_loss_total = 0.0  # Track leg-specific reconstruction loss
    pelvis_loss_total = 0.0
    foot_skating_loss_total = 0.0
    num_batches = len(dataloader)

    # Define leg and hand indices
    leg_joint_indices = [1, 2, 4, 5, 7, 8, 10, 11]
    right_hand_indices = torch.cat([torch.arange(64, 79), torch.arange(121, 143)]).to(DEVICE)

    for clip_img_joints, clip_img_markers, slerp_img, traj, joint_start_global, joint_end_global, *_ in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
        slerp_img = slerp_img.to(DEVICE, dtype=torch.float32)
        temp_original_joints = clip_img_joints.to(DEVICE, dtype=torch.float32)
        lift_original_markers = clip_img_markers.to(DEVICE, dtype=torch.float32)

        # Align dimensions
        temp_original_joints = temp_original_joints.permute(0, 3, 2, 1)  # [B, C, J, T]
        lift_original_markers = lift_original_markers.permute(0, 3, 2, 1)  # [B, C, M, T]
        
        traj = traj[:, :2, :].to(DEVICE)  # Take only x and y, shape [batch_size, 2, frames]
        joint_start_global = joint_start_global.to(DEVICE)
        joint_end_global = joint_end_global.to(DEVICE)

        pelvis_start = joint_start_global[:, 0, :2]  # Extract pelvis (x, y) from start
        pelvis_end = joint_end_global[:, 0, :2]      # Extract pelvis (x, y) from end

        frames = 61  # Number of frames (matching original_clip and slerp_img)
        interp_weights = torch.linspace(0, 1, frames).view(1, -1, 1).to(pelvis_start.device)  # Interpolation weights

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
        foot_velocity = foot_positions[:, 1:, :] - foot_positions[:, :-1, :]
        foot_skating_loss = (foot_velocity ** 2).sum() / foot_velocity.numel()
        
        pelvis_loss_total += pelvis_loss_weighted.item()
        foot_skating_loss_total += foot_skating_loss.item()
        velocity_loss_total += weighted_velocity_loss.item()
        acceleration_loss_total += weighted_acceleration_loss.item()

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

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Track Epoch Losses
        epoch_loss += total_loss.item()
        
    avg_epoch_loss = epoch_loss / num_batches
    logger.info(
        f"Epoch {epoch + 1}: Temporal Loss: {temporal_loss:.4f}, "
        f"Lift-Up Loss: {lift_loss:.4f}, Total Loss: {avg_epoch_loss:.4f}"
    )

    return avg_epoch_loss  # Return average loss for the epoch

def validate_combined(model, dataloader, DEVICE, save_dir=None, epoch=None, exp_name="default"):
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
        for i, (clip_img_joints, clip_img_markers, slerp_img, traj, joint_start_global, joint_end_global, *_) in enumerate(tqdm(dataloader, desc="Validating")):
            slerp_img = slerp_img.to(DEVICE, dtype=torch.float32)
            temp_original_joints = clip_img_joints.to(DEVICE, dtype=torch.float32)
            lift_original_markers = clip_img_markers.to(DEVICE, dtype=torch.float32)

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
            interp_weights = torch.linspace(0, 1, frames).view(1, -1, 1).to(pelvis_start.device)  # Interpolation weights

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
    grab_dir = '../../../data/edwarde/dataset/include_global_traj'
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

    temporal_checkpoint_path = '../../../data/edwarde/dataset/finetune_temporal_log/abb_no_acc_no_footskat/epoch_100.pth'
    liftup_checkpoint_path = 'finetune_liftup_log/test3/epoch_55.pth'

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
    validate_combined(model, val_loader, DEVICE, save_dir=SAVE_DIR, exp_name=exp_name)
    logger.info("Evaluation Complete.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoseBridge Inference and Evaluation Script")
    parser.add_argument("--exp_name", required=True, help="Experiment name")
    args = parser.parse_args()
    main(args.exp_name)
