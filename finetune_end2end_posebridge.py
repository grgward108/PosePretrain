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
LEARNING_RATE = 3e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLIP_SECONDS = 2
CLIP_FPS = 30
MARKERS_TYPE = 'f15_p22'
MODE = 'local_markers_3dv'
SMPLX_MODEL_PATH = 'body_utils/body_models'
NUM_JOINTS = 22
NUM_MARKERS = 143
VALIDATE_EVERY = 5


grab_dir = '../../../data/edwarde/dataset/preprocessed_grab'
train_datasets = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
test_datasets = ['s9', 's10']

#save the infilled joints, lifted markers, original joints, original markers, slerp input in an npz file

def save_reconstruction_npz(slerp_img, temp_filled_joints, temp_original_joints, lift_predicted_markers,  lift_original_markers, traj_gt, rot_0_pivot, transf_matrix_smplx, joint_start, save_dir, epoch, exp_name):
    save_path = os.path.join(save_dir, exp_name)
    os.makedirs(save_path, exist_ok=True) 
    np.savez_compressed(os.path.join(save_path, f"e2e_finetune_{exp_name}_reconstruction_epoch_{epoch+1}.npz"),
             slerp_img=slerp_img.cpu().numpy(),
             temp_filled_joints=temp_filled_joints.cpu().numpy(),
             temp_original_joints=temp_original_joints.cpu().numpy(),
             lift_predicted_markers=lift_predicted_markers.cpu().numpy(),
             lift_original_markers=lift_original_markers.cpu().numpy(),
             traj_gt=traj_gt.cpu().numpy(),
             rot_0_pivot=rot_0_pivot.cpu().numpy(),
             transf_matrix_smplx=transf_matrix_smplx.cpu().numpy(),
             joint_start=joint_start.cpu().numpy()
             )
    print(f"Reconstruction saved at {os.path.join(save_path, f'{exp_name}_reconstruction_epoch_{epoch+1}.npz')}")
    

def train_combined(model, optimizer, dataloader, epoch, logger, DEVICE):
    model.train()
    epoch_loss = 0.0
    num_batches = len(dataloader)

    # Define leg and hand indices
    leg_joint_indices = [1, 2, 4, 5, 7, 8, 10, 11]
    right_hand_indices = torch.cat([torch.arange(64, 79), torch.arange(121, 143)]).to(DEVICE)

    for clip_img_joints, clip_img_markers, slerp_img, traj, *_ in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
        slerp_img = slerp_img.to(DEVICE, dtype=torch.float32)
        temp_original_joints = clip_img_joints.to(DEVICE, dtype=torch.float32)
        lift_original_markers = clip_img_markers.to(DEVICE, dtype=torch.float32)

        # Forward Pass
        temp_filled_joints, lift_predicted_markers = model(slerp_img)

        # Align dimensions
        temp_original_joints = temp_original_joints.permute(0, 3, 2, 1)  # [B, C, J, T]
        
        ############################ Temporal Transformer Loss ############################
        weights = torch.ones_like(temp_filled_joints).to(DEVICE)
        weights[:, :, leg_joint_indices, :] *= 3.0
        weighted_rec_loss = ((temp_filled_joints - temp_original_joints) ** 2 * weights).sum() / weights.sum()

        # Velocity Loss
        original_velocity = temp_original_joints[:, :, :, 1:] - temp_original_joints[:, :, :, :-1]
        reconstructed_velocity = temp_filled_joints[:, :, :, 1:] - temp_filled_joints[:, :, :, :-1]
        weighted_velocity_loss = ((original_velocity - reconstructed_velocity) ** 2 * weights[:, :, :, 1:]).sum() / weights[:, :, :, 1:].sum()

        # Acceleration Loss
        original_acceleration = original_velocity[:, :, :, 1:] - original_velocity[:, :, :, :-1]
        reconstructed_acceleration = reconstructed_velocity[:, :, :, 1:] - reconstructed_velocity[:, :, :, :-1]
        weighted_acceleration_loss = ((original_acceleration - reconstructed_acceleration) ** 2 * weights[:, :, :, 2:]).sum() / weights[:, :, :, 2:].sum()

        # Temporal Loss
        temporal_loss = (
            0.6 * weighted_rec_loss +
            0.3 * weighted_velocity_loss +
            0.1 * weighted_acceleration_loss
        )

        ############################ Lift-Up Transformer Loss ############################
        lift_original_markers = lift_original_markers.permute(0, 3, 2, 1)  # [B, C, M, T]
        weights = torch.ones_like(lift_predicted_markers).to(DEVICE)
        weights[:, :, right_hand_indices, :] *= 2.0
        lift_loss = ((lift_predicted_markers - lift_original_markers) ** 2 * weights).sum() / weights.sum()

        # Total Loss
        total_loss = 0.6 * temporal_loss + 0.4 * lift_loss

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
    temporal_loss_total = 0.0
    lift_loss_total = 0.0
    first_batch_saved = False  # Flag to ensure we only save the first batch

    # Define leg and hand indices
    leg_joint_indices = [1, 2, 4, 5, 7, 8, 10, 11]
    right_hand_indices = torch.cat([torch.arange(64, 79), torch.arange(121, 143)]).to(DEVICE)

    with torch.no_grad():
        for clip_img_joints, clip_img_markers, slerp_img, traj_gt, smplx_beta, gender, rot_0_pivot, transf_matrix_smplx, smplx_params_gt, marker_start, marker_end, joint_start, joint_end in tqdm(dataloader, desc="Validating"):
            slerp_img = slerp_img.to(DEVICE, dtype=torch.float32)
            temp_original_joints = clip_img_joints.to(DEVICE, dtype=torch.float32)
            lift_original_markers = clip_img_markers.to(DEVICE, dtype=torch.float32)

            # Forward Pass
            temp_filled_joints, lift_predicted_markers = model(slerp_img)

            # Align dimensions (same as training)
            temp_original_joints = temp_original_joints.permute(0, 3, 2, 1)  # [B, C, J, T]
            lift_original_markers = lift_original_markers.permute(0, 3, 2, 1)  # [B, C, M, T]

            ############################ Temporal Transformer Loss ############################
            weights = torch.ones_like(temp_filled_joints).to(DEVICE)
            weights[:, :, leg_joint_indices, :] *= 3.0
            weighted_rec_loss = ((temp_filled_joints - temp_original_joints) ** 2 * weights).sum() / weights.sum()

            # Velocity Loss
            original_velocity = temp_original_joints[:, :, :, 1:] - temp_original_joints[:, :, :, :-1]
            reconstructed_velocity = temp_filled_joints[:, :, :, 1:] - temp_filled_joints[:, :, :, :-1]
            weighted_velocity_loss = ((original_velocity - reconstructed_velocity) ** 2 * weights[:, :, :, 1:]).sum() / weights[:, :, :, 1:].sum()

            # Acceleration Loss
            original_acceleration = original_velocity[:, :, :, 1:] - original_velocity[:, :, :, :-1]
            reconstructed_acceleration = reconstructed_velocity[:, :, :, 1:] - reconstructed_velocity[:, :, :, :-1]
            weighted_acceleration_loss = ((original_acceleration - reconstructed_acceleration) ** 2 * weights[:, :, :, 2:]).sum() / weights[:, :, :, 2:].sum()

            # Temporal Loss
            temporal_loss = (
                0.6 * weighted_rec_loss +
                0.3 * weighted_velocity_loss +
                0.1 * weighted_acceleration_loss
            )
            temporal_loss_total += temporal_loss.item()

            ############################ LiftUp Transformer Loss ############################
            weights = torch.ones_like(lift_predicted_markers).to(DEVICE)
            weights[:, :, right_hand_indices, :] *= 2.0
            lift_loss = ((lift_predicted_markers - lift_original_markers) ** 2 * weights).sum() / weights.sum()
            lift_loss_total += lift_loss.item()

            # Total Loss
            total_loss = 0.6 * temporal_loss + 0.4 * lift_loss
            val_loss += total_loss.item()

            # Save reconstruction for the first batch only
            if not first_batch_saved and save_dir is not None:
                save_reconstruction_npz(
                    slerp_img=slerp_img[0],  # Save only the first sequence in the batch
                    temp_filled_joints=temp_filled_joints[0],
                    temp_original_joints=temp_original_joints[0],
                    lift_predicted_markers=lift_predicted_markers[0],
                    lift_original_markers=lift_original_markers[0],
                    traj_gt=traj_gt[0],
                    rot_0_pivot=rot_0_pivot[0],
                    transf_matrix_smplx=transf_matrix_smplx[0],
                    joint_start=joint_start[0],
                    save_dir=save_dir,
                    epoch=epoch,
                    exp_name=exp_name
                )
                first_batch_saved = True

    # Compute average validation loss
    avg_val_loss = val_loss / len(dataloader)
    avg_temporal_loss = temporal_loss_total / len(dataloader)
    avg_lift_loss = lift_loss_total / len(dataloader)

    print(f"Validation - Total Loss: {avg_val_loss:.4f}, Temporal Loss: {avg_temporal_loss:.4f}, Lift Loss: {avg_lift_loss:.4f}")
    return avg_val_loss



def main(exp_name):
    # Logging and Checkpoint Directories
    SAVE_DIR = os.path.join('posebridge_log', exp_name, 'ckpt')
    os.makedirs(SAVE_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(SAVE_DIR, f"{exp_name}.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    # Initialize WandB and Log Configuration
    wandb.init(
        entity='edward-effendy-tokyo-tech696',
        project='PoseBridge',
        name=exp_name,
        config={
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "clip_seconds": CLIP_SECONDS,
            "clip_fps": CLIP_FPS,
            "markers_type": MARKERS_TYPE,
            "mode": MODE,
            "num_joints": NUM_JOINTS,
            "num_markers": NUM_MARKERS,
            "device": DEVICE.type
        },
        mode='disabled'
    )

    logger.info("Training Configuration:")
    logger.info(wandb.config)

    # Dataset paths
    grab_dir = '../../../data/edwarde/dataset/preprocessed_grab'
    train_datasets = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
    test_datasets = ['s9', 's10']

    # Initialize Dataset and DataLoaders
    train_dataset = PreprocessedMotionLoader(grab_dir, train_datasets)
    val_dataset = PreprocessedMotionLoader(grab_dir, test_datasets)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Initialize Models
    temporal_transformer = TemporalTransformer(
        dim_in=3,
        dim_out=3,
        dim_feat=128,
        depth=5,
        num_heads=8,
        num_joints=25,
        maxlen=CLIP_SECONDS * CLIP_FPS + 1
    ).to(DEVICE)

    liftup_transformer = LiftUpTransformer(
        input_dim=3,
        embed_dim=64,
        num_joints=NUM_JOINTS,
        num_markers=NUM_MARKERS,
        num_layers=6,
        num_heads=4,
    ).to(DEVICE)

    # Load Pre-Trained Checkpoints
    temporal_checkpoint_path = 'finetune_temporal_log/test7_addjerkloss_changedataloader/epoch_45.pth'
    liftup_checkpoint_path ='finetune_liftup_log/test3/epoch_100.pth' 

    if os.path.exists(temporal_checkpoint_path):
        logger.info(f"Loading TemporalTransformer from checkpoint: {temporal_checkpoint_path}")
        temporal_checkpoint = torch.load(temporal_checkpoint_path, map_location=DEVICE)
        temporal_transformer.load_state_dict(temporal_checkpoint['model_state_dict'])
        logger.info("TemporalTransformer checkpoint loaded successfully.")
    else:
        logger.warning(f"TemporalTransformer checkpoint not found at {temporal_checkpoint_path}. Training from scratch.")

    if os.path.exists(liftup_checkpoint_path):
        logger.info(f"Loading LiftUpTransformer from checkpoint: {liftup_checkpoint_path}")
        liftup_checkpoint = torch.load(liftup_checkpoint_path, map_location=DEVICE)
        liftup_transformer.load_state_dict(liftup_checkpoint['model_state_dict'])
        logger.info("LiftUpTransformer checkpoint loaded successfully.")
    else:
        logger.warning(f"LiftUpTransformer checkpoint not found at {liftup_checkpoint_path}. Training from scratch.")

    # Combine Models
    model = EndToEndModel(temporal_transformer, liftup_transformer).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # Best validation loss tracking
    best_val_loss = float('inf')

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        logger.info(f"Starting Epoch {epoch + 1}/{NUM_EPOCHS}")

        # Training Phase
        train_loss = train_combined(model, optimizer, train_loader, epoch, logger, DEVICE)
        logger.info(f"Epoch {epoch + 1} Training Loss: {train_loss:.8f}")

        # Log training loss to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss
        })

        # Validation Phase (every 5 epochs)
        if (epoch + 1) % VALIDATE_EVERY == 0 or epoch + 1 == NUM_EPOCHS:
            val_loss = validate_combined(model, val_loader, DEVICE, save_dir=SAVE_DIR, epoch=epoch, exp_name=exp_name)
            logger.info(f"Epoch {epoch + 1} Validation Loss: {val_loss:.8f}")

            # Log validation loss to WandB
            wandb.log({
                "epoch": epoch + 1,
                "val_loss": val_loss
            })

            # Save Best Model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_model_epoch_{epoch + 1}.pth"))
                logger.info(f"Best model saved at epoch {epoch + 1} with Validation Loss: {val_loss:.8f}")

        # Step the scheduler
        scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoseBridge Training Script")
    parser.add_argument("--exp_name", required=True, help="Experiment name")
    args = parser.parse_args()
    main(args.exp_name)
