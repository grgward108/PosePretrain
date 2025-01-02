import os
import torch
import numpy as np
from PoseBridge.data.dataloader import GRAB_DataLoader  
from torch.utils.data import DataLoader
import argparse
import logging
import wandb
from TemporalTransformer.models.models import TemporalTransformer
from tqdm import tqdm
import torch.optim as optim
from PoseBridge.data.preprocessed_dataloader import PreprocessedMotionLoader


BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 3e-5
MASK_RATIO = 0.15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLIP_SECONDS = 2
CLIP_FPS = 30
MARKERS_TYPE = 'f15_p5'  # Not really used for joint extraction now, but keep consistent
MODE = 'local_joints_3dv'
SMPLX_MODEL_PATH = 'body_utils/body_models'
STRIDE = 30
NUM_JOINTS = 26 # change to 26 because we add one more pelvis global joint
TEMPORAL_CHECKPOINT_PATH = 'pretrained_models/epoch_15_checkpoint_pretrained_globalpelvis.pth'

grab_dir = '../../../data/edwarde/dataset/include_global_traj'
train_datasets = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
test_datasets = ['s9', 's10']
smplx_model_path = 'body_utils/body_models'
markers_type = 'f15_p22'  # Example markers type
mode = 'local_joints_3dv' 
VALIDATE_EVERY = 5

def save_reconstruction_npz(markers, reconstructed_markers, original_markers, mask, save_dir, epoch, exp_name):
    os.makedirs(save_dir, exist_ok=True)
    masked = markers.cpu().numpy()
    reconstructed = reconstructed_markers.cpu().numpy()
    ground_truth = original_markers.cpu().numpy()

    # Handle the case where mask is None
    if mask is not None:
        mask_np = mask.cpu().numpy()
    else:
        mask_np = None  # No mask to save

    npz_path = os.path.join(save_dir, f"finetune_temporal_{exp_name}_epoch_{epoch}_reconstruction.npz")
    np.savez_compressed(npz_path, masked=masked, reconstructed=reconstructed, ground_truth=ground_truth, mask=mask_np)
    print(f"Saved reconstruction data to {npz_path}")

def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def validate(model, val_loader, device, save_reconstruction=False, save_dir=None, epoch=None, exp_name=None):
    model.eval()
    val_loss = 0.0
    velocity_loss_total = 0.0
    acceleration_loss_total = 0.0
    leg_loss_total = 0.0
    pelvis_loss_total = 0.0
    foot_skating_loss_total = 0.0
    first_batch_saved = False

    # Define leg indices
    leg_indices = [1, 2, 4, 5, 7, 8, 10, 11]

    with torch.no_grad():
        for i, (clip_img, _, slerp_img, *_ ) in enumerate(tqdm(val_loader, desc="Validating")):
            original_clip = clip_img.to(device)
            slerp_img = slerp_img.to(device)

            # Permute to match expected input format
            original_clip = original_clip.permute(0, 3, 2, 1)

            # Forward pass
            outputs = model(slerp_img)

            # Higher weight for pelvis reconstruction
            pelvis_output = outputs[:, :, 0, :]
            pelvis_original = original_clip[:, :, 0, :]
            pelvis_loss = ((pelvis_output - pelvis_original) ** 2).mean()
            pelvis_loss_weighted = 5.0 * pelvis_loss

            # Create weight tensor
            weights = torch.ones_like(outputs)  # Default weight of 1 for all joints
            weights[:, :, leg_indices, :] *= 2.0  # Double the weight for leg joints

            # Weighted reconstruction loss
            weighted_rec_loss = ((outputs - original_clip) ** 2 * weights).sum() / weights.sum()

            # Leg-specific reconstruction loss (for logging)
            leg_loss = ((outputs[:, :, leg_indices, :] - original_clip[:, :, leg_indices, :]) ** 2).mean()
            leg_loss_total += leg_loss.item()

            # Compute velocity loss
            original_velocity = original_clip[:, :, :, 1:] - original_clip[:, :, :, :-1]
            reconstructed_velocity = outputs[:, :, :, 1:] - outputs[:, :, :, :-1]
            velocity_diff = (original_velocity - reconstructed_velocity) ** 2 * weights[:, :, :, 1:]
            weighted_velocity_loss = velocity_diff.sum() / weights[:, :, :, 1:].sum()

            # Compute acceleration loss
            original_acceleration = original_velocity[:, :, :, 1:] - original_velocity[:, :, :, :-1]
            reconstructed_acceleration = reconstructed_velocity[:, :, :, 1:] - reconstructed_velocity[:, :, :, :-1]
            acceleration_diff = (original_acceleration - reconstructed_acceleration) ** 2 * weights[:, :, :, 2:]
            weighted_acceleration_loss = acceleration_diff.sum() / weights[:, :, :, 2:].sum()

            # Step 2: Global context restoration for foot skating loss
            global_translation = outputs[:, :, 0:1, :]

            # local_joints: shape (B, T, J-1, F)
            local_joints = outputs[:, :, 1:, :]
            restored_joints = local_joints + global_translation  # Restore global context

            # Step 3: Compute foot skating loss
            # Compute foot skating loss
            feet_indices = [7, 8, 10, 11]
            foot_positions = restored_joints[:, :, feet_indices, :]  # Use restored joints only
            foot_velocity = foot_positions[:, 1:, :] - foot_positions[:, :-1, :]
            foot_skating_loss = (foot_velocity ** 2).sum() / foot_velocity.numel()

            # Accumulate losses
            val_loss += weighted_rec_loss.item()
            velocity_loss_total += weighted_velocity_loss.item()
            acceleration_loss_total += weighted_acceleration_loss.item()
            pelvis_loss_total += pelvis_loss_weighted.item()
            foot_skating_loss_total += foot_skating_loss.item()

            # Save reconstruction for the first batch if needed
            if save_reconstruction and not first_batch_saved and save_dir and epoch and i == 0:
                save_reconstruction_npz(
                    markers=slerp_img,
                    reconstructed_markers=outputs,
                    original_markers=original_clip,
                    mask=None,  # Mask is not used
                    save_dir=save_dir,
                    epoch=epoch,
                    exp_name=exp_name
                )
                first_batch_saved = True

    # Compute average losses
    avg_rec_loss = val_loss / len(val_loader)
    avg_velocity_loss = velocity_loss_total / len(val_loader)
    avg_acceleration_loss = acceleration_loss_total / len(val_loader)
    avg_pelvis_loss = pelvis_loss_total / len(val_loader)  # Compute average pelvis loss
    avg_foot_skating_loss = foot_skating_loss_total / len(val_loader)  # Compute average foot skating loss
    avg_leg_loss = leg_loss_total / len(val_loader)

    # Updated total loss calculation with all components
    avg_total_loss = (
        0.55 * avg_rec_loss +
        0.15 * avg_velocity_loss +
        0.1 * avg_acceleration_loss +
        0.1 * avg_pelvis_loss +
        0.1 * avg_foot_skating_loss
    )

    # Log metrics to WandB
    wandb.log({
        "Validation Loss (Reconstruction)": avg_rec_loss,
        "Validation Loss (Velocity)": avg_velocity_loss,
        "Validation Loss (Acceleration)": avg_acceleration_loss,
        "Validation Loss (Pelvis)": avg_pelvis_loss,
        "Validation Loss (Foot Skating)": avg_foot_skating_loss,
        "Validation Loss (Leg)": avg_leg_loss,
        "Validation Loss (Total)": avg_total_loss,
    })

    return avg_total_loss

def train(model, optimizer, train_loader, val_loader, logger, checkpoint_dir, exp_name):
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        velocity_loss_total = 0.0
        acceleration_loss_total = 0.0  # Track acceleration loss for the epoch
        leg_loss_total = 0.0  # Track leg-specific reconstruction loss
        pelvis_loss_total = 0.0
        foot_skating_loss_total = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for clip_img_joints, _, slerp_img, traj, joint_start_global, joint_end_global, *_ in progress_bar:
            original_clip = clip_img_joints.to(DEVICE)
            original_clip = original_clip.permute(0, 3, 2, 1)  # [batch_size, 61, 25, 3]
            slerp_img = slerp_img.to(DEVICE)

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
            original_clip = torch.cat([traj, original_clip], dim=2)  # Concatenate without unsqueezing

            # Prepend interp_pelvis to slerp_img
            slerp_img = torch.cat([interp_pelvis, slerp_img], dim=2)  # Concatenate without unsqueezing

            # Forward pass
            outputs = model(slerp_img)

            # Higher weight for pelvis reconsturction
            pelvis_output = outputs[:, :, 0, :]
            pelvis_original = original_clip[:, :, 0, :]
            pelvis_loss = ((pelvis_output - pelvis_original) ** 2).mean()
            pelvis_loss_weighted = 5.0 * pelvis_loss
            
            # Define leg indices
            leg_indices = [1, 2, 4, 5, 7, 8, 10, 11, 18, 19, 20, 21]

            # Create weight tensor (double the weight for leg joints)
            weights = torch.ones_like(outputs)  
            weights[:, :, leg_indices, :] *= 2.0  

            # Weighted reconstruction loss
            weighted_rec_loss = ((outputs - original_clip) ** 2 * weights).sum() / weights.sum()

            # Compute velocity (1st derivative)
            original_velocity = original_clip[:, :, :, 1:] - original_clip[:, :, :, :-1]
            reconstructed_velocity = outputs[:, :, :, 1:] - outputs[:, :, :, :-1]
            velocity_diff = (original_velocity - reconstructed_velocity) ** 2 * weights[:, :, :, 1:]
            weighted_velocity_loss = velocity_diff.sum() / weights[:, :, :, 1:].sum()

            # Compute acceleration (2nd derivative)
            original_acceleration = original_velocity[:, :, :, 1:] - original_velocity[:, :, :, :-1]
            reconstructed_acceleration = reconstructed_velocity[:, :, :, 1:] - reconstructed_velocity[:, :, :, :-1]
            acceleration_diff = (original_acceleration - reconstructed_acceleration) ** 2 * weights[:, :, :, 2:]
            weighted_acceleration_loss = acceleration_diff.sum() / weights[:, :, :, 2:].sum()

            # Leg-specific reconstruction loss (for logging)
            leg_loss = ((outputs[:, :, leg_indices, :] - original_clip[:, :, leg_indices, :]) ** 2).mean()
            leg_loss_total += leg_loss.item()
            
            # Step 2: Global context restoration for foot skating loss
            global_translation = outputs[:, :, 0:1, :]
            print("global_translation shape: ", global_translation.shape)

            # local_joints: shape (B, T, J-1, F)
            local_joints = outputs[:, :, 1:, :]
            restored_joints = local_joints + global_translation  # Restore global context
            print("restored_joints shape: ", restored_joints.shape)

            # Step 3: Compute foot skating loss
            # Compute foot skating loss
            feet_indices = [7, 8, 10, 11]
            foot_positions = restored_joints[:, :, feet_indices, :]  # Use restored joints only
            foot_velocity = foot_positions[:, 1:, :] - foot_positions[:, :-1, :]
            foot_skating_loss = (foot_velocity ** 2).sum() / foot_velocity.numel()
            
            pelvis_loss_total += pelvis_loss_weighted.item()
            foot_skating_loss_total += foot_skating_loss.item()

            # Combine losses
            total_loss = (
                0.55 * weighted_rec_loss +
                0.15 * weighted_velocity_loss +
                0.10 * weighted_acceleration_loss +
                0.10 * foot_skating_loss +
                0.10 * pelvis_loss_weighted
            )


            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update epoch loss
            epoch_loss += weighted_rec_loss.item()
            velocity_loss_total += weighted_velocity_loss.item()
            acceleration_loss_total += weighted_acceleration_loss.item()
            progress_bar.set_postfix({
                "Reconstruction Loss": weighted_rec_loss.item(),
                "Velocity Loss": weighted_velocity_loss.item(),
                "Acceleration Loss": weighted_acceleration_loss.item(),
                "Foot Skating Loss": foot_skating_loss.item(),
                "Pelvis Loss": pelvis_loss_weighted.item(),
                "Total Loss": total_loss.item()
            })

            # Log training metrics to WandB
            wandb.log({
                "Training Loss (Reconstruction)": weighted_rec_loss.item(),
                "Training Loss (Velocity)": weighted_velocity_loss.item(),
                "Training Loss (Acceleration)": weighted_acceleration_loss.item(),
                "Training Loss (Foot Skating)": foot_skating_loss.item(),
                "Training Loss (Pelvis)": pelvis_loss_weighted.item(),
                "Training Loss (Batch Total)": total_loss.item(),
            })

        # Log average training loss for the epoch
        avg_epoch_rec_loss = epoch_loss / len(train_loader)
        avg_epoch_velocity_loss = velocity_loss_total / len(train_loader)
        avg_epoch_acceleration_loss = acceleration_loss_total / len(train_loader)
        avg_epoch_pelvis_loss = pelvis_loss_total / len(train_loader)  # Compute average pelvis loss
        avg_epoch_foot_skating_loss = foot_skating_loss_total / len(train_loader)  # Compute average foot skating loss
        avg_epoch_leg_loss = leg_loss_total / len(train_loader)

        # Updated total loss calculation with all components
        avg_epoch_total_loss = (
            0.55 * avg_epoch_rec_loss +
            0.15 * avg_epoch_velocity_loss +
            0.1 * avg_epoch_acceleration_loss +
            0.1 * avg_epoch_pelvis_loss +
            0.1 * avg_epoch_foot_skating_loss
        )

        # Log to console
        logger.info(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Reconstruction Loss: {avg_epoch_rec_loss:.4f}, "
            f"Velocity Loss: {avg_epoch_velocity_loss:.4f}, "
            f"Acceleration Loss: {avg_epoch_acceleration_loss:.4f}, "
            f"Pelvis Loss: {avg_epoch_pelvis_loss:.4f}, "
            f"Foot Skating Loss: {avg_epoch_foot_skating_loss:.4f}, "
            f"Leg Loss: {avg_epoch_leg_loss:.4f}, "
            f"Total Loss: {avg_epoch_total_loss:.4f}"
        )

        # Log to WandB
        wandb.log({
            "Epoch Training Reconstruction Loss": avg_epoch_rec_loss,
            "Epoch Training Velocity Loss": avg_epoch_velocity_loss,
            "Epoch Training Acceleration Loss": avg_epoch_acceleration_loss,
            "Epoch Training Pelvis Loss": avg_epoch_pelvis_loss,
            "Epoch Training Foot Skating Loss": avg_epoch_foot_skating_loss,
            "Epoch Training Leg Loss": avg_epoch_leg_loss,
            "Epoch Training Total Loss": avg_epoch_total_loss,
            "Epoch": epoch + 1,
        })


        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_total_loss,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved at {checkpoint_path}")

        # Validate every VALIDATE_EVERY epochs
        if (epoch + 1) % VALIDATE_EVERY == 0:
            save_dir = os.path.join(checkpoint_dir, "reconstruction_val")
            val_loss = validate(
                model,
                val_loader=val_loader,
                device=DEVICE,
                save_reconstruction=True,  # Enable saving reconstruction
                save_dir=save_dir,
                epoch=epoch + 1,
                exp_name=exp_name
            )
            logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Validation Loss: {val_loss:.4f}")
            wandb.log({"Validation Loss": val_loss, "Epoch": epoch + 1})


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train TemporalTransformer')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    args = parser.parse_args()
    exp_name = args.exp_name
    checkpoint_dir = os.path.join('finetune_temporal_log', exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_file = os.path.join(checkpoint_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    logger.info("Starting training script with the following parameters:")
    logger.info(f"Experiment Name: {exp_name}")
    logger.info(f"Checkpoint Directory: {checkpoint_dir}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    logger.info(f"Mask Ratio: {MASK_RATIO}")
    logger.info(f"Clip Seconds: {CLIP_SECONDS}")
    logger.info(f"Clip FPS: {CLIP_FPS}")
    logger.info(f"Mode: {MODE}")
    logger.info(f"Markers Type: {MARKERS_TYPE}")
    logger.info(f"Device: {DEVICE}")

    wandb.init(entity='edward-effendy-tokyo-tech696', project='TemporalTransformer', name=exp_name, mode='disabled')
    wandb.config.update({
        "experiment_name": exp_name,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "mask_ratio": MASK_RATIO,
        "clip_seconds": CLIP_SECONDS,
        "clip_fps": CLIP_FPS,
        "mode": MODE,
        "markers_type": MARKERS_TYPE,
        "device": DEVICE.type,
    })

    grab_dir = '../../../data/edwarde/dataset/include_global_traj'
    train_datasets = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
    test_datasets = ['s9', 's10']

    # Initialize Dataset and DataLoaders
    train_dataset = PreprocessedMotionLoader(grab_dir, train_datasets)
    val_dataset = PreprocessedMotionLoader(grab_dir, test_datasets)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = TemporalTransformer(
        dim_in=3,
        dim_out=3,
        dim_feat=128,
        depth=5,
        num_heads=8,
        num_joints=NUM_JOINTS,
        maxlen=CLIP_SECONDS * CLIP_FPS + 1  # This should match the input length of your sequence
    ).to(DEVICE)

    # Load checkpoint
    if os.path.exists(TEMPORAL_CHECKPOINT_PATH):
        logger.info(f"Loading model from checkpoint: {TEMPORAL_CHECKPOINT_PATH}")
        checkpoint = torch.load(TEMPORAL_CHECKPOINT_PATH, map_location=DEVICE)
        
        # Fix keys with "module." prefix
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")  # Remove "module." prefix
            new_state_dict[new_key] = value

        # Load the cleaned state_dict
        model.load_state_dict(new_state_dict)
        logger.info("Checkpoint loaded successfully.")
    else:
        logger.warning(f"Checkpoint not found at {TEMPORAL_CHECKPOINT_PATH}. Training from scratch.")


    num_params = count_learnable_parameters(model)
    logger.info(f"Number of learnable parameters in TemporalTransformer: {num_params:,}")
    wandb.config.update({"num_learnable_parameters": num_params})

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Start training
    train(model, optimizer, train_loader, val_loader, logger, checkpoint_dir, exp_name)

