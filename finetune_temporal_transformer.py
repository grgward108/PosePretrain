import os
import torch
import numpy as np
from PoseBridge.data.dataloader import GRAB_DataLoader  # Replace 'your_module' with the actual module where MotionLoader is defined
from torch.utils.data import DataLoader
import argparse
import logging
import wandb
from TemporalTransformer.models.models import TemporalTransformer
from tqdm import tqdm
import torch.optim as optim

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-5
MASK_RATIO = 0.15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLIP_SECONDS = 2
CLIP_FPS = 30
MARKERS_TYPE = 'f15_p5'  # Not really used for joint extraction now, but keep consistent
MODE = 'local_joints_3dv'
SMPLX_MODEL_PATH = 'body_utils/body_models'
STRIDE = 30
NUM_JOINTS = 25
TEMPORAL_CHECKPOINT_PATH = 'temporal_pretrained/epoch_15.pth'

grab_dir = '../../../data/edwarde/dataset/grab/GraspMotion'
train_datasets = ['s1']
test_datasets = ['s9']
smplx_model_path = 'body_utils/body_models'
markers_type = 'f15_p22'  # Example markers type
mode = 'local_joints_3dv' 
VALIDATE_EVERY = 1

def save_reconstruction_npz(markers, reconstructed_markers, original_markers, mask, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    masked = markers.cpu().numpy()
    reconstructed = reconstructed_markers.cpu().numpy()
    ground_truth = original_markers.cpu().numpy()

    # Handle the case where mask is None
    if mask is not None:
        mask_np = mask.cpu().numpy()
    else:
        mask_np = None  # No mask to save

    npz_path = os.path.join(save_dir, f"epoch_{epoch}_reconstruction.npz")
    np.savez_compressed(npz_path, masked=masked, reconstructed=reconstructed, ground_truth=ground_truth, mask=mask_np)
    print(f"Saved reconstruction data to {npz_path}")

def generate_static_mask(length=61):
    mask = torch.ones(length, dtype=torch.float32)
    mask[0] = 0 # First frame unmasked
    mask[-1] = 0  # Last frame unmasked
    return mask

def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def validate(model, val_loader, device, save_reconstruction=False, save_dir=None, epoch=None):
    model.eval()
    val_loss = 0.0
    velocity_loss_total = 0.0
    acceleration_loss_total = 0.0
    leg_loss_total = 0.0
    first_batch_saved = False

    # Define leg indices
    leg_indices = [1, 2, 4, 5, 7, 8, 10, 11]

    with torch.no_grad():
        for i, (clip_img, slerp_img, *_ ) in enumerate(tqdm(val_loader, desc="Validating")):
            original_clip = clip_img.to(device)
            slerp_img = slerp_img.to(device)

            # Permute to match expected input format
            original_clip = original_clip.permute(0, 3, 2, 1)

            # Forward pass
            outputs = model(slerp_img)

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

            # Accumulate losses
            val_loss += weighted_rec_loss.item()
            velocity_loss_total += weighted_velocity_loss.item()
            acceleration_loss_total += weighted_acceleration_loss.item()

            # Save reconstruction for the first batch if needed
            if save_reconstruction and not first_batch_saved and save_dir and epoch and i == 0:
                save_reconstruction_npz(
                    markers=slerp_img,
                    reconstructed_markers=outputs,
                    original_markers=original_clip,
                    mask=None,  # Mask is not used
                    save_dir=save_dir,
                    epoch=epoch
                )
                first_batch_saved = True

    # Compute average losses
    avg_rec_loss = val_loss / len(val_loader)
    avg_velocity_loss = velocity_loss_total / len(val_loader)
    avg_acceleration_loss = acceleration_loss_total / len(val_loader)
    avg_leg_loss = leg_loss_total / len(val_loader)
    total_loss = (
        0.5 * avg_rec_loss +
        0.3 * avg_velocity_loss +
        0.2 * avg_acceleration_loss
    )

    # Log metrics to WandB
    wandb.log({
        "Validation Loss (Reconstruction)": avg_rec_loss,
        "Validation Loss (Velocity)": avg_velocity_loss,
        "Validation Loss (Acceleration)": avg_acceleration_loss,
        "Validation Loss (Leg Reconstruction)": avg_leg_loss,
        "Validation Loss (Total)": total_loss,
    })

    return total_loss

def train(model, optimizer, train_loader, val_loader, logger, checkpoint_dir):
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        velocity_loss_total = 0.0
        acceleration_loss_total = 0.0  # Track acceleration loss for the epoch
        leg_loss_total = 0.0  # Track leg-specific reconstruction loss
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for clip_img, slerp_img, mask, *_ in progress_bar:
            original_clip = clip_img.to(DEVICE)
            original_clip = original_clip.permute(0, 3, 2, 1)
            slerp_img = slerp_img.to(DEVICE)

            # Forward pass
            outputs = model(slerp_img)

            # Define leg indices
            leg_indices = [1, 2, 4, 5, 7, 8, 10, 11]

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

            # Combine losses
            total_loss = (
                0.5 * weighted_rec_loss +
                0.3 * weighted_velocity_loss +
                0.2 * weighted_acceleration_loss
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
                "Total Loss": total_loss.item()
            })

            # Log training metrics to WandB
            wandb.log({
                "Training Loss (Reconstruction)": weighted_rec_loss.item(),
                "Training Loss (Velocity)": weighted_velocity_loss.item(),
                "Training Loss (Acceleration)": weighted_acceleration_loss.item(),
                "Leg Reconstruction Loss": leg_loss.item(),
                "Training Loss (Batch Total)": total_loss.item(),
            })

        # Log average training loss for the epoch
        avg_epoch_rec_loss = epoch_loss / len(train_loader)
        avg_epoch_velocity_loss = velocity_loss_total / len(train_loader)
        avg_epoch_acceleration_loss = acceleration_loss_total / len(train_loader)
        avg_epoch_leg_loss = leg_loss_total / len(train_loader)
        avg_epoch_total_loss = (
            0.5 * avg_epoch_rec_loss +
            0.3 * avg_epoch_velocity_loss +
            0.2 * avg_epoch_acceleration_loss
        )

        logger.info(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Reconstruction Loss: {avg_epoch_rec_loss:.4f}, "
            f"Velocity Loss: {avg_epoch_velocity_loss:.4f}, "
            f"Acceleration Loss: {avg_epoch_acceleration_loss:.4f}, "
            f"Leg Loss: {avg_epoch_leg_loss:.4f}, "
            f"Total Loss: {avg_epoch_total_loss:.4f}"
        )

        wandb.log({
            "Epoch Training Reconstruction Loss": avg_epoch_rec_loss,
            "Epoch Training Velocity Loss": avg_epoch_velocity_loss,
            "Epoch Training Acceleration Loss": avg_epoch_acceleration_loss,
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
                epoch=epoch + 1
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

    wandb.init(entity='edward-effendy-tokyo-tech696', project='TemporalTransformer', name=exp_name, mode='dryrun')
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

    train_dataset = GRAB_DataLoader(clip_seconds=2, clip_fps=30, mode=mode, markers_type=markers_type)
    train_dataset.read_data(train_datasets, grab_dir)

    """143 markers / 55 joints if with_hand else 72 markers / 25 joints"""
    train_dataset.create_body_repr(with_hand=False, smplx_model_path=smplx_model_path)

    val_dataset = GRAB_DataLoader(clip_seconds=2, clip_fps=30, mode=mode, markers_type=markers_type)
    val_dataset.read_data(test_datasets, grab_dir)

    """143 markers / 55 joints if with_hand else 72 markers / 25 joints"""
    val_dataset.create_body_repr(with_hand=False, smplx_model_path=smplx_model_path)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

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
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Checkpoint loaded successfully.")
    else:
        logger.warning(f"Checkpoint not found at {TEMPORAL_CHECKPOINT_PATH}. Training from scratch.")

    num_params = count_learnable_parameters(model)
    logger.info(f"Number of learnable parameters in TemporalTransformer: {num_params:,}")
    wandb.config.update({"num_learnable_parameters": num_params})

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Start training
    train(model, optimizer, train_loader, val_loader, logger, checkpoint_dir)

