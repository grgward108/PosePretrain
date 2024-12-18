#finetune liftup transformer
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
import logging
import wandb
from LiftUpTransformer.models.models import LiftUpTransformer
from tqdm import tqdm
import torch.optim as optim
from PoseBridge.data.preprocessed_dataloader import PreprocessedMotionLoader

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 3e-4
MASK_RATIO = 0.15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLIP_SECONDS = 2
CLIP_FPS = 30
MARKERS_TYPE = 'f15_p5'  # Not really used for joint extraction now, but keep consistent
MODE = 'local_joints_3dv'
SMPLX_MODEL_PATH = 'body_utils/body_models'
STRIDE = 30
NUM_JOINTS = 22
NUM_MARKERS = 143
LIFTUP_CHECKPOINT_PATH = 'pretrained_models/best_model_epoch_30_liftup.pth'

grab_dir = '../../../data/edwarde/dataset/grab/GraspMotion'
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

    npz_path = os.path.join(save_dir, f"finetune_liftup_{exp_name}_epoch_{epoch}_reconstruction.npz")
    np.savez_compressed(npz_path, masked=masked, reconstructed=reconstructed, ground_truth=ground_truth, mask=mask_np)
    print(f"Saved reconstruction data to {npz_path}")

def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def validate(model, val_loader, device, save_reconstruction=False, save_dir=None, epoch=None, exp_name=None):
    model.eval()
    val_loss = 0.0
    first_batch_saved = False

    # Define hand marker indices
    hand_marker_indices = torch.cat([torch.arange(64, 79), torch.arange(121, 143)]).to(device)

    with torch.no_grad():
        for i, (clip_img_joint, clip_img_marker, *_ ) in enumerate(tqdm(val_loader, desc="Validating")):
            clip_img_joint = clip_img_joint.to(device)
            clip_img_joint = clip_img_joint[:, :, :22, :]  # Use only the first 22 joints
            clip_img_marker = clip_img_marker.to(device)

            # Permute dimensions to match [batch_size, sequence_length, num_joints, 3]
            clip_img_joint = clip_img_joint.permute(0, 3, 2, 1)  # Shape: [batch_size, sequence_length, num_joints, 3]
            clip_img_marker = clip_img_marker.permute(0, 3, 2, 1)  # Shape: [batch_size, sequence_length, num_markers, 3]

            batch_size, num_frames, num_joints, coords = clip_img_joint.shape
            assert coords == 3, f"Expected last dimension to be 3, got {coords}"

            # Ensure tensor is contiguous before reshaping
            input_for_transformer = clip_img_joint.contiguous().view(batch_size * num_frames, num_joints, coords)

            # Forward pass
            predicted_markers = model(input_for_transformer)  # Output: [batch_size * num_frames, markers, 3]

            # Reshape output back to [batch_size, num_frames, markers, 3]
            predicted_markers = predicted_markers.view(batch_size, num_frames, -1, coords)

            # Compute the per-marker loss (Mean Squared Error)
            loss_per_marker = (predicted_markers - clip_img_marker) ** 2

            # Create weight tensor (double the weight for hand markers)
            weights = torch.ones_like(loss_per_marker).to(device)
            weights[:, :, hand_marker_indices, :] *= 2.0

            # Weighted Mean Squared Error
            weighted_loss = (loss_per_marker * weights).sum() / weights.sum()

            # Accumulate loss
            val_loss += weighted_loss.item()

            # Save reconstruction for the first batch if needed
            if save_reconstruction and not first_batch_saved and save_dir and epoch and i == 0:
                save_reconstruction_npz(
                    markers=clip_img_joint,
                    reconstructed_markers=predicted_markers,
                    original_markers=clip_img_marker,
                    mask=None,  # Mask is not used
                    save_dir=save_dir,
                    epoch=epoch,
                    exp_name=exp_name
                )
                first_batch_saved = True

    # Compute average validation loss
    avg_val_loss = val_loss / len(val_loader)

    # Log validation metrics to WandB
    wandb.log({
        "Validation Loss (Reconstruction)": avg_val_loss,
        "Epoch": epoch
    })

    return avg_val_loss


def train(model, optimizer, train_loader, val_loader, logger, checkpoint_dir, exp_name):
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for clip_img_joint, clip_img_marker, *_ in progress_bar:
            clip_img_joint = clip_img_joint.to(DEVICE)
            clip_img_joint = clip_img_joint[:, :, :22, :]
            clip_img_marker = clip_img_marker.to(DEVICE)


            # Permute dimensions to match [batch_size, sequence_length, num_joints, 3]
            clip_img_joint = clip_img_joint.permute(0, 3, 2, 1)  # Shape: [batch_size, sequence_length, num_joints, 3]
            clip_img_marker = clip_img_marker.permute(0, 3, 2, 1)  # Shape: [batch_size, sequence_length, num_markers, 3]


            batch_size, num_frames, num_joints, coords = clip_img_joint.shape
            assert coords == 3, f"Expected last dimension to be 3, got {coords}"

            # Ensure tensor is contiguous before reshaping
            input_for_transformer = clip_img_joint.contiguous().view(batch_size * num_frames, num_joints, coords)

            # Forward pass
            predicted_markers = model(input_for_transformer)  # Output: [batch_size * num_frames, markers, 3]

            # Reshape output back to the original format: [batch_size, num_frames, markers, 3]
            predicted_markers = predicted_markers.view(batch_size, num_frames, -1, coords)

            # Compute the per-marker loss (Mean Squared Error)
            loss_per_marker = (predicted_markers - clip_img_marker) ** 2

            # Weighted loss (optional)
            weights = torch.ones_like(loss_per_marker).to(DEVICE)  # Initialize all weights as 1

            # Define hand marker indices (example based on your structure)
            hand_marker_indices = torch.cat([torch.arange(64, 79), torch.arange(121, 143)]).to(DEVICE)  
            weights[:, :, hand_marker_indices, :] *= 2.0  # Double the weight for hand markers

            # Weighted Mean Squared Error
            weighted_loss = (loss_per_marker * weights).sum() / weights.sum()

            # Backpropagation
            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            # Accumulate epoch loss
            epoch_loss += weighted_loss.item()
            progress_bar.set_postfix({
                "Loss": weighted_loss.item()
            })

        # Average loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{EPOCHS} Loss: {avg_epoch_loss:.4f}")
        wandb.log({"Training Loss": avg_epoch_loss, "Epoch": epoch + 1})

        # Validate every few epochs (optional)
        if (epoch + 1) % 5 == 0:
            save_dir = os.path.join(checkpoint_dir, "validation_reconstructions")
            val_loss = validate(
                model, val_loader, DEVICE, 
                save_reconstruction=True, save_dir=save_dir, 
                epoch=epoch + 1, exp_name=exp_name
            )
            logger.info(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")
            wandb.log({"Validation Loss": val_loss, "Epoch": epoch + 1})

            

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train LiftupTransformer')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    args = parser.parse_args()
    exp_name = args.exp_name
    checkpoint_dir = os.path.join('finetune_liftup_log', exp_name)
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

    wandb.init(entity='edward-effendy-tokyo-tech696', project='LiftupTransformer', name=exp_name, mode='disabled')
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

    grab_dir = '../../../data/edwarde/dataset/preprocessed_grab'
    train_datasets = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
    test_datasets = ['s9', 's10']

    # Initialize Dataset and DataLoaders
    train_dataset = PreprocessedMotionLoader(grab_dir, train_datasets)
    val_dataset = PreprocessedMotionLoader(grab_dir, test_datasets)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    liftup_transformer = LiftUpTransformer(
        input_dim=3,
        embed_dim=64,
        num_joints=NUM_JOINTS,
        num_markers=NUM_MARKERS,
        num_layers=6,
        num_heads=4,
    ).to(DEVICE)
    
    
    # Load checkpoint
    if os.path.exists(LIFTUP_CHECKPOINT_PATH):
        logger.info(f"Loading model from checkpoint: {LIFTUP_CHECKPOINT_PATH}")
        checkpoint = torch.load(LIFTUP_CHECKPOINT_PATH, map_location=DEVICE)
        
        # Fix keys with "module." prefix
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")  # Remove "module." prefix
            new_state_dict[new_key] = value

        # Load the cleaned state_dict
        liftup_transformer.load_state_dict(new_state_dict)
        logger.info("Checkpoint loaded successfully.")
    else:
        logger.warning(f"Checkpoint not found at {LIFTUP_CHECKPOINT_PATH}. Training from scratch.")


    num_params = count_learnable_parameters(liftup_transformer)
    logger.info(f"Number of learnable parameters in LiftUpTransformer: {num_params:,}")
    wandb.config.update({"num_learnable_parameters": num_params})

    optimizer = optim.AdamW(liftup_transformer.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Start training
    train(liftup_transformer, optimizer, train_loader, val_loader, logger, checkpoint_dir, exp_name)

