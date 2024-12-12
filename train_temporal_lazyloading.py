import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from TemporalTransformer.models.models import TemporalTransformer
from TemporalTransformer.data.lazyloading import MotionLoader  # This should be the updated lazy-loading version
import argparse
import logging
import wandb
import numpy as np

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
MASK_RATIO = 0.15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLIP_SECONDS = 2
CLIP_FPS = 30
MARKERS_TYPE = 'f15_p5'  # Not really used for joint extraction now, but keep consistent
MODE = 'local_joints_3dv'
SMPLX_MODEL_PATH = 'body_utils/body_models'
STRIDE = 30

# Validation frequency
VALIDATE_EVERY = 1

def save_reconstruction_npz(markers, reconstructed_markers, original_markers, mask, save_dir, epoch):
    os.makedirs(save_dir, exist_ok=True)
    masked = markers.cpu().numpy()
    reconstructed = reconstructed_markers.cpu().numpy()
    ground_truth = original_markers.cpu().numpy()
    mask_np = mask.cpu().numpy()
    npz_path = os.path.join(save_dir, f"epoch_{epoch}_reconstruction.npz")
    np.savez_compressed(npz_path, masked=masked, reconstructed=reconstructed, ground_truth=ground_truth, mask=mask_np)
    print(f"Saved reconstruction data to {npz_path}")

def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def validate(model, val_loader, mask_ratio, device, save_reconstruction=False, save_dir=None, epoch=None):
    model.eval()
    val_loss = 0.0
    velocity_loss_total = 0.0
    first_batch_saved = False
    i = 0

    with torch.no_grad():
        for masked_clip, mask, original_clip in tqdm(val_loader, desc="Validating"):
            masked_clip = masked_clip.to(device)
            mask = mask.to(device)
            original_clip = original_clip.to(device)

            # Forward pass
            outputs = model(masked_clip)

            # Expand mask dimensions
            mask = mask.unsqueeze(-1).unsqueeze(-1)

            # Invert mask for unseen elements
            inverted_mask = 1 - mask
            unseen_outputs = outputs * inverted_mask
            unseen_original = original_clip * inverted_mask

            # Compute loss for unseen elements
            unseen_loss = ((unseen_outputs - unseen_original) ** 2) * inverted_mask
            raw_loss = unseen_loss.sum()
            normalized_loss = raw_loss / inverted_mask.sum()
            val_loss += normalized_loss.item()

            # Compute velocity loss for masked parts
            original_velocity = original_clip[:, :, :, 1:] - original_clip[:, :, :, :-1]
            reconstructed_velocity = outputs[:, :, :, 1:] - outputs[:, :, :, :-1]
            velocity_mask = inverted_mask[:, :, :, 1:]  # Mask velocity for unseen parts
            velocity_diff = (original_velocity - reconstructed_velocity) ** 2 * velocity_mask
            velocity_loss = velocity_diff.sum() / velocity_mask.sum()
            velocity_loss_total += velocity_loss.item()

            # Save first batch reconstruction if required
            if save_reconstruction and not first_batch_saved:
                if save_dir is not None and epoch is not None and i == 15:  # Save for the first batch
                    save_reconstruction_npz(
                        markers=masked_clip,
                        reconstructed_markers=outputs,
                        original_markers=original_clip,
                        mask=inverted_mask.squeeze(-1).squeeze(-1),  # Use inverted mask for saving
                        save_dir=save_dir,
                        epoch=epoch
                    )
                    first_batch_saved = True

            i += 1

    avg_rec_loss = val_loss / len(val_loader)
    avg_velocity_loss = velocity_loss_total / len(val_loader)
    total_loss = 0.8 * avg_rec_loss + 0.2 * avg_velocity_loss
    wandb.log({
        "Validation Loss (Reconstruction)": avg_rec_loss,
        "Validation Loss (Velocity)": avg_velocity_loss,
        "Validation Loss (Total)": total_loss,
    })
    return total_loss

def train(model, optimizer, train_loader, val_loader, logger, checkpoint_dir):
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        velocity_loss_total = 0.0  # Track velocity loss for the epoch
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for masked_clip, mask, original_clip in progress_bar:
            masked_clip = masked_clip.to(DEVICE)
            original_clip = original_clip.to(DEVICE)

            # Forward pass
            outputs = model(masked_clip)

            # Expand mask dimensions
            mask = mask.to(outputs.device).unsqueeze(-1).unsqueeze(-1)

            # Invert mask to compute loss for unseen elements
            inverted_mask = 1 - mask
            unseen_outputs = outputs * inverted_mask
            unseen_original = original_clip * inverted_mask

            # Compute reconstruction loss for unseen elements
            unseen_loss = ((unseen_outputs - unseen_original) ** 2) * inverted_mask
            raw_rec_loss = unseen_loss.sum()
            normalized_rec_loss = raw_rec_loss / inverted_mask.sum()

            # Compute velocity loss for unseen elements
            original_velocity = original_clip[:, :, :, 1:] - original_clip[:, :, :, :-1]
            reconstructed_velocity = outputs[:, :, :, 1:] - outputs[:, :, :, :-1]
            velocity_mask = inverted_mask[:, :, :, 1:]  # Adjust mask for velocity computation
            velocity_diff = (original_velocity - reconstructed_velocity) ** 2 * velocity_mask
            raw_velocity_loss = velocity_diff.sum()
            normalized_velocity_loss = raw_velocity_loss / velocity_mask.sum()

            # Combine reconstruction and velocity losses
            total_loss = 0.8 * normalized_rec_loss + 0.2 * normalized_velocity_loss

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update epoch loss
            epoch_loss += normalized_rec_loss.item()
            velocity_loss_total += normalized_velocity_loss.item()
            progress_bar.set_postfix({
                "Reconstruction Loss": normalized_rec_loss.item(),
                "Velocity Loss": normalized_velocity_loss.item(),
                "Total Loss": total_loss.item()
            })

            # Log training metrics to WandB
            wandb.log({
                "Training Loss (Reconstruction)": normalized_rec_loss.item(),
                "Training Loss (Velocity)": normalized_velocity_loss.item(),
                "Training Loss (Batch Total)": total_loss.item(),
            })

        # Log average training loss for the epoch
        avg_epoch_rec_loss = epoch_loss / len(train_loader)
        avg_epoch_velocity_loss = velocity_loss_total / len(train_loader)
        avg_epoch_total_loss = 0.8 * avg_epoch_rec_loss + 0.2 * avg_epoch_velocity_loss

        logger.info(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Reconstruction Loss: {avg_epoch_rec_loss:.4f}, "
            f"Velocity Loss: {avg_epoch_velocity_loss:.4f}, "
            f"Total Loss: {avg_epoch_total_loss:.4f}"
        )

        wandb.log({
            "Training Loss (Epoch Reconstruction)": avg_epoch_rec_loss,
            "Training Loss (Epoch Velocity)": avg_epoch_velocity_loss,
            "Training Loss (Epoch Total)": avg_epoch_total_loss,
            "Epoch": epoch + 1,
        })

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
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
                mask_ratio=MASK_RATIO,
                device=DEVICE,
                save_reconstruction=True,  # Enable saving reconstruction
                save_dir=save_dir,
                epoch=epoch + 1
            )
            logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Validation Loss: {val_loss:.4f}")
            wandb.log({"Validation Loss": val_loss, "Epoch": epoch + 1})


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description='Train TemporalTransformer')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    args = parser.parse_args()
    exp_name = args.exp_name

    checkpoint_dir = os.path.join('temporal_log', exp_name)
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

    wandb.init(entity='edward-effendy-tokyo-tech696', project='TemporalTransformer', name=exp_name)

    log_dir = os.path.join('./temporal_log', exp_name)

    # Initialize dataset without reading data immediately
    train_dataset = MotionLoader(
        clip_seconds=CLIP_SECONDS,
        clip_fps=CLIP_FPS,
        normalize=True,
        split='train',
        markers_type=MARKERS_TYPE,
        mode=MODE,
        mask_ratio=MASK_RATIO,
        log_dir=log_dir,
        smplx_model_path=SMPLX_MODEL_PATH
    )

    # Now read the data (lazy loading metadata)
    train_dataset.read_data(
        amass_datasets=['HumanEva', 'ACCAD', 'CMU','DanceDB', 'Eyes_Japan_Dataset', 'GRAB'],
        amass_dir='../../../data/edwarde/dataset/AMASS',
        stride=STRIDE
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    val_dataset = MotionLoader(
        clip_seconds=CLIP_SECONDS,
        clip_fps=CLIP_FPS,
        normalize=True,
        split='val',
        markers_type=MARKERS_TYPE,
        mode=MODE,
        mask_ratio=MASK_RATIO,
        log_dir=log_dir,
        smplx_model_path=SMPLX_MODEL_PATH
    )

    val_dataset.read_data(
        amass_datasets=['HUMAN4D', 'KIT'],
        amass_dir='../../../data/edwarde/dataset/AMASS',
        stride=STRIDE
    )

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # To determine num_joints dynamically from the dataset:
    # Grab a single sample
    sample_masked_clip, sample_mask, sample_original_clip = train_dataset[0]
    num_joints = sample_masked_clip.shape[1]

    model = TemporalTransformer(
        dim_in=3,
        dim_out=3,
        dim_feat=128,
        depth=5,
        num_heads=8,
        num_joints=num_joints,
        maxlen=CLIP_SECONDS * CLIP_FPS + 1  # This should match the input length of your sequence
    ).to(DEVICE)

    num_params = count_learnable_parameters(model)
    logger.info(f"Number of learnable parameters in TemporalTransformer: {num_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Start training
    train(model, optimizer, train_loader, val_loader, logger, checkpoint_dir)
