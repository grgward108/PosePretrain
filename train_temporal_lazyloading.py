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
MASK_RATIO = 0.70
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLIP_SECONDS = 2
CLIP_FPS = 30
MARKERS_TYPE = 'f15_p5'  # Not really used for joint extraction now, but keep consistent
MODE = 'local_joints_3dv'
SMPLX_MODEL_PATH = '../../../../gs/bs/tga-openv/edwarde/body_utils/body_models'
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
    acceleration_loss_total = 0.0
    first_batch_saved = False

    with torch.no_grad():
        for i, (masked_clip, mask, original_clip) in enumerate(tqdm(val_loader, desc="Validating")):
            masked_clip = masked_clip.to(device)
            original_clip = original_clip.to(device)

            # Forward pass
            outputs = model(masked_clip)

            # Expand mask dimensions
            mask = mask.unsqueeze(-1).unsqueeze(-1).to(device)

            # Invert mask for unseen elements
            inverted_mask = 1 - mask
            unseen_outputs = outputs * inverted_mask
            unseen_original = original_clip * inverted_mask

            # Reconstruction loss
            unseen_loss = ((unseen_outputs - unseen_original) ** 2) * inverted_mask
            raw_rec_loss = unseen_loss.sum()
            normalized_rec_loss = raw_rec_loss / inverted_mask.sum()

            # Velocity loss
            original_velocity = original_clip[:, :, :, 1:] - original_clip[:, :, :, :-1]
            reconstructed_velocity = outputs[:, :, :, 1:] - outputs[:, :, :, :-1]
            velocity_diff = (original_velocity - reconstructed_velocity) ** 2
            raw_velocity_loss = velocity_diff.sum()
            normalized_velocity_loss = raw_velocity_loss / velocity_diff.numel()

            # Acceleration smoothness loss
            original_acceleration = original_velocity[:, :, :, 1:] - original_velocity[:, :, :, :-1]
            reconstructed_acceleration = reconstructed_velocity[:, :, :, 1:] - reconstructed_velocity[:, :, :, :-1]
            acceleration_diff = (original_acceleration - reconstructed_acceleration) ** 2
            raw_acceleration_loss = acceleration_diff.sum()
            normalized_acceleration_loss = raw_acceleration_loss / acceleration_diff.numel()

            # Accumulate losses
            val_loss += normalized_rec_loss.item()
            velocity_loss_total += normalized_velocity_loss.item()
            acceleration_loss_total += normalized_acceleration_loss.item()

            # Save reconstruction for the first batch if needed
            if save_reconstruction and not first_batch_saved and save_dir and epoch and i == 0:
                save_reconstruction_npz(
                    markers=masked_clip,
                    reconstructed_markers=outputs,
                    original_markers=original_clip,
                    mask=inverted_mask.squeeze(-1).squeeze(-1),
                    save_dir=save_dir,
                    epoch=epoch
                )
                first_batch_saved = True

    # Average losses
    avg_rec_loss = val_loss / len(val_loader)
    avg_velocity_loss = velocity_loss_total / len(val_loader)
    avg_acceleration_loss = acceleration_loss_total / len(val_loader)
    total_loss = (
        0.7 * avg_rec_loss + 
        0.2 * avg_velocity_loss + 
        0.1 * avg_acceleration_loss
    )

    wandb.log({
        "Validation Loss (Reconstruction)": avg_rec_loss,
        "Validation Loss (Velocity)": avg_velocity_loss,
        "Validation Loss (Acceleration)": avg_acceleration_loss,
        "Validation Loss (Total)": total_loss,
    })

    return total_loss


def train(model, optimizer, train_loader, val_loader, logger, checkpoint_dir):
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        velocity_loss_total = 0.0
        acceleration_loss_total = 0.0  # Track acceleration loss for the epoch

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

            # Compute Reconstruction loss for masked elements
            unseen_loss = ((unseen_outputs - unseen_original) ** 2) * inverted_mask
            raw_rec_loss = unseen_loss.sum()
            normalized_rec_loss = raw_rec_loss / inverted_mask.sum()

            # Velocity loss
            original_velocity = original_clip[:, :, :, 1:] - original_clip[:, :, :, :-1]
            reconstructed_velocity = outputs[:, :, :, 1:] - outputs[:, :, :, :-1]
            velocity_diff = (original_velocity - reconstructed_velocity) ** 2
            raw_velocity_loss = velocity_diff.sum()
            normalized_velocity_loss = raw_velocity_loss / velocity_diff.numel()

            # Acceleration smoothness loss
            original_acceleration = original_velocity[:, :, :, 1:] - original_velocity[:, :, :, :-1]
            reconstructed_acceleration = reconstructed_velocity[:, :, :, 1:] - reconstructed_velocity[:, :, :, :-1]
            acceleration_diff = (original_acceleration - reconstructed_acceleration) ** 2
            raw_acceleration_loss = acceleration_diff.sum()
            normalized_acceleration_loss = raw_acceleration_loss / acceleration_diff.numel()

            # Combine all losses
            total_loss = (
                0.7 * normalized_rec_loss + 
                0.2 * normalized_velocity_loss + 
                0.1 * normalized_acceleration_loss
            )

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update epoch losses
            epoch_loss += normalized_rec_loss.item()
            velocity_loss_total += normalized_velocity_loss.item()
            acceleration_loss_total += normalized_acceleration_loss.item()

            # Log training metrics
            progress_bar.set_postfix({
                "Reconstruction Loss": normalized_rec_loss.item(),
                "Velocity Loss": normalized_velocity_loss.item(),
                "Acceleration Loss": normalized_acceleration_loss.item(),
                "Total Loss": total_loss.item()
            })

            wandb.log({
                "Training Loss (Reconstruction)": normalized_rec_loss.item(),
                "Training Loss (Velocity)": normalized_velocity_loss.item(),
                "Training Loss (Acceleration)": normalized_acceleration_loss.item(),
                "Training Loss (Batch Total)": total_loss.item(),
            })

        # Log average training losses for the epoch
        avg_epoch_rec_loss = epoch_loss / len(train_loader)
        avg_epoch_velocity_loss = velocity_loss_total / len(train_loader)
        avg_epoch_acceleration_loss = acceleration_loss_total / len(train_loader)
        avg_epoch_total_loss = (
            0.7 * avg_epoch_rec_loss + 
            0.2 * avg_epoch_velocity_loss + 
            0.1 * avg_epoch_acceleration_loss
        )

        logger.info(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Reconstruction Loss: {avg_epoch_rec_loss:.4f}, "
            f"Velocity Loss: {avg_epoch_velocity_loss:.4f}, "
            f"Acceleration Loss: {avg_epoch_acceleration_loss:.4f}, "
            f"Total Loss: {avg_epoch_total_loss:.4f}"
        )

        wandb.log({
            "Training Loss (Epoch Reconstruction)": avg_epoch_rec_loss,
            "Training Loss (Epoch Velocity)": avg_epoch_velocity_loss,
            "Training Loss (Epoch Acceleration)": avg_epoch_acceleration_loss,
            "Training Loss (Epoch Total)": avg_epoch_total_loss,
            "Epoch": epoch + 1,
        })

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
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
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
        amass_datasets=['HumanEva', 'ACCAD', 'CMU', 'DanceDB', 'Eyes_Japan_Dataset', 'GRAB'],
        amass_dir='../../../../gs/bs/tga-openv/edwarde/AMASS',
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
        amass_dir='../../../../gs/bs/tga-openv/edwarde/AMASS',
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

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = torch.nn.DataParallel(model)

    num_params = count_learnable_parameters(model)
    logger.info(f"Number of learnable parameters in TemporalTransformer: {num_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Start training
    train(model, optimizer, train_loader, val_loader, logger, checkpoint_dir)
