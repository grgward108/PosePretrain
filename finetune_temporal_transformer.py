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
    mask_np = mask.cpu().numpy()
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

def validate(model, val_loader, mask_ratio, device, save_reconstruction=False, save_dir=None, epoch=None):
    model.eval()
    val_loss = 0.0
    velocity_loss_total = 0.0
    first_batch_saved = False

    with torch.no_grad():
        for i, (clip_img, slerp_img, *_ ) in enumerate(tqdm(val_loader, desc="Validating")):
            original_clip = clip_img.to(device)
            slerp_img = slerp_img.to(device)

            # Permute to match expected input format
            original_clip = original_clip.permute(0, 3, 2, 1)
            # Forward pass
            outputs = model(slerp_img)

            # Generate static mask and expand
            static_mask = generate_static_mask(length=slerp_img.shape[-1]).to(device)
            static_mask = static_mask.unsqueeze(0).unsqueeze(0).expand(outputs.shape)


            # Invert mask and compute unseen losses
            inverted_mask = static_mask
            unseen_outputs = outputs * inverted_mask
            unseen_original = original_clip * inverted_mask

            unseen_loss = ((unseen_outputs - unseen_original) ** 2) * inverted_mask
            val_loss += unseen_loss.sum().item() / inverted_mask.sum().item()

            # Compute velocity loss
            original_velocity = original_clip[:, :, :, 1:] - original_clip[:, :, :, :-1]
            reconstructed_velocity = outputs[:, :, :, 1:] - outputs[:, :, :, :-1]
            velocity_mask = inverted_mask[:, :, :, 1:]
            velocity_loss_total += ((original_velocity - reconstructed_velocity) ** 2 * velocity_mask).sum().item() / velocity_mask.sum().item()

            # Save reconstruction
            if save_reconstruction and not first_batch_saved and save_dir and epoch and i == 0:
                save_reconstruction_npz(
                    markers=slerp_img,
                    reconstructed_markers=outputs,
                    original_markers=original_clip,
                    mask=inverted_mask.squeeze(-1).squeeze(-1),
                    save_dir=save_dir,
                    epoch=epoch
                )
                first_batch_saved = True

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

        for clip_img, slerp_img, mask, *_ in progress_bar:
            original_clip = clip_img.to(DEVICE)
            original_clip = original_clip.permute(0, 3, 2, 1)
            slerp_img = slerp_img.to(DEVICE)
            # Forward pass
            outputs = model(slerp_img)
            static_mask = generate_static_mask(length=slerp_img.shape[-1]).to(DEVICE)
            static_mask =  static_mask.unsqueeze(0).unsqueeze(0)
            static_mask = static_mask.expand(outputs.shape)
            # Invert mask to compute loss for unseen elements
            inverted_mask = static_mask
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
                mask_ratio=MASK_RATIO,
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

