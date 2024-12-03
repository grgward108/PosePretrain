import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from SpatialTransformer.models.models import SpatialTransformer
from SpatialTransformer.data.dataloader import FrameLoader
import torch.multiprocessing as mp
import argparse
import logging  # Import logging for logging functionality
import numpy as np 
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler


# Hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 0.001

NUM_EPOCHS = 50
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 6
N_PARTS = 9  # Number of body parts
N_MARKERS = 143  # Number of markers
MASKING_RATIO = 0.15  # Ratio of markers to mask
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths and Dataset Parameters
DATASET_DIR = '../../../../gs/bs/tga-openv/edwarde/AMASS'
SMPLX_MODEL_PATH = '../../../../gs/bs/tga-openv/edwarde/body_utils/body_models'
MARKERS_TYPE = 'f15_p22'
NORMALIZE = True

TRAIN_DATASET = ['HumanEva', 'ACCAD', 'CMU','DanceDB', 'Eyes_Japan_Dataset', 'GRAB']
VAL_DATASET = ['HUMAN4D', 'KIT']

# Define Loss Function
criterion = nn.MSELoss()


def setup_logger(exp_name, save_dir):
    """Setup logger to write logs to both console and a file."""
    log_file = os.path.join(save_dir, f"{exp_name}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # File handler to write logs to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler to display logs in the terminal
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def train(model, dataloader, optimizer, epoch, logger):
    model.train()
    epoch_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
        # Data preparation
        markers = batch['markers'].to(DEVICE, non_blocking=True)
        original_markers = batch['original_markers'].to(DEVICE, non_blocking=True)
        part_labels = batch['part_labels'].to(DEVICE, non_blocking=True)
        mask = batch['mask'].to(DEVICE, non_blocking=True)
        optimizer.zero_grad()

        with autocast():
            reconstructed_markers = model(markers, part_labels, mask=mask)
            # Loss computation
            mask_expanded = mask.unsqueeze(-1)
            masked_elements = mask_expanded.sum()
            if masked_elements > 0:
                loss = criterion(reconstructed_markers * mask_expanded, original_markers * mask_expanded)
                loss = loss / masked_elements
            else:
                continue

        # Backward pass with GradScaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    logger.info(f"Epoch {epoch + 1} Training Loss: {avg_loss:.8f}")
    return avg_loss


def save_reconstruction_npz(markers, reconstructed_markers, original_markers, mask, save_dir, epoch):
    """
    Save reconstruction data for visualization in an .npz file.
    Args:
        markers (torch.Tensor): Masked input markers [batch_size, n_markers, 3].
        reconstructed_markers (torch.Tensor): Reconstructed markers [batch_size, n_markers, 3].
        original_markers (torch.Tensor): Ground truth markers [batch_size, n_markers, 3].
        mask (torch.Tensor): Mask tensor [batch_size, n_markers].
        save_dir (str): Directory to save the results.
        epoch (int): Current epoch number.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Convert to numpy
    masked = markers.cpu().numpy()
    reconstructed = reconstructed_markers.cpu().numpy()
    ground_truth = original_markers.cpu().numpy()
    mask_np = mask.cpu().numpy()

    # Save to .npz file
    npz_path = os.path.join(save_dir, f"epoch_{epoch}_reconstruction.npz")
    np.savez_compressed(npz_path, masked=masked, reconstructed=reconstructed, ground_truth=ground_truth, mask=mask_np)
    print(f"Saved reconstruction data to {npz_path}")



def validate(model, dataloader, epoch, logger, save_dir):
    model.eval()
    epoch_loss = 0.0
    valid_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Validation Epoch {epoch + 1}")):
            markers = batch['markers'].to(DEVICE, non_blocking=True)
            original_markers = batch['original_markers'].to(DEVICE, non_blocking=True)
            part_labels = batch['part_labels'].to(DEVICE, non_blocking=True)
            mask = batch['mask'].to(DEVICE, non_blocking=True)

            # Forward pass
            reconstructed_markers = model(markers, part_labels, mask=mask)

            # Expand mask for loss computation
            mask_expanded = mask.unsqueeze(-1)

            # Compute losses
            diff = reconstructed_markers - original_markers
            raw_loss = (diff ** 2).mean()  # MSE over all markers

            masked_elements = mask_expanded.sum()
            if masked_elements > 0:
                masked_diff = diff * mask_expanded
                masked_loss = (masked_diff ** 2).sum() / masked_elements  # MSE over masked markers
            else:
                masked_loss = 0.0

            # Combine losses
            alpha = 0.9
            loss = alpha * masked_loss + (1 - alpha) * raw_loss

            # Aggregate epoch loss
            epoch_loss += loss.item()
            valid_batches += 1

            # Save reconstruction for the first batch
            if batch_idx == 0:
                logger.info("Saving reconstruction examples from the first validation batch...")
                save_reconstruction_npz(markers, reconstructed_markers, original_markers, mask, save_dir, epoch)

        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
        else:
            logger.error("No valid batches in validation set!")
            avg_loss = 0.0

    logger.info(f"Epoch {epoch + 1} Validation Loss: {avg_loss:.8f}")
    return avg_loss



def main(exp_name):
    # Set SAVE_DIR dynamically based on exp_name
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank()
    DEVICE = torch.device(f'cuda:{local_rank}')
    SAVE_DIR = os.path.join('spatial_log', exp_name, 'ckpt')
    os.makedirs(SAVE_DIR, exist_ok=True)
    logger = setup_logger(exp_name, SAVE_DIR)
    
    logger.info(f"[INFO] Checkpoints will be saved to: {SAVE_DIR}")

    logger.info(f"""[INFO] The configuration is as follows:
                \t- Batch Size: {BATCH_SIZE}
                \t- Learning Rate: {LEARNING_RATE}
                \t- Number of Epochs: {NUM_EPOCHS}
                \t- Embedding Dimension: {EMBED_DIM}
                \t- Number of Heads: {NUM_HEADS}
                \t- Number of Layers: {NUM_LAYERS}
                \t- Number of Parts: {N_PARTS}
                \t- Number of Markers: {N_MARKERS}
                \t- Masking Ratio: {MASKING_RATIO}
                \t- Device: {DEVICE}""")

    wandb.init(entity='edward-effendy-tokyo-tech696', project='PoseTrain_Spatial', name=exp_name)

    # Update WandB config
    wandb.config.update({
        "experiment": exp_name,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
        "embed_dim": EMBED_DIM,
        "num_heads": NUM_HEADS,
        "num_layers": NUM_LAYERS,
        "n_parts": N_PARTS,
        "n_markers": N_MARKERS,
        "masking_ratio": MASKING_RATIO,
    })
    

    # Initialize Dataset and Dataloader
    logger.info("[INFO] Initializing FrameLoader...")

    train_dataset = FrameLoader(
        dataset_dir=DATASET_DIR,
        smplx_model_path=SMPLX_MODEL_PATH,
        markers_type=MARKERS_TYPE,
        normalize=NORMALIZE,
        dataset_list=TRAIN_DATASET,
        apply_masking=True,        # Enable masking for training
        masking_ratio=MASKING_RATIO       # Adjust masking ratio as desired
    )


    val_dataset = FrameLoader(
        dataset_dir=DATASET_DIR,
        smplx_model_path=SMPLX_MODEL_PATH,
        markers_type=MARKERS_TYPE,
        normalize=NORMALIZE,
        dataset_list=VAL_DATASET,
        apply_masking=True,        # Enable masking during validation
        masking_ratio=MASKING_RATIO       # Same or different masking ratio as training
    )


    logger.info(f"[INFO] Number of training samples: {len(train_dataset)}")
    logger.info(f"[INFO] Number of validation samples: {len(val_dataset)}")


    from torch.utils.data.distributed import DistributedSampler

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=train_dataset.collate_fn,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        collate_fn=val_dataset.collate_fn,
        num_workers=6,
        pin_memory=True
    )


    # Initialize Model and Optimizer
    logger.info("[INFO] Initializing Model...")
    model = SpatialTransformer(
        n_markers=N_MARKERS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        n_parts=N_PARTS
    ).to(DEVICE)

    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)


    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)



    # Training Loop
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        logger.info(f"\n[INFO] Starting Epoch {epoch + 1}/{NUM_EPOCHS}...")
        train_loss = train(model, train_loader, optimizer, epoch, logger)


        scheduler.step()

        if local_rank == 0:
            # Perform validation every 3 epochs
            if (epoch + 1) % 2 == 0:
                val_loss = validate(model, val_loader, epoch, logger, save_dir=SAVE_DIR)

                wandb.log({
                    "epoch": epoch + 1,
                    "training_loss": train_loss,
                    "validation_loss": val_loss,
                    "learning_rate": scheduler.get_last_lr()[0],  # Log current learning rate
                })

                # Save Best Model if Validation Improves
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(SAVE_DIR, f"best_model_epoch_{epoch + 1}.pth")
                    torch.save(model.state_dict(), checkpoint_path)
                    logger.info(f"[INFO] Saved Best Model to {checkpoint_path}.")

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch + 1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"[INFO] Checkpoint saved at {checkpoint_path}.")

    logger.info("[INFO] Training Complete.")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", required=True, help="Experiment name")
    parser.add_argument("--local_rank", type=int, help="Local rank for distributed training")
    args = parser.parse_args()
    main(args.exp_name)
