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

# Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 2e-4

NUM_EPOCHS = 50
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 6
N_PARTS = 9  # Number of body parts
N_MARKERS = 143  # Number of markers
MASKING_RATIO = 0.35  # Ratio of markers to mask
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths and Dataset Parameters
DATASET_DIR = '../../../data/edwarde/dataset/AMASS'
SMPLX_MODEL_PATH = 'body_utils/body_models'
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
        # Get masked markers, original markers, part labels, and mask
        markers = batch['markers'].to(DEVICE)              # Masked markers (input to the model)
        original_markers = batch['original_markers'].to(DEVICE)  # Unmasked markers (ground truth)
        part_labels = batch['part_labels'].to(DEVICE)
        mask = batch['mask'].to(DEVICE)

        # Forward pass
        optimizer.zero_grad()
        reconstructed_markers = model(markers, part_labels, mask=mask)

        # Expand mask for loss computation
        mask_expanded = mask.unsqueeze(-1)  # Shape: [batch_size, n_markers, 1]

        # Compute loss only on masked markers
        masked_elements = mask_expanded.sum()
        if masked_elements > 0:
            loss = criterion(reconstructed_markers * mask_expanded, original_markers * mask_expanded)
            loss = loss / masked_elements  # Normalize by the number of masked elements
        else:
            # If no elements are masked, skip the batch
            continue

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

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
    valid_batches = 0  # Track the number of valid batches

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Validation Epoch {epoch + 1}")):
            # Get masked markers, original markers, part labels, and mask
            markers = batch['markers'].to(DEVICE)
            original_markers = batch['original_markers'].to(DEVICE)
            part_labels = batch['part_labels'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)

            # Forward pass
            reconstructed_markers = model(markers, part_labels, mask=mask)

            # Expand mask for loss computation
            mask_expanded = mask.unsqueeze(-1)

            # Compute loss only on masked markers
            masked_elements = mask_expanded.sum()
            if masked_elements > 0:
                diff = (reconstructed_markers - original_markers) * mask_expanded
                loss = (diff ** 2).sum() / masked_elements  # Manually compute MSE for masked markers
                epoch_loss += loss.item()
                valid_batches += 1  # Increment valid batch counter
            else:
                logger.warning("No masked elements in this batch, skipping...")
                continue

            # Save reconstruction for the first batch
            if batch_idx == 0:
                logger.info("Saving reconstruction examples from the first validation batch...")
                save_reconstruction_npz(markers, reconstructed_markers, original_markers, mask, save_dir, epoch)

        # Compute average loss only over valid batches
        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
        else:
            logger.error("No valid batches in validation set!")
            avg_loss = 0.0

    logger.info(f"Epoch {epoch + 1} Validation Loss: {avg_loss:.8f}")
    return avg_loss



def main(exp_name):
    # Set SAVE_DIR dynamically based on exp_name
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


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=4  # Adjust as needed
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=4  # Adjust as needed
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

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        logger.info(f"\n[INFO] Starting Epoch {epoch + 1}/{NUM_EPOCHS}...")
        train_loss = train(model, train_loader, optimizer, epoch, logger)

        # Perform validation every 3 epochs
        if (epoch + 1) % 1 == 0:
            val_loss = validate(model, val_loader, epoch, logger, save_dir=SAVE_DIR)

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


if __name__ == "__main__":
    mp.set_start_method('spawn')

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Spatial Transformer Training Script")
    parser.add_argument("--exp_name", required=True, help="Name of the experiment for logging and saving checkpoints")
    args = parser.parse_args()

    main(args.exp_name)
