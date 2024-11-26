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

# Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 2e-4

NUM_EPOCHS = 50
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 6
N_PARTS = 9  # Number of body parts
N_MARKERS = 143  # Number of markers
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths and Dataset Parameters
DATASET_DIR = '../../../data/edwarde/dataset/AMASS'
SMPLX_MODEL_PATH = 'body_utils/body_models'
MARKERS_TYPE = 'f15_p22'
NORMALIZE = True

TRAIN_DATASET = ['HumanEva', 'ACCAD']
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
        markers = batch['markers'].to(DEVICE)          # [bs, n_markers, 3]
        part_labels = batch['part_labels'].to(DEVICE)  # [bs, n_markers]
        mask = batch['mask'].to(DEVICE)                # [bs, n_markers]

        # Forward pass
        optimizer.zero_grad()
        reconstructed_markers = model(markers, part_labels, mask=mask)

        # Compute Loss only on masked markers
        # Mask shape: [bs, n_markers]
        # We need to expand mask to match markers shape
        mask_expanded = mask.unsqueeze(-1)  # [bs, n_markers, 1]

        # Calculate loss only for masked markers
        loss = criterion(reconstructed_markers * mask_expanded, markers * mask_expanded)

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    logger.info(f"Epoch {epoch + 1} Training Loss: {avg_loss:.8f}")
    return avg_loss



def validate(model, dataloader, epoch, logger):
    """Validation loop."""
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation Epoch {epoch + 1}"):
            markers = batch['markers'].to(DEVICE)          # [batch_size, n_markers, 3]
            part_labels = batch['part_labels'].to(DEVICE)  # [batch_size, n_markers]
            mask = batch['mask'].to(DEVICE)                # [batch_size, n_markers]

            # Forward pass
            reconstructed_markers = model(markers, part_labels, mask=mask)

            # Compute Loss only on masked markers
            mask_expanded = mask.unsqueeze(-1)  # [batch_size, n_markers, 1]
            masked_elements = mask_expanded.sum()
            num_masked = masked_elements.item()
            logger.info(f"Number of masked markers in batch: {num_masked}")



            if masked_elements > 0:
                loss = criterion(reconstructed_markers * mask_expanded, markers * mask_expanded)
                loss = loss / masked_elements  # Normalize by the number of masked elements
            else:
                loss = torch.tensor(0.0).to(DEVICE)

            epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    logger.info(f"Epoch {epoch + 1} Validation Loss: {avg_loss:.8f}")
    return avg_loss



def main(exp_name):
    # Set SAVE_DIR dynamically based on exp_name
    SAVE_DIR = os.path.join('spatial_log', exp_name, 'ckpt')
    os.makedirs(SAVE_DIR, exist_ok=True)
    logger = setup_logger(exp_name, SAVE_DIR)
    
    logger.info(f"[INFO] Checkpoints will be saved to: {SAVE_DIR}")

    # Initialize Dataset and Dataloader
    logger.info("[INFO] Initializing FrameLoader...")

    train_dataset = FrameLoader(
        dataset_dir=DATASET_DIR,
        smplx_model_path=SMPLX_MODEL_PATH,
        markers_type=MARKERS_TYPE,
        normalize=NORMALIZE,
        dataset_list=TRAIN_DATASET,
        apply_masking=True,        # Enable masking for training
        masking_ratio=0.15         # Adjust masking ratio as desired
    )


    val_dataset = FrameLoader(
        dataset_dir=DATASET_DIR,
        smplx_model_path=SMPLX_MODEL_PATH,
        markers_type=MARKERS_TYPE,
        normalize=NORMALIZE,
        dataset_list=VAL_DATASET,
        apply_masking=True,        # Enable masking during validation
        masking_ratio=0.15         # Same or different masking ratio as training
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

        # Perform validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            val_loss = validate(model, val_loader, epoch, logger)

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
