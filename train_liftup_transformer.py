import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import argparse
import logging
import torch.multiprocessing as mp
import time

from LiftUpTransformer.models.models import LiftUpTransformer
from LiftUpTransformer.data.dataloader import FrameLoader

# Hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 6
NUM_JOINTS = 22  # Number of input joints
NUM_MARKERS = 143  # Number of output markers
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths and Dataset Parameters
DATASET_DIR = '../../../data/edwarde/dataset/AMASS'
SMPLX_MODEL_PATH = 'body_utils/body_models'
MARKERS_TYPE = 'f15_p22'
NORMALIZE = True

TRAIN_DATASET = ['HumanEva', 'ACCAD', 'CMU', 'DanceDB', 'Eyes_Japan_Dataset', 'GRAB']
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


def train(model, dataloader, optimizer, epoch, logger, DEVICE):
    model.train()
    epoch_loss = 0.0
    start_time = time.time()

    for i, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")):
        # Convert NumPy arrays to PyTorch tensors
        joints = torch.tensor(batch['joints'], dtype=torch.float32, device=DEVICE)

        original_markers = batch['original_markers'].clone().detach().to(DEVICE, dtype=torch.float32)

        optimizer.zero_grad()

        # Forward pass
        reconstructed_markers = model(joints)
        # Loss computation
        loss = criterion(reconstructed_markers, original_markers)

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    logger.info(f"Epoch {epoch + 1} Training Loss: {avg_loss:.8f}")
    return avg_loss


def validate(model, dataloader, logger, DEVICE):
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Validation")):
            joints = torch.tensor(batch['joints'], dtype=torch.float32, device=DEVICE)

            original_markers = batch['original_markers'].clone().detach().to(DEVICE, dtype=torch.float32)

            # Forward pass
            reconstructed_markers = model(joints)
            # Loss computation
            loss = criterion(reconstructed_markers, original_markers)

            epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    logger.info(f"Validation Loss: {avg_loss:.8f}")
    return avg_loss



def main(exp_name):
    SAVE_DIR = os.path.join('liftup_log', exp_name, 'ckpt')
    os.makedirs(SAVE_DIR, exist_ok=True)
    logger = setup_logger(exp_name, SAVE_DIR)

    logger.info(f"[INFO] Checkpoints will be saved to: {SAVE_DIR}")

    wandb.init(entity='edward-effendy-tokyo-tech696', project='PoseTrain_LiftUp', name=exp_name)


    # Dataset and Dataloader
    train_dataset = FrameLoader(
        dataset_dir=DATASET_DIR,
        smplx_model_path=SMPLX_MODEL_PATH,
        markers_type=MARKERS_TYPE,
        normalize=NORMALIZE,
        dataset_list=TRAIN_DATASET,
    )
    val_dataset = FrameLoader(
        dataset_dir=DATASET_DIR,
        smplx_model_path=SMPLX_MODEL_PATH,
        markers_type=MARKERS_TYPE,
        normalize=NORMALIZE,
        dataset_list=VAL_DATASET,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=8,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=8,
    )

    # Initialize Model and Optimizer
    model = LiftUpTransformer(
        input_dim=3,
        embed_dim=EMBED_DIM,
        num_joints=NUM_JOINTS,
        num_markers=NUM_MARKERS,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # Training Loop
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        logger.info(f"\n[INFO] Starting Epoch {epoch + 1}/{NUM_EPOCHS}...")
        train_loss = train(model, train_loader, optimizer, epoch, logger, DEVICE)
        scheduler.step()

        # Validation
        if (epoch + 1) % 2 == 0:
            val_loss = validate(model, val_loader, epoch, logger)
            wandb.log({"epoch": epoch + 1, "training_loss": train_loss, "validation_loss": val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(SAVE_DIR, f"best_model_epoch_{epoch + 1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Saved Best Model to {checkpoint_path}")

    wandb.finish()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="LiftUp Transformer Training Script")
    parser.add_argument("--exp_name", required=True, help="Name of the experiment for logging and saving checkpoints")
    args = parser.parse_args()
    main(args.exp_name)
