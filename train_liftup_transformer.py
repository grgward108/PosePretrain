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
import numpy as np

from LiftUpTransformer.models.models import LiftUpTransformer
from LiftUpTransformer.data.dataloader import FrameLoader

# Hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 6
NUM_JOINTS = 25  # Number of input joints
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


def save_reconstruction_npz(joints, reconstructed_markers, original_markers, save_dir, epoch):
    """
    Save reconstruction data for visualization in an .npz file.
    Args:
        joints (torch.Tensor): input of joints [batch_size,joint_numbers , 3].
        reconstructed_markers (torch.Tensor): Reconstructed markers [batch_size, n_markers, 3].
        original_markers (torch.Tensor): Ground truth markers [batch_size, n_markers, 3].

    """
    os.makedirs(save_dir, exist_ok=True)

    # Convert to numpy
    joints = joints.cpu().numpy()
    reconstructed = reconstructed_markers.cpu().numpy()
    ground_truth = original_markers.cpu().numpy()

    # Save to .npz file
    npz_path = os.path.join(save_dir, f"epoch_{epoch}_liftup_reconstruction.npz")
    np.savez_compressed(npz_path, joints=joints, reconstructed=reconstructed, ground_truth=ground_truth)
    print(f"Saved reconstruction data to {npz_path}")


def validate(model, dataloader, epoch, logger, DEVICE):
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Validation")):
            joints = torch.from_numpy(batch['joints']).to(DEVICE, dtype=torch.float32)

            original_markers = batch['original_markers'].clone().detach().to(DEVICE, dtype=torch.float32)

            # Forward pass
            reconstructed_markers = model(joints)
            # Loss computation
            loss = criterion(reconstructed_markers, original_markers)

            epoch_loss += loss.item()

            if batch_idx == 0:
                save_reconstruction_npz(joints, reconstructed_markers, original_markers, 'liftup_log', epoch)

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

    start_epoch = 0  # Initialize start_epoch

    # Load checkpoint if provided
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        logger.info(f"Loading checkpoint '{args.checkpoint_path}'")
        checkpoint = torch.load(args.checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"Resumed from checkpoint '{args.checkpoint_path}' at epoch {start_epoch} with best validation loss {best_val_loss:.8f}")
    else:
        logger.info("No checkpoint found. Starting training from scratch.")
        best_val_loss = float('inf')


    # Training Loop
    for epoch in range(start_epoch, NUM_EPOCHS):
        logger.info(f"\n[INFO] Starting Epoch {epoch + 1}/{NUM_EPOCHS}...")
        train_loss = train(model, train_loader, optimizer, epoch, logger, DEVICE)
        scheduler.step()

        # Validation
        if (epoch + 1) % 2 == 0:
            val_loss = validate(model, val_loader, epoch, logger, DEVICE)
            wandb.log({"epoch": epoch + 1, "training_loss": train_loss, "validation_loss": val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(SAVE_DIR, f"best_model_epoch_{epoch + 1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                }, checkpoint_path)
                logger.info(f"Saved Best Model to {checkpoint_path}")


    wandb.finish()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="LiftUp Transformer Training Script")
    parser.add_argument("--exp_name", required=True, help="Name of the experiment for logging and saving checkpoints")
    parser.add_argument("--checkpoint_path", default=None, help="Path to the checkpoint to resume training")
    args = parser.parse_args()
    main(args.exp_name)
