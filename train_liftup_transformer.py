import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.distributed import DistributedSampler
import argparse
import logging
import time

from LiftUpTransformer.models import LiftUpTransformer  # Import your LiftUpTransformer model
from dataloader import FrameLoader  # Import your FrameLoader dataloader

# Hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 6
NUM_JOINTS = 17  # Number of input joints
NUM_MARKERS = 143  # Number of output markers
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths and Dataset Parameters
DATASET_DIR = '../../../../gs/bs/tga-openv/edwarde/AMASS'
SMPLX_MODEL_PATH = '../../../../gs/bs/tga-openv/edwarde/body_utils/body_models'
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


def train(model, dataloader, optimizer, epoch, scaler, logger, DEVICE):
    model.train()
    epoch_loss = 0.0
    start_time = time.time()

    for i, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")):
        # Data preparation
        joints = batch['joints'].to(DEVICE, non_blocking=True)  # Shape: [batch_size, NUM_JOINTS, 3]
        original_markers = batch['original_markers'].to(DEVICE, non_blocking=True)  # Shape: [batch_size, NUM_MARKERS, 3]

        optimizer.zero_grad()

        with autocast('cuda'):
            # Forward pass
            reconstructed_markers = model(joints)
            # Loss computation
            loss = criterion(reconstructed_markers, original_markers)

        # Backward pass with GradScaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    logger.info(f"Epoch {epoch + 1} Training Loss: {avg_loss:.8f}")
    logger.info(f"Rank {dist.get_rank()} | Epoch {epoch + 1} completed in {time.time() - start_time:.2f} seconds.")
    return avg_loss


def validate(model, dataloader, epoch, logger):
    model.eval()
    epoch_loss = 0.0
    valid_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Validation Epoch {epoch + 1}")):
            joints = batch['joints'].to(DEVICE, non_blocking=True)  # Shape: [batch_size, NUM_JOINTS, 3]
            original_markers = batch['original_markers'].to(DEVICE, non_blocking=True)  # Shape: [batch_size, NUM_MARKERS, 3]

            # Forward pass
            reconstructed_markers = model(joints)

            # Compute loss
            loss = criterion(reconstructed_markers, original_markers)
            epoch_loss += loss.item()
            valid_batches += 1

        avg_loss = epoch_loss / valid_batches if valid_batches > 0 else 0.0
    logger.info(f"Epoch {epoch + 1} Validation Loss: {avg_loss:.8f}")
    return avg_loss


def main(exp_name, args):
    local_rank = int(os.environ["LOCAL_RANK"])  # Get the local rank
    DEVICE = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(DEVICE)

    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://')
    scaler = GradScaler()

    print(f"[Process {dist.get_rank()}] Using device: {DEVICE}")
    SAVE_DIR = os.path.join('liftup_log', exp_name, 'ckpt')
    os.makedirs(SAVE_DIR, exist_ok=True)
    logger = setup_logger(exp_name, SAVE_DIR)

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

    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=train_dataset.collate_fn,
        num_workers=32,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        collate_fn=val_dataset.collate_fn,
        num_workers=32,
        pin_memory=True,
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

    model = DDP(model, device_ids=[DEVICE.index], output_device=DEVICE.index)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # Training Loop
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        train_loss = train(model, train_loader, optimizer, epoch, scaler, logger, DEVICE)
        scheduler.step()

        # Validation
        if local_rank == 0 and (epoch + 1) % 2 == 0:
            val_loss = validate(model, val_loader, epoch, logger)
            wandb.log({"epoch": epoch + 1, "training_loss": train_loss, "validation_loss": val_loss})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(SAVE_DIR, f"best_model_epoch_{epoch + 1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f"Saved Best Model to {checkpoint_path}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiftUp Transformer Training Script")
    parser.add_argument("--exp_name", required=True, help="Name of the experiment for logging and saving checkpoints")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    args = parser.parse_args()
    main(args.exp_name, args)
