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
import random
import numpy as np

from LiftUpTransformer.models.models import LiftUpTransformer  
from LiftUpTransformer.data.dataloader import FrameLoader 
from LiftUpTransformer.data.preprocessed_data import PreprocessedDataset 

# Hyperparameters
BATCH_SIZE = 128  # Per GPU batch size
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 6
NUM_JOINTS = 17   # Number of input joints
NUM_MARKERS = 143 # Number of output markers

# Paths and Dataset Parameters
DATASET_DIR = '../../../../gs/bs/tga-openv/edwarde/AMASS/preprocessed'
MARKERS_TYPE = 'f15_p22'
NORMALIZE = True

TRAIN_DATASET = ['HumanEva', 'ACCAD', 'CMU', 'DanceDB', 'Eyes_Japan_Dataset', 'GRAB']
VAL_DATASET = ['HUMAN4D', 'KIT']

# Define Loss Function
criterion = nn.MSELoss()

def setup_logger(exp_name, save_dir, rank=0):
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

def train(model, dataloader, optimizer, epoch, scaler, logger, device):
    model.train()
    epoch_loss = 0.0
    start_time = time.time()

    for i, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch + 1}", disable=(dist.get_rank() != 0))):
        # Data preparation
        joints = batch['joints'].to(device, non_blocking=True)  
        original_markers = batch['original_markers'].to(device, non_blocking=True)  

        optimizer.zero_grad()

        with autocast():
            # Forward pass
            reconstructed_markers = model(joints)
            # Loss computation
            loss = criterion(reconstructed_markers, original_markers)

        # Backward pass with GradScaler
        scaler.scale(loss).backward()

        # Gradient Clipping
        scaler.unscale_(optimizer)
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()

        # Accumulate loss
        epoch_loss += loss.item()

    # Synchronize and compute average loss across all processes
    total_loss_tensor = torch.tensor(epoch_loss, device=device)
    dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
    avg_loss = total_loss_tensor.item() / dist.get_world_size() / len(dataloader)

    if dist.get_rank() == 0:
        logger.info(f"Epoch {epoch + 1} Training Loss: {avg_loss:.8f}")
        logger.info(f"Epoch {epoch + 1} completed in {time.time() - start_time:.2f} seconds.")
    return avg_loss

def validate(model, dataloader, epoch, logger, device):
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
    
def main():
    parser = argparse.ArgumentParser(description="LiftUp Transformer Training Script")
    parser.add_argument("--exp_name", required=True, help="Name of the experiment for logging and saving checkpoints")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    args = parser.parse_args()

    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)

    # Set random seeds for reproducibility
    seed = 42 + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Set up logger
    SAVE_DIR = os.path.join('liftup_log', args.exp_name, 'ckpt')
    if dist.get_rank() == 0:
        os.makedirs(SAVE_DIR, exist_ok=True)
    logger = setup_logger(args.exp_name, SAVE_DIR, rank=dist.get_rank())

    if dist.get_rank() == 0:
        logger.info(f"[INFO] Checkpoints will be saved to: {SAVE_DIR}")
        logger.info(f"""[INFO] The configuration is as follows:
                    \t- Batch Size: {BATCH_SIZE}
                    \t- Learning Rate: {LEARNING_RATE}
                    \t- Number of Epochs: {NUM_EPOCHS}
                    \t- Embedding Dimension: {EMBED_DIM}
                    \t- Number of Heads: {NUM_HEADS}
                    \t- Number of Layers: {NUM_LAYERS}
                    \t- Device: {device}""")

    # Initialize wandb only on the main process
    if dist.get_rank() == 0:
        wandb.init(entity='edward-effendy-tokyo-tech696', project='PoseTrain_LiftUp', name=args.exp_name)
    else:
        os.environ['WANDB_MODE'] = 'offline'

    # Dataset and Dataloader
    PREPROCESSED_TRAIN_DIR = '../../../../gs/bs/tga-openv/edwarde/AMASS/preprocessed'
    PREPROCESSED_VAL_DIR = '../../../../gs/bs/tga-openv/edwarde/AMASS/preprocessed'

    train_dataset = PreprocessedDataset(PREPROCESSED_TRAIN_DIR)
    val_dataset = PreprocessedDataset(PREPROCESSED_VAL_DIR)

    train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=8,
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
    ).to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    scaler = GradScaler()

    # Resume from checkpoint if provided
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info(f"Resumed training from checkpoint: {args.resume}, starting from epoch {start_epoch}")
    else:
        start_epoch = 0

    # Training Loop
    best_val_loss = float('inf')
    checkpoint_interval = 5  # Save checkpoint every 5 epochs

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_sampler.set_epoch(epoch)
        train_loss = train(model, train_loader, optimizer, epoch, scaler, logger, device)
        scheduler.step()

        # Validation
        val_loss = validate(model, val_loader, epoch, logger, device)
        if dist.get_rank() == 0:
            wandb.log({"epoch": epoch + 1, "training_loss": train_loss, "validation_loss": val_loss})

            # Save checkpoint every checkpoint_interval epochs
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch + 1}.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint_path = os.path.join(SAVE_DIR, f"best_model_epoch_{epoch + 1}.pth")
                torch.save(model.state_dict(), best_checkpoint_path)
                logger.info(f"Saved Best Model to {best_checkpoint_path}")

    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    main()
