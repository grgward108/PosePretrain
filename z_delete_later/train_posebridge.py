import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PoseBridge.data.dataloader import MotionLoader
from tqdm import tqdm
import wandb
import argparse
import logging
import numpy as np

from PoseBridge.models.models import EndToEndModel
from TemporalTransformer.models.models import TemporalTransformer
from LiftUpTransformer.models.models import LiftUpTransformer

BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
MASK_RATIO = 0.15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLIP_SECONDS = 2
CLIP_FPS = 30
MARKERS_TYPE = 'f15_p5'  # Example markers type
MODE = 'local_joints_3dv'
SMPLX_MODEL_PATH = 'body_utils/body_models'
STRIDE = 30
NUM_JOINTS = 25
NUM_MARKERS = 143

TEMPORAL_CHECKPOINT_PATH = "path/to/temporal_checkpoint.pth"
LIFTUP_CHECKPOINT_PATH = "path/to/liftup_checkpoint.pth"

def save_reconstruction_npz(
    masked_joints, 
    reconstructed_joints, 
    reconstructed_markers, 
    original_joints, 
    original_markers, 
    mask, 
    save_dir, 
    epoch
):
    """
    Save reconstruction data for joints and markers in an .npz file.

    Args:
        masked_joints (torch.Tensor): Masked input joints [batch_size, num_joints, 3].
        reconstructed_joints (torch.Tensor): Reconstructed joints [batch_size, num_joints, 3].
        reconstructed_markers (torch.Tensor): Reconstructed markers [batch_size, num_markers, 3].
        original_joints (torch.Tensor): Ground truth joints [batch_size, num_joints, 3].
        original_markers (torch.Tensor): Ground truth markers [batch_size, num_markers, 3].
        mask (torch.Tensor): Mask tensor [batch_size, num_frames].
        save_dir (str): Directory to save the results.
        epoch (int): Current epoch number.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Convert to numpy arrays
    masked_joints_np = masked_joints.cpu().numpy()
    reconstructed_joints_np = reconstructed_joints.cpu().numpy()
    reconstructed_markers_np = reconstructed_markers.cpu().numpy()
    original_joints_np = original_joints.cpu().numpy()
    original_markers_np = original_markers.cpu().numpy()
    mask_np = mask.cpu().numpy()

    # Save to .npz file
    npz_path = os.path.join(save_dir, f"epoch_{epoch}_reconstruction.npz")
    np.savez_compressed(
        npz_path,
        masked_joints=masked_joints_np,
        reconstructed_joints=reconstructed_joints_np,
        reconstructed_markers=reconstructed_markers_np,
        ground_truth_joints=original_joints_np,
        ground_truth_markers=original_markers_np,
        mask=mask_np
    )
    print(f"Saved reconstruction data to {npz_path}")


# Training Function
def train_combined(model, optimizer, dataloader, criterion_temp, criterion_lift, epoch, logger, DEVICE):
    model.train()
    epoch_loss = 0.0
    for masked_joints, mask, original_joints, original_markers in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
        masked_joints = masked_joints.to(DEVICE, dtype=torch.float32)
        original_joints = original_joints.to(DEVICE, dtype=torch.float32)
        original_markers = original_markers.to(DEVICE, dtype=torch.float32)

        optimizer.zero_grad()

        # Forward pass
        filled_joints, predicted_markers = model(masked_joints)

        # Compute losses
        loss_temp = criterion_temp(filled_joints, original_joints)
        loss_lift = criterion_lift(predicted_markers, original_markers)
        total_loss = loss_temp + loss_lift

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

    avg_loss = epoch_loss / len(dataloader)
    logger.info(f"Epoch {epoch + 1} Training Loss: {avg_loss:.8f}")
    return avg_loss

# Validation Function
def validate_combined(model, dataloader, criterion_temp, criterion_lift, DEVICE, save_dir=None, epoch=None):
    model.eval()
    val_loss = 0.0
    saved_reconstruction = False
    with torch.no_grad():
        for masked_joints, mask, original_joints, original_markers in tqdm(dataloader, desc="Validating"):
            masked_joints = masked_joints.to(DEVICE, dtype=torch.float32)
            original_joints = original_joints.to(DEVICE, dtype=torch.float32)
            original_markers = original_markers.to(DEVICE, dtype=torch.float32)

            # Forward pass
            filled_joints, predicted_markers = model(masked_joints)

            # Compute losses
            loss_temp = criterion_temp(filled_joints, original_joints)
            loss_lift = criterion_lift(predicted_markers, original_markers)
            total_loss = loss_temp + loss_lift

            val_loss += total_loss.item()

            if save_dir and epoch is not None and not saved_reconstruction:
                save_reconstruction_npz(
                    masked_joints=masked_joints,
                    reconstructed_joints=filled_joints,
                    reconstructed_markers=predicted_markers,
                    original_joints=original_joints,
                    original_markers=original_markers,
                    mask=mask,
                    save_dir=save_dir,
                    epoch=epoch
                )
                saved_reconstruction = True

    avg_loss = val_loss / len(dataloader)
    return avg_loss

# Main Training Script
def main(exp_name):

    # Logging and Checkpoint Directories
    SAVE_DIR = os.path.join('posebridge_log', exp_name, 'ckpt')
    os.makedirs(SAVE_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(SAVE_DIR, f"{exp_name}.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    wandb.init(entity='edward-effendy-tokyo-tech696', project='PoseBridge', name=exp_name)

    wandb.config.update({
        "batch_size": BATCH_SIZE,
        "num_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "mask_ratio": MASK_RATIO,
        "stride": STRIDE,
        "clip_seconds": CLIP_SECONDS,
        "clip_fps": CLIP_FPS,
        "num_joints": NUM_JOINTS,
        "num_markers": NUM_MARKERS
    })

    log_dir = os.path.join('./posebridge_log', exp_name)
    os.makedirs(log_dir, exist_ok=True)


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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Models
    temporal_transformer = TemporalTransformer(
        dim_in=3,
        dim_out=3,
        dim_feat=128,
        depth=5,
        num_heads=8,
        num_joints=NUM_JOINTS,  # Number of joints/markers
        maxlen=CLIP_SECONDS * CLIP_FPS + 1  # This should match the input length of sequence
    ).to(DEVICE)

    liftup_transformer = LiftUpTransformer(
        input_dim=3,
        embed_dim=128,
        num_joints=NUM_JOINTS,
        num_markers=143,
        num_layers=6,
        num_heads=4,
    ).to(DEVICE)    

    temporal_transformer.load_state_dict(torch.load(TEMPORAL_CHECKPOINT_PATH, map_location=DEVICE))
    liftup_transformer.load_state_dict(torch.load(LIFTUP_CHECKPOINT_PATH, map_location=DEVICE))

    model = EndToEndModel(temporal_transformer, liftup_transformer).to(DEVICE)

    # Loss Functions and Optimizer
    criterion_temp = nn.MSELoss()
    criterion_lift = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)


    # Training Loop
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        logger.info(f"Starting Epoch {epoch + 1}/{NUM_EPOCHS}")
        train_loss = train_combined(model, optimizer, train_loader, criterion_temp, criterion_lift, epoch, logger, DEVICE)
        val_loss = validate_combined(model, val_loader, criterion_temp, criterion_lift, DEVICE, save_dir=log_dir, epoch=epoch)

        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        scheduler.step()

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_model_epoch_{epoch + 1}.pth"))
            logger.info(f"Saved Best Model for Epoch {epoch + 1}")

        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved intermediate checkpoint at {checkpoint_path}")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoseBridge Training Script")
    parser.add_argument("--exp_name", required=True, help="Experiment name for logging and saving checkpoints")
    args = parser.parse_args()
    main(args.exp_name)
