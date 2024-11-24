import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from TemporalTransformer.models.models import TemporalTransformer
from TemporalTransformer.data.dataloader import MotionLoader
import argparse
import logging

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
MASK_RATIO = 0.15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLIP_SECONDS = 2
CLIP_FPS = 30
MARKERS_TYPE = 'f15_p5'
MODE = 'local_joints_3dv'
SMPLX_MODEL_PATH = 'body_utils/body_models'

# Validation frequency
VALIDATE_EVERY = 10


def validate(model, criterion, val_loader, mask_ratio, device):
    """
    Validate the model on the validation dataset with the same masking logic as training.
    Args:
        model: The model to validate.
        criterion: Loss function.
        val_loader: DataLoader for the validation dataset.
        mask_ratio: Ratio of markers to mask for validation.
        device: The device to run the validation on.
    Returns:
        Average validation loss.
    """
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for masked_clip, mask, original_clip in tqdm(val_loader, desc="Validating"):
            masked_clip = masked_clip.to(device)
            original_clip = original_clip.to(device)

            # Dynamically create a mask (same logic as training)
            mask = (torch.rand(masked_clip.shape[:-1], device=device) < mask_ratio).float()
            mask = mask.unsqueeze(-1).expand_as(masked_clip)  # Match shape to [B, T, J, C]
            masked_clip[mask.bool()] = 0.0  # Zero out masked markers

            # Forward pass
            outputs = model(masked_clip)

            # Compute loss only on masked parts
            loss = criterion(outputs[mask.bool()], original_clip[mask.bool()])
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss




def train(model, optimizer, criterion, train_loader, val_loader, logger, checkpoint_dir):
    """
    Train the model and validate periodically.
    Args:
        model: The TemporalTransformer model.
        optimizer: Optimizer for the model.
        criterion: Loss function.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
        logger: Logger for recording training and validation logs.
        checkpoint_dir: Directory to save model checkpoints.
    """
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for masked_clip, mask, original_clip in progress_bar:
            masked_clip = masked_clip.to(DEVICE)
            original_clip = original_clip.to(DEVICE)

            # Forward pass
            outputs = model(masked_clip)

            # Masking to compute loss only on masked frames
            mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(outputs).to(DEVICE)
            loss = criterion(outputs[mask.bool()], original_clip[mask.bool()])

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})

        # Log average training loss for the epoch
        avg_epoch_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Training Loss: {avg_epoch_loss:.4f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved at {checkpoint_path}")

        # Validate every VALIDATE_EVERY epochs
        if (epoch + 1) % VALIDATE_EVERY == 0:
            val_loss = validate(model, criterion, val_loader, MASK_RATIO, DEVICE)
            logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Validation Loss: {val_loss:.4f}")



if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description='Train TemporalTransformer')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    args = parser.parse_args()
    exp_name = args.exp_name

    # Create checkpoint and log directories
    checkpoint_dir = os.path.join('logs', 'TemporalTransformer', exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Configure logging
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

    # Dataset and DataLoader
    train_dataset = MotionLoader(
        clip_seconds=CLIP_SECONDS,
        clip_fps=CLIP_FPS,
        normalize=True,
        split='train',
        markers_type=MARKERS_TYPE,
        mode=MODE,
        mask_ratio=MASK_RATIO,
        log_dir='./logs',
    )
    train_dataset.read_data(['HumanEva', 'ACCAD', 'CMU','DanceDB', 'Eyes_Japan_Dataset', 'GRAB'], amass_dir='../../../data/edwarde/dataset/AMASS')
    train_dataset.create_body_repr(smplx_model_path=SMPLX_MODEL_PATH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    val_dataset = MotionLoader(
        clip_seconds=CLIP_SECONDS,
        clip_fps=CLIP_FPS,
        normalize=True,
        split='val',
        markers_type=MARKERS_TYPE,
        mode=MODE,
        mask_ratio=MASK_RATIO,  # Apply the same masking ratio as training
        log_dir='./logs',
    )

    val_dataset.read_data(['HUMAN4D', 'KIT'], amass_dir='../../../data/edwarde/dataset/AMASS')
    val_dataset.create_body_repr(smplx_model_path=SMPLX_MODEL_PATH)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Model
    model = TemporalTransformer(
        dim_in=3,
        dim_out=3,
        dim_feat=128,
        depth=5,
        num_heads=8,
        num_joints=train_dataset[0][0].shape[1],  # Number of joints/markers
        maxlen=CLIP_SECONDS * CLIP_FPS + 1,
    ).to(DEVICE)

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Start training
    train(model, optimizer, criterion, train_loader, val_loader, logger, checkpoint_dir)
