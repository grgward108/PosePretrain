import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import logging
import numpy as np

from LiftUpTransformer.models.models import LiftUpTransformer
from LiftUpTransformer.data.dataloader import FrameLoader

# Hyperparameters
BATCH_SIZE = 256
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


def save_reconstruction_npz(joints, reconstructed_markers, original_markers, save_dir, epoch):
    """
    Save reconstruction data for visualization in an .npz file.
    Args:
        joints (torch.Tensor): input of joints [batch_size, joint_numbers, 3].
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
            joints = torch.tensor(batch['joints'], dtype=torch.float32).to(DEVICE)
            original_markers = torch.tensor(batch['original_markers'], dtype=torch.float32).to(DEVICE)

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


def main(exp_name, checkpoint_path):
    SAVE_DIR = os.path.join('liftup_log', exp_name)
    os.makedirs(SAVE_DIR, exist_ok=True)
    logger = setup_logger(exp_name, SAVE_DIR)

    logger.info(f"[INFO] Logs and validation outputs will be saved to: {SAVE_DIR}")

    # Dataset and Dataloader
    val_dataset = FrameLoader(
        dataset_dir=DATASET_DIR,
        smplx_model_path=SMPLX_MODEL_PATH,
        markers_type=MARKERS_TYPE,
        normalize=NORMALIZE,
        dataset_list=VAL_DATASET,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=0,
    )

    # Initialize Model
    model = LiftUpTransformer(
        input_dim=3,
        embed_dim=EMBED_DIM,
        num_joints=NUM_JOINTS,
        num_markers=NUM_MARKERS,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
    ).to(DEVICE)


    # Load checkpoint
    if checkpoint_path and os.path.isfile(checkpoint_path):
        logger.info(f"[INFO] Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info(f"[INFO] Successfully loaded checkpoint.")
    else:
        logger.error(f"[ERROR] Checkpoint path {checkpoint_path} is invalid or does not exist.")
        return

    # Validate
    logger.info(f"[INFO] Starting validation...")
    validate(model, val_loader, epoch=start_epoch, logger=logger, DEVICE=DEVICE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiftUp Transformer Validation Script")
    parser.add_argument("--exp_name", required=True, help="Name of the experiment for logging and saving outputs")
    parser.add_argument("--checkpoint_path", required=True, help="Path to the checkpoint to load the model")
    args = parser.parse_args()

    main(args.exp_name, args.checkpoint_path)
