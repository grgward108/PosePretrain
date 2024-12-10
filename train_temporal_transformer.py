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
import numpy as np

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
STRIDE = 30

# Validation frequency
VALIDATE_EVERY = 2

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


def count_learnable_parameters(model):
    """
    Count the number of learnable parameters in a model.
    Args:
        model: The PyTorch model.
    Returns:
        Total number of learnable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def validate(model, val_loader, mask_ratio, device, save_reconstruction=False, save_dir=None, epoch=None):
    """
    Validate the model on the validation dataset with the same masking logic as training.
    Args:
        model: The model to validate.
        val_loader: DataLoader for the validation dataset.
        mask_ratio: Ratio of markers to mask for validation.
        device: The device to run the validation on.
    Returns:
        Average validation loss.
    """
    model.eval()
    val_loss = 0.0
    first_batch_saved = False
    i = 0

    with torch.no_grad():
        for masked_clip, mask, original_clip in tqdm(val_loader, desc="Validating"):
            masked_clip = masked_clip.to(device)
            mask = mask.to(device)
            original_clip = original_clip.to(device)

            # Forward pass
            outputs = model(masked_clip)

            # Expand mask dimensions to match outputs and ground truth
            mask = mask.unsqueeze(-1).unsqueeze(-1)

            # Apply the mask
            masked_outputs = outputs * mask
            masked_original = original_clip * mask

            # Compute the mean squared error loss only on the masked elements
            masked_loss = ((masked_outputs - masked_original) ** 2) * mask
            raw_loss = masked_loss.sum()  # Total loss across all masked elements
            normalized_loss = raw_loss / mask.sum()  # Normalize by the number of masked elements

            val_loss += normalized_loss.item()

            # Save first batch reconstruction if required
            if save_reconstruction and not first_batch_saved:
                if save_dir is not None and epoch is not None and i == 6:  # Save for the first batch
                    save_reconstruction_npz(
                        markers=masked_clip,
                        reconstructed_markers=outputs,
                        original_markers=original_clip,
                        mask=mask.squeeze(-1).squeeze(-1),  # Original mask shape for saving
                        save_dir=save_dir,
                        epoch=epoch
                    )
                    first_batch_saved = True

            i += 1

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss


def train(model, optimizer, train_loader, val_loader, logger, checkpoint_dir):
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for masked_clip, mask, original_clip in progress_bar:
            masked_clip = masked_clip.to(DEVICE)
            original_clip = original_clip.to(DEVICE)

            # Forward pass
            outputs = model(masked_clip)

            # Expand mask dimensions
            mask = mask.to(outputs.device).unsqueeze(-1).unsqueeze(-1)

            # Apply mask
            masked_outputs = outputs * mask
            masked_original = original_clip * mask

            # Compute masked loss and normalize
            masked_loss = ((masked_outputs - masked_original) ** 2) * mask
            raw_loss = masked_loss.sum()  # Total loss
            normalized_loss = raw_loss / mask.sum()  # Normalize by the number of masked elements
            

            # Backward pass and optimization
            optimizer.zero_grad()
            normalized_loss.backward()
            optimizer.step()

            epoch_loss += normalized_loss.item()
            progress_bar.set_postfix({"Loss": normalized_loss.item()})

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
            save_dir = os.path.join(checkpoint_dir, "reconstruction_val")
            val_loss = validate(
                model,
                val_loader=val_loader,
                mask_ratio=MASK_RATIO,
                device=DEVICE,
                save_reconstruction=True,  # Enable saving reconstruction
                save_dir=save_dir,
                epoch=epoch + 1
            )
            logger.info(f"Epoch [{epoch+1}/{EPOCHS}] Validation Loss: {val_loss:.4f}")




if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description='Train TemporalTransformer')
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')
    args = parser.parse_args()
    exp_name = args.exp_name

    # Create checkpoint and log directories
    checkpoint_dir = os.path.join('temporal_log', exp_name)
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

    log_dir = os.path.join('./temporal_log', exp_name)

    # Dataset and DataLoader
    train_dataset = MotionLoader(
        clip_seconds=CLIP_SECONDS,
        clip_fps=CLIP_FPS,
        normalize=True,
        split='train',
        markers_type=MARKERS_TYPE,
        mode=MODE,
        mask_ratio=MASK_RATIO,
        log_dir=log_dir,
    )
    train_dataset.read_data(['HumanEva', 'ACCAD', 'CMU','DanceDB', 'Eyes_Japan_Dataset', 'GRAB'], amass_dir='../../../data/edwarde/dataset/AMASS', stride=STRIDE)
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
        log_dir=log_dir,
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

    # Print the number of learnable parameters
    num_params = count_learnable_parameters(model)
    logger.info(f"Number of learnable parameters in TemporalTransformer: {num_params:,}")

    # Loss and Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Start training
    train(model, optimizer, train_loader, val_loader, logger, checkpoint_dir)