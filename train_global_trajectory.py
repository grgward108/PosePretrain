import torch
from GlobalTrajectory.data.dataloader import PreprocessedMotionLoader
from GlobalTrajectory.models.models import Traj_MLP_CVAE
import numpy as np
from torch.utils.data import DataLoader
import os
import argparse
import logging

# Constants for training
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 3e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VALIDATE_EVERY = 5

# Data configuration
grab_dir = '../../../data/edwarde/dataset/preprocessed_grab'
train_datasets = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
test_datasets = ['s9', 's10']

# Function to save reconstructions
def save_reconstruction_npz(traj_gt, predicted_traj, save_dir, exp_name, epoch):
    save_path = os.path.join(save_dir, exp_name)
    os.makedirs(save_path, exist_ok=True)
    np.savez_compressed(os.path.join(save_path, f"reconstruction_epoch_{epoch}.npz"),
                        traj_gt=traj_gt, 
                        predicted_traj=predicted_traj)

# Training function
def train(model, optimizer, train_loader, epoch, logger, device):
    model.train()
    epoch_loss = 0
    for i, data in enumerate(train_loader):
        traj, marker_start_global, marker_end_global = data
        traj = traj.to(device)
        marker_start_global = marker_start_global.to(device)
        marker_end_global = marker_end_global.to(device)

        # Linear interpolation for pelvis marker (only x and y)
        pelvis_start = marker_start_global[:, 0, :2]  # Shape: (batch_size, 2)
        pelvis_end = marker_end_global[:, 0, :2]      # Shape: (batch_size, 2)
        frames = traj.size(1)
        interp_pelvis = torch.linspace(0, 1, frames).to(device).view(1, -1, 1) * (pelvis_end.unsqueeze(1) - pelvis_start.unsqueeze(1)) + pelvis_start.unsqueeze(1)

        # Forward pass
        pred, mu, logvar = model(interp_pelvis, marker_start_global)

        # Compute loss
        rec_loss = torch.nn.functional.mse_loss(pred, traj)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / traj.size(0)
        loss = rec_loss + kld_loss
        
        #Compute Velocity Loss
        traj_velocity = traj[:, 1:, :] - traj[:, :-1, :]
        pred_velocity = pred[:, 1:, :] - pred[:, :-1, :]
        velocity_loss = torch.nn.functional.mse_loss(pred_velocity, traj_velocity)

        loss = rec_loss + kld_loss + velocity_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log loss
        epoch_loss += loss.item()
        logger.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Iteration [{i+1}/{len(train_loader)}] Loss: {loss.item():.8f}")

    return epoch_loss / len(train_loader)

# Validation function
def validate(model, val_loader, device, save_dir, epoch, exp_name):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            traj, marker_start_global, marker_end_global = data
            traj = traj.to(device)
            marker_start_global = marker_start_global.to(device)
            marker_end_global = marker_end_global.to(device)

            # Linear interpolation for pelvis marker
            pelvis_start = marker_start_global[:, 0, :2]  # Shape: (batch_size, 2)
            pelvis_end = marker_end_global[:, 0, :2]      # Shape: (batch_size, 2)
            frames = traj.size(1)
            interp_pelvis = torch.linspace(0, 1, frames).to(device).view(1, -1, 1) * (pelvis_end.unsqueeze(1) - pelvis_start.unsqueeze(1)) + pelvis_start.unsqueeze(1)

            # Forward pass
            pred, mu, logvar = model(interp_pelvis, marker_start_global)

            # Compute loss
            rec_loss = torch.nn.functional.mse_loss(pred, traj)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / traj.size(0)

            # Compute velocity loss
            traj_velocity = traj[:, 1:, :] - traj[:, :-1, :]
            pred_velocity = pred[:, 1:, :] - pred[:, :-1, :]
            velocity_loss = torch.nn.functional.mse_loss(pred_velocity, traj_velocity)

            loss = rec_loss + kld_loss + velocity_loss

            val_loss += loss.item()

            # Save reconstructions
            if i == 0:  # Save only the first batch for visualization
                save_reconstruction_npz(traj.cpu().numpy(), pred.cpu().numpy(), save_dir, exp_name, epoch)

    return val_loss / len(val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoseBridge Training Script")
    parser.add_argument("--exp_name", required=True, help="Experiment name")
    args = parser.parse_args()

    SAVE_DIR = os.path.join('globaltrajectory_log', args.exp_name)
    os.makedirs(SAVE_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(SAVE_DIR, f"{args.exp_name}.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    # Prepare datasets
    train_dataset = PreprocessedMotionLoader(grab_dir, train_datasets)
    val_dataset = PreprocessedMotionLoader(grab_dir, test_datasets)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Define model
    nz = 32
    feature_dim = 64
    T = 32
    model = Traj_MLP_CVAE(nz, feature_dim, T).to(DEVICE)

    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(NUM_EPOCHS):
        logger.info(f"Starting Epoch {epoch + 1}/{NUM_EPOCHS}")

        # Training
        train_loss = train(model, optimizer, train_loader, epoch, logger, DEVICE)
        logger.info(f"Epoch {epoch + 1} Training Loss: {train_loss:.8f}")

        # Validation
        if (epoch + 1) % VALIDATE_EVERY == 0 or epoch + 1 == NUM_EPOCHS:
            val_loss = validate(model, val_loader, DEVICE, save_dir=SAVE_DIR, epoch=epoch, exp_name=args.exp_name)
            logger.info(f"Epoch {epoch + 1} Validation Loss: {val_loss:.8f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_model_epoch_{epoch + 1}.pth"))
                logger.info(f"Best model saved at epoch {epoch + 1} with Validation Loss: {val_loss:.8f}")

        # Step the scheduler
        scheduler.step()
