import torch
from GlobalTrajectory.data.dataloader import PreprocessedMotionLoader
from GlobalTrajectory.models.models import Traj_MLP
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
grab_dir = '../../../data/edwarde/dataset/include_global_traj'
train_datasets = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
test_datasets = ['s9', 's10']

# Function to save reconstructions
def save_reconstruction_npz(traj_gt, predicted_traj, interp_pelvis, save_dir, exp_name, epoch):
    save_path = os.path.join(save_dir, exp_name)
    os.makedirs(save_path, exist_ok=True)
    np.savez_compressed(os.path.join(save_path, f"globaltraj_reconstruction_{exp_name}_epoch_{epoch}.npz"),
                        traj_gt=traj_gt, 
                        predicted_traj=predicted_traj,
                        interp_pelvis=interp_pelvis)

def train(model, optimizer, train_loader, epoch, logger, device):
    model.train()
    epoch_loss = 0
    total_batches = len(train_loader)
    for i, data in enumerate(train_loader):
        traj, marker_start_global, marker_end_global = data
        traj = traj[:, :2, :].to(device)  # Take only x and y, shape [batch_size, 2, frames]
        marker_start_global = marker_start_global.to(device)
        marker_end_global = marker_end_global.to(device)

        pelvis_start = marker_start_global[:, 0, :2]  # Shape: (batch_size, 2)
        pelvis_end = marker_end_global[:, 0, :2]      # Shape: (batch_size, 2)

        # Subtract pelvis_start to normalize to zero
        pelvis_end_normalized = pelvis_end - pelvis_start  # Shape: (batch_size, 2)

        # Number of frames
        frames = traj.size(1)

        # Interpolate pelvis trajectory
        frames = 62  # Number of frames (adjust if needed)
        interp_pelvis = torch.linspace(0, 1, frames).to(device).view(1, -1, 1) * (pelvis_end.unsqueeze(1) - pelvis_start.unsqueeze(1)) + pelvis_start.unsqueeze(1)
        interp_pelvis = interp_pelvis.view(interp_pelvis.size(0), -1)  # Flatten to (batch_size, 124)
        
        pred = model(interp_pelvis)  # Input the flattened x and y components

        # Compute loss
        rec_loss = torch.nn.functional.mse_loss(pred, traj.reshape(traj.size(0), -1))

        # Compute velocity loss
        traj_velocity = traj[:, :, 1:] - traj[:, :, :-1]  # [batch_size, 2, frames-1]
        pred_velocity = pred.reshape(traj.size(0), frames, 2)[:, 1:, :] - pred.reshape(traj.size(0), frames, 2)[:, :-1, :]  # [batch_size, frames-1, 2]
        velocity_loss = torch.nn.functional.mse_loss(pred_velocity, traj_velocity.permute(0, 2, 1))

        loss = rec_loss + velocity_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        epoch_loss += loss.item()

    # Log the average training loss per epoch
    avg_train_loss = epoch_loss / total_batches
    logger.info(f"Epoch {epoch + 1}: Training Loss = {avg_train_loss:.8f}")
    return avg_train_loss

def validate(model, val_loader, device, save_dir, epoch, exp_name):
    model.eval()
    val_loss = 0
    total_batches = len(val_loader)
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            traj, joint_start, joint_end = data  # Assuming joint_start and joint_end are part of your data
            traj = traj[:, :2, :].to(device)  # Take only x and y, shape [batch_size, 2, frames]
            joint_start = joint_start.to(device)
            joint_end = joint_end.to(device)
            
            print("Joint Start:", joint_start)
            print("Joint End:", joint_end)


            # Prepare joint_sr_input
            joint_sr_input, _, _, _ = prepare_traj_input_without_stats(joint_start, joint_end, device)

            # Take only the x and y components (first two dimensions)
            joint_sr_input_xy = joint_sr_input[:, :2, :]  # [batch_size, 2, frames]

            # Flatten for the model input
            model_input = joint_sr_input_xy.reshape(joint_sr_input_xy.size(0), -1)  # [batch_size, 2 * frames]

            # Forward pass
            pred = model(model_input)  # Input the flattened x and y components

            # Compute loss
            rec_loss = torch.nn.functional.mse_loss(pred, traj.reshape(traj.size(0), -1))

            # Compute velocity loss
            traj_velocity = traj[:, :, 1:] - traj[:, :, :-1]  # [batch_size, 2, frames-1]
            pred_velocity = pred.reshape(traj.size(0), traj.shape[2], 2)[:, 1:, :] - pred.reshape(traj.size(0), traj.shape[2], 2)[:, :-1, :]  # [batch_size, frames-1, 2]
            velocity_loss = torch.nn.functional.mse_loss(pred_velocity, traj_velocity.permute(0, 2, 1))

            # Combine losses
            loss = rec_loss + velocity_loss

            val_loss += loss.item()

            # Save reconstructions (only once per epoch)
            if i == 0:  # Save only the first batch for visualization
                save_reconstruction_npz(
                    traj.cpu().numpy(),
                    pred.cpu().numpy(),
                    joint_sr_input_xy.cpu().numpy(),
                    save_dir,
                    exp_name,
                    epoch
                )

    # Log the average validation loss per epoch
    avg_val_loss = val_loss / total_batches
    print(f"Epoch {epoch + 1}: Validation Loss = {avg_val_loss:.8f}")
    return avg_val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trajectory Training Script")
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
    input_dim = 124  # 62 frames * 2 (x, y)
    hidden_dim = 256
    output_dim = 124  # Corrected trajectory (same shape as input)
    model = Traj_MLP(input_dim, hidden_dim, output_dim).to(DEVICE)

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
