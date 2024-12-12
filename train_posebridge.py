import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import logging

from TemporalTransformer.models.models import TemporalTransformer
from LiftUpTransformer.models.models import LiftUpTransformer

# Combined Model
class EndToEndModel(nn.Module):
    def __init__(self, temporal_transformer, liftup_transformer):
        super(EndToEndModel, self).__init__()
        self.temporal_transformer = temporal_transformer
        self.liftup_transformer = liftup_transformer

    def forward(self, masked_joints):
        filled_joints = self.temporal_transformer(masked_joints)
        markers = self.liftup_transformer(filled_joints)
        return filled_joints, markers

# Training Function
def train_combined(model, optimizer, dataloader, criterion_temp, criterion_lift, epoch, logger, DEVICE):
    model.train()
    epoch_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
        masked_joints = batch['masked_joints'].to(DEVICE, dtype=torch.float32)
        original_joints = batch['original_joints'].to(DEVICE, dtype=torch.float32)
        original_markers = batch['original_markers'].to(DEVICE, dtype=torch.float32)

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
def validate_combined(model, dataloader, criterion_temp, criterion_lift, DEVICE):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            masked_joints = batch['masked_joints'].to(DEVICE, dtype=torch.float32)
            original_joints = batch['original_joints'].to(DEVICE, dtype=torch.float32)
            original_markers = batch['original_markers'].to(DEVICE, dtype=torch.float32)

            # Forward pass
            filled_joints, predicted_markers = model(masked_joints)

            # Compute losses
            loss_temp = criterion_temp(filled_joints, original_joints)
            loss_lift = criterion_lift(predicted_markers, original_markers)
            total_loss = loss_temp + loss_lift

            val_loss += total_loss.item()

    avg_loss = val_loss / len(dataloader)
    return avg_loss

# Main Training Script
def main(exp_name):
    # Configurations
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50

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

    wandb.init(project='PoseBridge', name=exp_name)

    # Dataset and DataLoader
    train_dataset = ...  # Define your dataset here
    val_dataset = ...    # Define your dataset here
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize Models
    temporal_transformer = TemporalTransformer(...).to(DEVICE)  # Pass appropriate parameters
    liftup_transformer = LiftUpTransformer(...).to(DEVICE)      # Pass appropriate parameters
    model = EndToEndModel(temporal_transformer, liftup_transformer).to(DEVICE)

    # Loss Functions and Optimizer
    criterion_temp = nn.MSELoss()
    criterion_lift = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        logger.info(f"Starting Epoch {epoch + 1}/{NUM_EPOCHS}")
        train_loss = train_combined(model, optimizer, train_loader, criterion_temp, criterion_lift, epoch, logger, DEVICE)
        val_loss = validate_combined(model, val_loader, criterion_temp, criterion_lift, DEVICE)

        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_model_epoch_{epoch + 1}.pth"))
            logger.info(f"Saved Best Model for Epoch {epoch + 1}")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoseBridge Training Script")
    parser.add_argument("--exp_name", required=True, help="Experiment name for logging and saving checkpoints")
    args = parser.parse_args()
    main(args.exp_name)
