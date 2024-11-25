import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from SpatialTransformer.models.models import SpatialTransformer
from SpatialTransformer.data.dataloader import FrameLoader
import torch.multiprocessing as mp

# Hyperparameters
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 6
N_PARTS = 9  # Number of body parts
N_MARKERS = 143  # Number of markers
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_DIR = './checkpoints'

# Paths and Dataset Parameters
DATASET_DIR = '../../../data/edwarde/dataset/AMASS'
SMPLX_MODEL_PATH = 'body_utils/body_models'
MARKERS_TYPE = 'f15_p22'
NORMALIZE = True

TRAIN_DATASET = ['HumanEva', 'ACCAD', 'CMU', 'DanceDB', 'Eyes_Japan_Dataset', 'GRAB']
VAL_DATASET = ['HUMAN4D', 'KIT']

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Define Loss Function
criterion = nn.MSELoss()


def train(model, dataloader, optimizer, epoch):
    """Training loop for one epoch."""
    model.train()
    epoch_loss = 0.0

    for batch in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
        markers = batch['markers'].to(DEVICE)  # Shape: [bs, n_markers, 3]
        part_labels = batch['part_labels'].to(DEVICE)  # Shape: [bs, n_markers]

        # Forward pass
        optimizer.zero_grad()
        reconstructed_markers = model(markers, part_labels)

        # Compute Loss
        loss = criterion(reconstructed_markers, markers)

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch + 1} Training Loss: {avg_loss:.4f}")
    return avg_loss


def validate(model, dataloader, epoch):
    """Validation loop."""
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Validation Epoch {epoch + 1}"):
            markers = batch['markers'].to(DEVICE)  # Shape: [bs, n_markers, 3]
            part_labels = batch['part_labels'].to(DEVICE)  # Shape: [bs, n_markers]

            # Forward pass
            reconstructed_markers = model(markers, part_labels)

            # Compute Loss
            loss = criterion(reconstructed_markers, markers)
            epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch + 1} Validation Loss: {avg_loss:.4f}")
    return avg_loss


def main():
    # Initialize Dataset and Dataloader
    print("[INFO] Initializing FrameLoader...")

    train_dataset = FrameLoader(
        dataset_dir=DATASET_DIR,
        smplx_model_path=SMPLX_MODEL_PATH,
        markers_type=MARKERS_TYPE,
        normalize=NORMALIZE,
        dataset_list=TRAIN_DATASET
    )

    val_dataset = FrameLoader(
        dataset_dir=DATASET_DIR,
        smplx_model_path=SMPLX_MODEL_PATH,
        markers_type=MARKERS_TYPE,
        normalize=NORMALIZE,
        dataset_list=VAL_DATASET
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=4  # Adjust as needed
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=4  # Adjust as needed
    )

    # Initialize Model and Optimizer
    print("[INFO] Initializing Model...")
    model = SpatialTransformer(
        n_markers=N_MARKERS,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        n_parts=N_PARTS
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        print(f"\n[INFO] Starting Epoch {epoch + 1}/{NUM_EPOCHS}...")
        train_loss = train(model, train_loader, optimizer, epoch)

        # Perform validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            val_loss = validate(model, val_loader, epoch)

            # Save Best Model if Validation Improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(SAVE_DIR, f"best_model_epoch_{epoch + 1}.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"[INFO] Saved Best Model to {checkpoint_path}.")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(SAVE_DIR, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[INFO] Checkpoint saved at {checkpoint_path}.")

    print("[INFO] Training Complete.")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
