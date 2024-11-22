import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from TemporalTransformer.models.models import TemporalTransformer
from TemporalTransformer.data.dataloader import MotionLoader

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
MASK_PERCENTAGE = 0.15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLIP_SECONDS = 2
CLIP_FPS = 30
MARKERS_TYPE = 'f15_p5'  # Example: 15 finger and 5 palm markers
MODE = 'local_markers_3dv'
SMPLX_MODEL_PATH = 'body_utils/body_models'

# Dataset and DataLoader
train_dataset = MotionLoader(
    clip_seconds=CLIP_SECONDS,
    clip_fps=CLIP_FPS,
    normalize=True,
    split='train',
    markers_type=MARKERS_TYPE,
    mode=MODE,
    mask_percentage=MASK_PERCENTAGE,
    log_dir='./logs',
)

print(f"Initialized dataset with {CLIP_SECONDS * CLIP_FPS} frames per clip and markers type {MARKERS_TYPE}.")

train_dataset.read_data(['HumanEva'], amass_dir='../../../data/edwarde/dataset/AMASS')
print(f"Loaded {train_dataset.n_samples} sub-clips from dataset.")

train_dataset.create_body_repr(smplx_model_path=SMPLX_MODEL_PATH)

example_clip, example_mask, original_clip = train_dataset[0]
print(f"Example clip shape: {example_clip.shape}, Example mask shape: {example_mask.shape}, Original clip shape: {original_clip.shape}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Debugging: Check first batch from DataLoader
for batch in train_loader:
    masked_clip, mask, original_clip = batch
    print(f"Batch shapes - Masked clip: {masked_clip.shape}, Mask: {mask.shape}, Original clip: {original_clip.shape}")
    break

# Model
model = TemporalTransformer(
    dim_in=3,  # Input is 3D coordinates
    dim_out=3,  # Reconstruct to 3D coordinates
    dim_feat=512,
    depth=5,
    num_heads=8,
    num_joints=example_clip.shape[1],  # Number of joints/markers
    maxlen=CLIP_SECONDS * CLIP_FPS,
).to(DEVICE)

# Loss and Optimizer
criterion = nn.MSELoss()  # Reconstruction loss
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# Training Loop
def train():
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for masked_clip, mask, original_clip in progress_bar:
            # Move data to device
            masked_clip = masked_clip.to(DEVICE)  # (B, T, J, C)
            original_clip = original_clip.to(DEVICE)  # (B, T, J, C)

            # Forward pass
            outputs = model(masked_clip)  # (B, T, J, C)

            # Masking to compute loss only on masked frames
            mask = mask.unsqueeze(-1).unsqueeze(-1).expand_as(outputs).to(DEVICE)  # Expand mask to (B, T, J, C)
            loss = criterion(outputs[mask], original_clip[mask])

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss / len(train_loader):.4f}")

# Run training
if __name__ == "__main__":
    train()
