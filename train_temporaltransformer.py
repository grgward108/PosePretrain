import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from TemporalTransformer.models.models import PosePretrain  # Ensure your model is named `PosePretrain` in models.py
from TemporalTransformer.data.dataloader import MotionLoader  # Updated data loader with masking

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
MASK_PERCENTAGE = 0.15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLIP_SECONDS = 8
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
train_dataset.read_data(['HumanEva'], amass_dir='dataset/AMASS')
train_dataset.create_body_repr(with_hand=True, smplx_model_path=SMPLX_MODEL_PATH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Model
model = PosePretrain(
    dim_in=3,  # Input is 3D coordinates
    dim_out=3,  # Reconstruct to 3D coordinates
    dim_feat=512,
    dim_rep=512,
    depth=5,
    num_heads=8,
    mlp_ratio=4,
    num_joints=68,  # Adjust based on the number of markers
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
            masked_clip = masked_clip.to(DEVICE)  # Input with masking
            original_clip = original_clip.to(DEVICE)  # Ground truth

            # Forward pass
            outputs = model(masked_clip)

            # Compute loss only on masked positions
            mask = mask.unsqueeze(-1).expand_as(outputs).to(DEVICE)  # Match dimensions
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
