import torch
from models.models import SpatialTransformer

# Define the model
model = SpatialTransformer(
    n_markers=143,
    embed_dim=64,
    num_heads=4,
    num_layers=6,
    dropout=0.1,
    n_parts=9
)

# Load the checkpoint
checkpoint_path = '../spatial_log/exp02/ckpt/best_model_epoch_3.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Load to CPU for inspection
model.load_state_dict(checkpoint)  # Use checkpoint directly
print(f"Loaded model from {checkpoint_path}")

# Count the number of learnable parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of learnable parameters: {num_params}")
