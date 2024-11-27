import torch
from TemporalTransformer.models.models import TemporalTransformer

# Define the model
model = TemporalTransformer(
    dim_in=3,
    dim_out=3,
    dim_feat=128,
    depth=5,
    num_heads=8,
    num_joints=25,  # Replace with the actual number of joints in your data
    maxlen=61,  # Replace with the actual sequence length
)

# Load the checkpoint
checkpoint_path = '/home/edwarde/PosePretrain/logs/TemporalTransformer/first_concept/epoch_50.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # Load to CPU for inspection
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded model from {checkpoint_path}")

# Count the number of learnable parameters
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of learnable parameters: {num_params}")
