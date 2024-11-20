import os
import torch
from TemporalTransformer.data.dataloader import MotionLoader

# Parameters for testing
amass_datasets = ['HumanEva']  # Replace with your dataset names
amass_dir = 'dataset/AMASS'
smplx_model_path = 'body_utils/body_models'
clip_seconds = 2
clip_fps = 30
markers_type = 'f15_p22'  # Example markers type
mode = 'local_markers_3dv'  # Choose mode to test

# Initialize the MotionLoader
print("[INFO] Initializing MotionLoader...")
dataset = MotionLoader(
    clip_seconds=clip_seconds,
    clip_fps=clip_fps,
    normalize=True,
    split='train',
    markers_type=markers_type,
    mode=mode
)

# Read data from the specified datasets
print("[INFO] Loading data...")
dataset.read_data(amass_datasets, amass_dir)

# Create body representations
print("[INFO] Creating body representations...")
dataset.create_body_repr(with_hand=True, smplx_model_path=smplx_model_path)

# Print dataset size
print(f"[INFO] Total clips in dataset: {len(dataset)}")

# Create a DataLoader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# Inspect a batch of data
print("[INFO] Inspecting loaded data...")
for batch_idx, batch in enumerate(dataloader):
    print(f"Batch {batch_idx + 1}:")
    for data in batch:
        print(f"Data shape: {data.shape}")
        print(f"Data sample:\n{data[0]}")
    # Stop after one batch to avoid excessive printing
    break
