import os
import torch
import numpy as np
from TemporalTransformer.data.dataloader import MotionLoader

# Parameters for testing
amass_datasets = ['HumanEva']
amass_dir = '../../../data/edwarde/dataset/AMASS'
smplx_model_path = 'body_utils/body_models'
clip_seconds = 2
clip_fps = 30
markers_type = 'f15_p22'  # Example markers type
mode = 'local_joints_3dv'  # Choose mode to test
output_folder = "./temporal_datadump"  # Directory to save output

# Ensure the output directory exists
os.makedirs(output_folder, exist_ok=True)

# Initialize the MotionLoader
print("[INFO] Initializing MotionLoader...")
dataset = MotionLoader(
    clip_seconds=clip_seconds,
    clip_fps=clip_fps,
    normalize=True,
    split='train',
    markers_type=markers_type,
    mode=mode,
    mask_ratio=0.15
)

# Read data from the specified datasets
print("[INFO] Loading data...")
dataset.read_data(amass_datasets, amass_dir)

# Create body representations
print("[INFO] Creating body representations...")
dataset.create_body_repr(with_hand=True, smplx_model_path=smplx_model_path)

masked_clip, mask, original_clip = dataset[7]  # Uses the __getitem__ method

# Ensure data is on the CPU and converted to NumPy arrays
if isinstance(masked_clip, torch.Tensor):
    masked_clip = masked_clip.cpu().detach().numpy()
if isinstance(mask, torch.Tensor):
    mask = mask.cpu().detach().numpy()
if isinstance(original_clip, torch.Tensor):
    original_clip = original_clip.cpu().detach().numpy()


# Save both to a compressed file
output_file = os.path.join(output_folder, f"{amass_datasets[0]}_seventh_batch_Test_marker_with_masking.npz")
np.savez_compressed(
    output_file,
    masked_clip=masked_clip,
    mask=mask,
    original_clip=original_clip
)

print(f"[INFO] Data for index 7 saved successfully to {output_file}")
print(f"[INFO] Masked Clip Shape: {masked_clip.shape}")
print(f"[INFO] Mask Shape: {mask.shape}")
print(f"[INFO] Original Clip Shape: {original_clip.shape}")
