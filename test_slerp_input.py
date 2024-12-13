import os
import torch
import numpy as np
from TemporalTransformer.data.input_slerp import MotionLoader
from torch.utils.data import DataLoader

# Parameters for testing
amass_datasets = ['s1']
amass_dir = '../../../../data/edwarde/dataset/grab/GraspMotion'
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
    mask_ratio=0.15,
    smplx_model_path=smplx_model_path
)

# Read data from the specified datasets
print("[INFO] Loading data...")
dataset.read_data(amass_datasets, amass_dir)

print(f"[DEBUG] Total subclips generated: {len(dataset.data_metadata_list)}")
if len(dataset.data_metadata_list) == 0:
    print("[ERROR] No subclips generated! Check your dataset paths or the structure of your .npz files.")
else:
    print("[INFO] Subclip metadata loaded successfully.")


# Create body representations
print("[INFO] Creating body representations...")
batch_size = 1  # Set batch size to 1 for easier visualization
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Fetch the first batch
print("[INFO] Fetching the first batch...")
first_batch = next(iter(data_loader))
interpolated_clip, original_clip = first_batch  # Unpack the returned values

