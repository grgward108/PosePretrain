import os
import torch
import numpy as np
from TemporalTransformer.data.dataloader import MotionLoader

# Parameters for testing
amass_datasets = ['CMU']
amass_dir = '../../../data/edwarde/dataset/AMASS'
smplx_model_path = 'body_utils/body_models'
clip_seconds = 2
clip_fps = 30
markers_type = 'f15_p22'  # Example markers type
mode = 'local_markers_3dv'  # Choose mode to test
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
    mode=mode
)

# Read data from the specified datasets
print("[INFO] Loading data...")
dataset.read_data(amass_datasets, amass_dir)

# Create body representations
print("[INFO] Creating body representations...")
dataset.create_body_repr(with_hand=True, smplx_model_path=smplx_model_path)

# Save only the first batch of processed data and print its shape
first_batch_data = dataset.clip_img_list[0]  # Get the first batch
print(f"[INFO] Shape of the first batch: {np.array(first_batch_data).shape}")

first_batch_file = os.path.join(output_folder, f"{amass_datasets[0]}_first_batch_Test_marker.npz")
np.savez_compressed(first_batch_file, data=first_batch_data)
print(f"[INFO] First batch saved to {first_batch_file}")
