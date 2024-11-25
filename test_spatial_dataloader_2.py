import os
import torch
import numpy as np
from SpatialTransformer.data.original_dataloader import FrameLoader
from torch.utils import data

# Parameters for testing
amass_datasets = ['HumanEva', 'CMU']
amass_dir = '../../../data/edwarde/dataset/AMASS'
smplx_model_path = 'body_utils/body_models'
markers_type = 'f15_p22'  # Example markers type
normalize = True

# Initialize the MotionLoader
print("[INFO] Initializing MotionLoader...")
dataset = FrameLoader(
    dataset_dir=amass_dir,
    smplx_model_path=smplx_model_path,
    markers_type=markers_type,
    normalize=normalize,
    dataset_list=amass_datasets  # Data will be loaded automatically
)

dataloader = data.DataLoader(dataset, batch_size=8, shuffle=True)


# Iterate over the dataloader
for batch in dataloader:
    markers = batch['markers']  # [batch_size, n_markers, 3]
    part_labels = batch['part_labels']  # [batch_size, n_markers]
    print(markers.shape, part_labels.shape)

    # Save the first sample in the batch
    first_sample = markers[0].detach().cpu().numpy()  # Detach and convert to NumPy
    first_part_labels = part_labels[0].detach().cpu().numpy()  # Detach part labels

    print("Saving the first sample to a file...")

    # Save to a .npz file
    output_file = "first_sample_markers.npz"
    np.savez_compressed(output_file, markers=first_sample, part_labels=first_part_labels)
    print(f"Markers and part labels saved to '{output_file}'.")

    break  # Exit after processing the first batch