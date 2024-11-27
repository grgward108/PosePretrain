import os
import torch
import numpy as np
from datetime import datetime  # For date formatting
from SpatialTransformer.data.dataloader import FrameLoader
from torch.utils import data
import torch.multiprocessing as mp


def main():
    # Parameters for testing
    amass_datasets = ['ACCAD']
    amass_dir = '../../../data/edwarde/dataset/AMASS'
    smplx_model_path = 'body_utils/body_models'
    markers_type = 'f15_p22'
    normalize = True

    # Initialize the FrameLoader
    print("[INFO] Initializing FrameLoader...")
    dataset = FrameLoader(
        dataset_dir=amass_dir,
        smplx_model_path=smplx_model_path,
        markers_type=markers_type,
        normalize=normalize,
        dataset_list=amass_datasets
    )

    # Use the custom collate function
    dataloader = data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=4  # Adjust as needed
    )

    # Iterate over the dataloader
    for batch_idx, batch in enumerate(dataloader):
        markers = batch['markers']        # [batch_size, n_markers, 3]
        part_labels = batch['part_labels']  # [batch_size, n_markers]
        print(f"Markers shape: {markers.shape}, dtype: {markers.dtype}")
        print(f"Part labels shape: {part_labels.shape}, dtype: {part_labels.dtype}")

        # Save the entire first batch
        print("Saving the entire first batch to a file...")

        # Convert to numpy
        markers_np = markers.cpu().numpy()
        part_labels_np = part_labels.cpu().numpy()

        # Generate file name with dataset name and date
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_file = f"{amass_datasets[0]}_first_batch_markers_and_labels_{date_str}.npz"

        # Save to a .npz file
        np.savez_compressed(output_file, markers=markers_np, part_labels=part_labels_np)
        print(f"Markers and part labels for the first batch saved to '{output_file}'.")

        break  # Exit after processing the first batch


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
