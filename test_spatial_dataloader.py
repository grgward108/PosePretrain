import os
import torch
import numpy as np
from SpatialTransformer.data.dataloader import FrameLoader
from torch.utils import data
import torch.multiprocessing as mp


def main():

    # Parameters for testing
    amass_datasets = ['HumanEva']
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
    for batch in dataloader:
        markers = batch['markers']        # [batch_size, n_markers, 3]
        part_labels = batch['part_labels']  # [batch_size, n_markers]
        print(f"Markers shape: {markers.shape}, dtype: {markers.dtype}")
        print(f"Part labels shape: {part_labels.shape}, dtype: {part_labels.dtype}")


        # Save the first sample in the batch
        first_sample = markers[0].cpu().numpy()  # Detach and move to CPU
        first_part_labels = part_labels[0].cpu().numpy()

        print("Saving the first sample to a file...")

        # Save to a .npz file
        output_file = "HumanEva_sample_markers.npz"
        np.savez_compressed(output_file, markers=first_sample, part_labels=first_part_labels)
        print(f"Markers and part labels saved to '{output_file}'.")

        break  # Exit after processing the first batch

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()