import os
import numpy as np
from torch.utils.data import Dataset
import torch


class PreprocessedMotionLoader(Dataset):
    def __init__(self, data_dir, datasets):
        """
        Initializes the dataset by listing all preprocessed `.npz` files in the specified datasets.
        Args:
            data_dir (str): Path to the base directory containing dataset folders.
            datasets (list of str): List of dataset folder names to include (e.g., ['s1', 's2']).
        """
        self.file_list = []
        for dataset in datasets:
            folder_path = os.path.join(data_dir, dataset)
            if os.path.exists(folder_path):
                self.file_list.extend(
                    [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.endswith('.npz')]
                )
            else:
                print(f"Warning: Folder {folder_path} does not exist.")
        self.file_list.sort()  # Ensure consistent order

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Load a single preprocessed `.npz` file and return its contents.
        Args:
            idx (int): Index of the sample.
        Returns:
            A tuple containing all data required for training or validation.
        """
        file_path = self.file_list[idx]
        data = np.load(file_path)
        print(data.keys)

        clip_img_joints = torch.tensor(data['clip_img_joints'], dtype=torch.float32)
        clip_img_markers = torch.tensor(data['clip_img_markers'], dtype=torch.float32)
        slerp_img = torch.tensor(data['slerp_img'], dtype=torch.float32)
        traj = torch.tensor(data['traj'], dtype=torch.float32)
        smplx_beta = torch.tensor(data['smplx_beta'], dtype=torch.float32)
        gender = int(data['gender'])  # Assuming gender is stored as an integer
        rot_0_pivot = torch.tensor(data['rot_0_pivot'], dtype=torch.float32)
        transf_matrix_smplx = torch.tensor(data['transf_matrix_smplx'], dtype=torch.float32)
        smplx_params_gt = torch.tensor(data['smplx_params_gt'], dtype=torch.float32)
        marker_start = torch.tensor(data['marker_start'], dtype=torch.float32)
        marker_end = torch.tensor(data['marker_end'], dtype=torch.float32)
        joint_start = torch.tensor(data['joint_start'], dtype=torch.float32)
        joint_end = torch.tensor(data['joint_end'], dtype=torch.float32)
        marker_start_global = torch.tensor(data['marker_start_global'], dtype=torch.float32) # Added global start marker
        marker_end_global  = torch.tensor(data['marker_end_global'], dtype=torch.float32) # Added global start marker

        return (
            traj,
            marker_start_global,  # Added global start marker
            marker_end_global     # Added global end marker
        )
