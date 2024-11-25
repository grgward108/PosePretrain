import os
import json
import glob
import numpy as np
import torch
from torch.utils import data
import smplx
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FrameLoader(data.Dataset):
    def __init__(self, dataset_dir, smplx_model_path, markers_type='f15_p22', dataset_list=None, normalize=True):
        """
        Frame-based dataloader with part labels and marker positions.

        Args:
            dataset_dir (str): Path to the dataset directory.
            smplx_model_path (str): Path to the SMPL-X model directory.
            markers_type (str): Marker configuration ('f0_p0', 'f15_p5', etc.).
            normalize (bool): Whether to normalize marker positions.
        """
        self.dataset_dir = dataset_dir
        self.smplx_model_path = smplx_model_path
        self.markers_type = markers_type
        self.normalize = normalize
        self.data_list = []  # List to store frame data
        self.marker_indices = self._get_marker_indices(markers_type)

        # Load SMPL-X models for gender-specific generation
        self.smplx_model_male = smplx.create(
            smplx_model_path, model_type='smplx', gender='male', ext='npz',
            use_pca=False, flat_hand_mean=True, batch_size=1
        ).to(device)
        self.smplx_model_female = smplx.create(
            smplx_model_path, model_type='smplx', gender='female', ext='npz',
            use_pca=False, flat_hand_mean=True, batch_size=1
        ).to(device)

        self.body_part_groups = {
            'head_and_neck': [2819, 3076, 1795, 2311, 1043, 919, 8985, 1696, 1703, 9002, 8757, 2383, 2898, 3035, 2148, 9066, 8947, 2041, 2813],
            'trunk': [4391, 4297, 5615, 5944, 5532, 5533, 5678, 7145],
            'right_upper_limb': [7179, 7028, 7115, 7251, 7274, 7293, 6778, 7036],
            'left_upper_limb': [4509, 4245, 4379, 4515, 4538, 4557, 4039, 4258],
            'right_hand': [8001, 7781, 7750, 7978, 7756, 7884, 7500, 7419, 7984, 7633, 7602, 7667, 7860, 8082, 7351, 7611, 7867, 7423, 7357, 7396, 7443, 7446, 7536, 7589, 7618, 7625, 7692, 7706, 7730, 7748, 7789, 7847, 7858, 7924, 7931, 7976, 8039, 8050, 8087, 8122],
            'left_hand': [4897, 5250, 4931, 5124, 5346, 4615, 5321, 4875, 5131, 4683, 4686, 4748, 5268, 5045, 5014, 5242, 5020, 5149, 4628, 4641, 4660, 4690, 4691, 4710, 4750, 4885, 4957, 4970, 5001, 5012, 5082, 5111, 5179, 5193, 5229, 5296, 5306, 5315, 5353, 5387],
            'left_legs': [5857, 5893, 5899, 3479, 3781, 3638, 3705, 5761, 8852, 5726],
            'right_legs': [8551, 8587, 8593, 6352, 6539, 6401, 6466, 8455, 8634, 8421],
        }

        self._load_frames()

        if dataset_list:
            self.read_data(dataset_list, dataset_dir)

    def read_data(self, dataset_list, dataset_dir):
        """
        Read data from a list of datasets, calculate the total number of frames.

        Args:
            dataset_list (list): List of dataset names to load.
            dataset_dir (str): Path to the datasets directory.
        """
        total_frames = 0
        for dataset_name in tqdm(dataset_list, desc="Processing datasets"):
            dataset_path = os.path.join(dataset_dir, dataset_name)
            if not os.path.exists(dataset_path):
                print(f"[WARNING] Dataset '{dataset_name}' not found at {dataset_path}. Skipping.")
                continue
            
            # Glob all files for the dataset
            npz_files = glob.glob(os.path.join(dataset_path, '**', '*_poses.npz'), recursive=True)
            if not npz_files:
                print(f"[WARNING] No data found for dataset '{dataset_name}'.")
                continue

            for npz_file in npz_files:
                try:
                    data = np.load(npz_file)
                    num_frames = len(data['poses'])  # Count frames in this file
                    total_frames += num_frames

                    # Add frame data to the data list
                    for frame_idx in range(num_frames):
                        self.data_list.append({
                            'trans': data['trans'][frame_idx],  # Translation
                            'poses': data['poses'][frame_idx],  # Pose parameters
                            'betas': data['betas'],            # Shape parameters
                            'gender': str(data['gender'])      # Gender
                        })

                except Exception as e:
                    print(f"[ERROR] Failed to load file '{npz_file}': {e}")
                    continue

        self.n_samples = len(self.data_list)
        print(f"[INFO] Loaded {self.n_samples} frames from {len(dataset_list)} datasets.")
        print(f"[INFO] Total number of frames across all datasets: {total_frames}")


    def _get_marker_indices(self, markers_type):
        """
        Get marker indices based on the configuration.

        Args:
            markers_type (str): Marker configuration (e.g., 'f0_p0').

        Returns:
            list: List of marker indices.
        """
        f, p = markers_type.split('_')
        finger_n, palm_n = int(f[1:]), int(p[1:])
        with open('./body_utils/smplx_markerset.json') as f:
            markerset = json.load(f)['markersets']
            marker_indices = []
            for marker in markerset:
                if marker['type'] == 'finger' and finger_n == 0:
                    continue
                elif 'palm' in marker['type']:
                    if palm_n == 5 and marker['type'] == 'palm_5':
                        marker_indices += list(marker['indices'].values())
                    elif palm_n == 22 and marker['type'] == 'palm':
                        marker_indices += list(marker['indices'].values())
                    else:
                        continue
                else:
                    marker_indices += list(marker['indices'].values())
        print(f"Number of markers selected for '{markers_type}': {len(marker_indices)}")
        return marker_indices

    def _map_marker_to_part(self, marker_indices):
        """
        Map each marker to a specific body part.

        Args:
            marker_indices (list): List of marker indices.

        Returns:
            numpy.ndarray: Array of part labels for each marker.
        """
        part_labels = np.zeros(len(marker_indices))  # Initialize part labels as 0 (unassigned)
        for part_index, (part_name, part_indices) in enumerate(self.body_part_groups.items()):
            for marker_idx in part_indices:
                if marker_idx in marker_indices:
                    mapped_index = marker_indices.index(marker_idx)
                    part_labels[mapped_index] = part_index + 1
        return part_labels

    def _load_frames(self):
        """
        Load all frames from the dataset and store them as independent data points.
        """
        npz_files = glob.glob(os.path.join(self.dataset_dir, '**', '*_poses.npz'), recursive=True)
        for npz_file in tqdm(npz_files, desc="Loading frames"):
            data = np.load(npz_file)
            seq_len = len(data['poses'])
            for frame_idx in range(seq_len):
                self.data_list.append({
                    'trans': data['trans'][frame_idx],  # Translation
                    'poses': data['poses'][frame_idx],  # Pose parameters
                    'betas': data['betas'],            # Shape parameters
                    'gender': str(data['gender'])      # Gender
                })

    def _create_frame_repr(self, frame):
        """
        Generate body marker representation and part labels for a single frame.

        Args:
            frame (dict): Frame data containing pose, translation, and other metadata.

        Returns:
            tuple: Marker positions and part labels.
        """
        # Prepare body parameters for SMPL-X
        body_params = {
            'transl': torch.tensor(frame['trans'], device=device, dtype=torch.float32).unsqueeze(0),
            'global_orient': torch.tensor(frame['poses'][:3], device=device, dtype=torch.float32).unsqueeze(0),
            'body_pose': torch.tensor(frame['poses'][3:66], device=device, dtype=torch.float32).unsqueeze(0),
            'left_hand_pose': torch.tensor(frame['poses'][66:111], device=device, dtype=torch.float32).unsqueeze(0),
            'right_hand_pose': torch.tensor(frame['poses'][111:], device=device, dtype=torch.float32).unsqueeze(0),
            'betas': torch.tensor(frame['betas'][:10], device=device, dtype=torch.float32).unsqueeze(0)
        }

        # Select the appropriate SMPL-X model
        smplx_model = self.smplx_model_male if frame['gender'] == 'male' else self.smplx_model_female
        smplx_output = smplx_model(return_verts=True, **body_params)

        # Extract marker positions
        markers = smplx_output.vertices[:, self.marker_indices, :].squeeze(0)  # [n_markers, 3]

        # Normalize markers (optional)
        if self.normalize:
            markers -= markers.mean(dim=0, keepdim=True)

        # Generate part labels
        part_labels = torch.tensor(self._map_marker_to_part(self.marker_indices), dtype=torch.long)

        return markers, part_labels


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        frame = self.data_list[idx]
        markers, part_labels = self._create_frame_repr(frame)
        return {
            'markers': markers,           # [n_markers, 3]
            'part_labels': part_labels    # [n_markers]
        }