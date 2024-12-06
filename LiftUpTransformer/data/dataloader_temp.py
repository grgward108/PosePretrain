# load the data of 17 joints and 143 joints of the same dataset
import os
import json
import glob
import numpy as np
import torch
from torch.utils import data
import smplx
from tqdm import tqdm
from human_body_prior.body_model.body_model import BodyModel
import logging
logging.basicConfig(level=logging.INFO)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FrameLoader(data.Dataset):
    def __init__(self, dataset_dir, smplx_model_path, markers_type='f15_p22',
                 dataset_list=None, normalize=True, apply_masking=False, masking_ratio=0.15,
                 regressor_path='./J_regressor_h36m.npy'):
        """
        Frame-based dataloader with part labels and marker positions.
        Args:
            dataset_dir (str): Path to the dataset directory.
            smplx_model_path (str): Path to the SMPL-X model directory.
            markers_type (str): Marker configuration ('f0_p0', 'f15_p5', etc.).
            normalize (bool): Whether to normalize marker positions.
        """
        self.dataset_dir = dataset_dir
        self.markers_type = markers_type
        self.smplx_model_path = smplx_model_path
        self.normalize = normalize
        self.data_list = []  # List to store frame data
        self.marker_indices = self._get_marker_indices(markers_type)

        self.smplx_model_male = smplx.create(
            smplx_model_path, model_type='smplx', gender='male', ext='npz',
            use_pca=False, flat_hand_mean=True, use_face_contour=False,  # Disable face-related vertices
            create_body_pose=True, create_global_orient=True, create_betas=True, create_transl=True
        ).to(device)

        self.smplx_model_female = smplx.create(
            smplx_model_path, model_type='smplx', gender='female', ext='npz',
            use_pca=False, flat_hand_mean=True, use_face_contour=False,  # Disable face-related vertices
            create_body_pose=True, create_global_orient=True, create_betas=True, create_transl=True
        ).to(device)


        logging.info("Loading SMPL models...")

                # Path to SMPL-H model and DMPLs
        male_bm_fname = "body_utils/body_models/smplh/male/model.npz"
        female_bm_fname = "body_utils/body_models/smplh/female/model.npz"

        logging.info("Loading SMPL-H BodyModel...")
        self.male_body_model = BodyModel(
            bm_path=male_bm_fname,
            model_type="smplh",
            num_betas=16,
            batch_size=1
        ).to(device)


        self.female_body_model = BodyModel(
            bm_path=female_bm_fname,
            model_type="smplh",
            num_betas=16,
            batch_size=1
        ).to(device)



        logging.info("SMPL-H BodyModel loaded successfully.")

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

        # Compute part labels once and store them
        self.part_labels = torch.tensor(self._map_marker_to_part(self.marker_indices), dtype=torch.long)

        self.samples = self._build_index(dataset_list)

        self.apply_masking = apply_masking
        self.masking_ratio = masking_ratio

        self.j_regressor = torch.tensor(np.load(regressor_path), dtype=torch.float32).to(device)
        logging.info(f"Loaded J_regressor with shape {self.j_regressor.shape}")

        print(f"[INFO] Loaded J_regressor with shape {self.j_regressor.shape}")

    def _build_index(self, dataset_list=None):
        """
        Build an index of all frames across datasets, without loading them into memory.
        """
        samples = []    
        total_frames = 0  # Keep track of the total number of frames
        if dataset_list:
            datasets_to_load = dataset_list
        else:
            datasets_to_load = [name for name in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, name))]

        for dataset_name in tqdm(datasets_to_load, desc="Indexing datasets"):
            dataset_path = os.path.join(self.dataset_dir, dataset_name)
            if not os.path.exists(dataset_path):
                print(f"[WARNING] Dataset '{dataset_name}' not found at {dataset_path}. Skipping.")
                continue

            npz_files = glob.glob(os.path.join(dataset_path, '**', '*_poses.npz'), recursive=True)
            if not npz_files:
                print(f"[WARNING] No data found for dataset '{dataset_name}'.")
                continue

            for npz_file in npz_files:
                try:
                    data = np.load(npz_file)
                    num_frames = len(data['poses'])
                    total_frames += num_frames
                    gender = str(data['gender'])
                    for frame_idx in range(num_frames):
                        samples.append({
                            'file': npz_file,
                            'frame_idx': frame_idx,
                            'gender': gender
                        })
                except Exception as e:
                    print(f"[ERROR] Failed to load file '{npz_file}': {e}")
                    continue
        print(f"[INFO] Indexed {len(samples)} frames from {len(datasets_to_load)} datasets.")
        print(f"[INFO] Total number of frames across all datasets: {total_frames}")
        return samples

    def collate_fn(self, batch):
        """
        Custom collate function for DataLoader.
        """
        frames = [sample['frame'] for sample in batch]
        genders = [sample['gender'] for sample in batch]

        markers, joints = self.batch_process_frames(frames, genders)

        return {
            'markers': markers,  # [batch_size, n_markers, 3]
            'joints': joints,    # [batch_size, 17, 3]
        }



    def batch_process_frames(self, frames, genders):
        """
        Processes a batch of frames to extract markers from SMPL-X and joints from SMPL-H models.

        Args:
            frames (list): List of frame dictionaries containing pose data.
            genders (list): List of genders ('male' or 'female') for each frame.

        Returns:
            torch.Tensor: Processed markers [batch_size, n_markers, 3].
            torch.Tensor: Processed joints [batch_size, n_joints, 3].
        """
        # Filter valid frames
        valid_frames = []
        valid_genders = []
        for i, (frame, gender) in enumerate(zip(frames, genders)):
            if len(frame['poses']) >= 111:  # Minimum pose length check
                valid_frames.append(frame)
                valid_genders.append(gender)
            else:
                print(f"[INFO] Skipping frame {i} due to insufficient pose length.")

        if not valid_frames:
            raise ValueError("No valid frames in the batch.")

        # Update batch size
        batch_size = len(valid_frames)

        # Extract parameters from valid frames
        poses = np.array([frame['poses'] for frame in valid_frames])
        transl = np.array([frame['trans'] for frame in valid_frames])
        betas = np.array([frame['betas'][:10] for frame in valid_frames])  # Truncate betas for SMPL-X

        # Extract pose components
        global_orient = poses[:, :3]
        body_pose = poses[:, 3:66]
        left_hand_pose = poses[:, 66:111]
        right_hand_pose = poses[:, 111:156]
        jaw_pose = poses[:, 156:159] if poses.shape[1] >= 159 else np.zeros((poses.shape[0], 3))
        leye_pose = poses[:, 159:162] if poses.shape[1] >= 162 else np.zeros((poses.shape[0], 3))
        reye_pose = poses[:, 162:165] if poses.shape[1] >= 165 else np.zeros((poses.shape[0], 3))
        expression = np.zeros((poses.shape[0], 10), dtype=np.float32)  # [batch_size, 10]

        # Prepare body parameters for SMPL-X
        body_params_smplx = {
            'transl': torch.tensor(transl, device=device, dtype=torch.float32),
            'global_orient': torch.tensor(global_orient, device=device, dtype=torch.float32),
            'body_pose': torch.tensor(body_pose, device=device, dtype=torch.float32),
            'left_hand_pose': torch.tensor(left_hand_pose, device=device, dtype=torch.float32),
            'right_hand_pose': torch.tensor(right_hand_pose, device=device, dtype=torch.float32),
            'jaw_pose': torch.tensor(jaw_pose, device=device, dtype=torch.float32),
            'leye_pose': torch.tensor(leye_pose, device=device, dtype=torch.float32),
            'reye_pose': torch.tensor(reye_pose, device=device, dtype=torch.float32),
            'betas': torch.tensor(betas, device=device, dtype=torch.float32),
            'expression': torch.tensor(expression, device=device, dtype=torch.float32)
        }

        # Separate indices by gender
        male_indices = [i for i, g in enumerate(valid_genders) if g == 'male']
        female_indices = [i for i, g in enumerate(valid_genders) if g == 'female']

        # Initialize lists for markers and joints
        markers_list = [None] * batch_size
        joints_list = [None] * batch_size

        # Process SMPL-X for markers
        for indices, smplx_model in [(male_indices, self.smplx_model_male), (female_indices, self.smplx_model_female)]:
            if indices:
                smplx_params = {k: v[indices] for k, v in body_params_smplx.items()}
                with torch.no_grad():
                    smplx_output = smplx_model(return_verts=True, **smplx_params)
                markers = smplx_output.vertices[:, self.marker_indices, :]  # [batch_size_gender, n_markers, 3]
                for idx, orig_idx in enumerate(indices):
                    markers_list[orig_idx] = markers[idx]

        # Process SMPL-H for joints
        for i, (frame, gender) in enumerate(zip(valid_frames, valid_genders)):
            body_model = self.male_body_model if gender == 'male' else self.female_body_model
            body_params_smplh = {
                'root_orient': torch.tensor(frame['poses'][:3], dtype=torch.float32).unsqueeze(0).to(device),
                'pose_body': torch.tensor(frame['poses'][3:66], dtype=torch.float32).unsqueeze(0).to(device),
                'pose_hand': torch.tensor(frame['poses'][66:156], dtype=torch.float32).unsqueeze(0).to(device),
                'trans': torch.tensor(frame['trans'], dtype=torch.float32).unsqueeze(0).to(device),
                'betas': torch.tensor(frame['betas'][:16], dtype=torch.float32).unsqueeze(0).to(device)
            }
            with torch.no_grad():
                body_output_h = body_model(**body_params_smplh)
                vertices_h = body_output_h.v[0]
                joints_h = torch.matmul(self.j_regressor, vertices_h)  # Map to joints
            joints_list[i] = joints_h

        # Stack results in the original order
        markers = torch.stack(markers_list, dim=0)  # [batch_size, n_markers, 3]
        joints = torch.stack(joints_list, dim=0)  # [batch_size, n_joints, 3]

        # Normalize if required
        if self.normalize:
            markers_mean = markers.mean(dim=1, keepdim=True)
            joints_mean = joints.mean(dim=1, keepdim=True)
            markers -= markers_mean
            joints -= joints_mean

        logging.info(f"Processed batch with markers shape {markers.shape} and joints shape {joints.shape}")
        return markers, joints





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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        npz_file = sample['file']
        frame_idx = sample['frame_idx']

        # Load only the required frame from the file
        data = np.load(npz_file)
        frame = {
            'trans': data['trans'][frame_idx].astype(np.float32),
            'poses': data['poses'][frame_idx].astype(np.float32),
            'betas': data['betas'].astype(np.float32)
        }
        gender = sample['gender']

        # Handle missing facial and eye pose parameters
        pose = frame['poses']
        if len(pose) == 156:  # Pose vector length for SMPL-H
            # Append zeros for the missing 9 parameters
            zeros_9 = np.zeros(9, dtype=pose.dtype)
            pose = np.concatenate([pose, zeros_9])
            frame['poses'] = pose

        return {
            'frame': frame,
            'gender': gender
        }