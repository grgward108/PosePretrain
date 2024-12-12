import glob
import json
import os
import sys
import numpy as np
import scipy.ndimage.filters as filters
import smplx
import torch
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.getcwd())
from human_body_prior.tools.model_loader import load_vposer
from utils.como.como_utils import *
from utils.Pivots import Pivots
from utils.Quaternions import Quaternions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MotionLoader(data.Dataset):
    def __init__(self, clip_seconds=8, clip_fps=30, normalize=False, split='train', markers_type=None, mode=None,
                 is_debug=False, mask_ratio=0.0, log_dir='', smplx_model_path=''):
        """
        Lazy loading version of the dataset.
        Now uses joints, not markers, and removes the last frame similar to the original code.
        """
        self.clip_seconds = clip_seconds
        self.clip_len = clip_seconds * clip_fps + 2  # T+2 frames for each clip
        self.data_metadata_list = []  # store metadata for lazy loading
        self.normalize = normalize
        self.clip_fps = clip_fps
        self.split = split  # train/test
        self.mode = mode
        self.is_debug = is_debug
        self.markers_type = markers_type
        self.log_dir = log_dir
        self.mask_ratio = mask_ratio

        # Even though markers_type is given, we are now using joints instead of markers.
        # This parameter may not be needed now, but we keep it for compatibility.
        assert self.markers_type is not None

        # Load SMPL-X models once
        self.smplx_model_male = smplx.create(
            smplx_model_path, model_type='smplx', gender='male', ext='npz',
            use_pca=False, flat_hand_mean=True, create_global_orient=True,
            create_body_pose=True, create_betas=True, create_left_hand_pose=True,
            create_right_hand_pose=True, create_expression=True, create_jaw_pose=True,
            create_leye_pose=True, create_reye_pose=True, create_transl=True,
            batch_size=self.clip_len
        ).to(device)

        self.smplx_model_female = smplx.create(
            smplx_model_path, model_type='smplx', gender='female', ext='npz',
            use_pca=False, flat_hand_mean=True, create_global_orient=True,
            create_body_pose=True, create_betas=True, create_left_hand_pose=True,
            create_right_hand_pose=True, create_expression=True, create_jaw_pose=True,
            create_leye_pose=True, create_reye_pose=True, create_transl=True,
            batch_size=self.clip_len
        ).to(device)

        # Load normalization stats if available and requested
        if self.normalize:
            prefix = os.path.join(self.log_dir, 'statistics')
            stats_file = f"{prefix}_{self.mode}.npz"
            if os.path.exists(stats_file):
                stats = np.load(stats_file)
                self.Xmean = stats['Xmean']
                self.Xstd = stats['Xstd']
            else:
                print("[WARNING] Normalization stats not found. Please compute them before using normalize=True.")
                self.Xmean = None
                self.Xstd = None

    def apply_nan_masking(self, clip_img):
        """Applies NaN masking to random frames in a clip.

        Args:
            clip_img: The input clip tensor (T, J, 3), where T is the number of frames, 
                    J is the number of joints, and 3 are the spatial dimensions.
            mask_ratio: The proportion of frames to mask (0.0 to 1.0).

        Returns:
            A tuple containing:
            - masked_clip: The clip with NaN values in masked frames.
            - mask: A boolean mask (True for masked frames, False otherwise).
        """
        num_frames = clip_img.shape[0]
        num_masked_frames = int(self.mask_ratio * num_frames)

        masked_clip = clip_img.clone()  # Important: Clone to avoid modifying the original
        mask = torch.zeros(num_frames, dtype=torch.bool) #initialize as all false

        if num_masked_frames > 0:
            masked_indices = torch.randperm(num_frames)[:num_masked_frames]
            masked_clip[masked_indices] = float('nan')
            mask[masked_indices] = True #set to true if it is masked

        return masked_clip, mask

    def apply_masking(self, clip_img):
        """
        Apply masking to random frames with Gaussian noise based on the mask_ratio.
        """
        num_frames = clip_img.shape[0]  # Number of frames
        mask = torch.ones(num_frames, dtype=torch.float32)
        num_masked_frames = int(self.mask_ratio * num_frames)

        if num_masked_frames > 0:
            masked_indices = torch.randperm(num_frames)[:num_masked_frames]
            mask[masked_indices] = 0.0
            clip_img[masked_indices] = 0.0  # Zero out the masked frames

        return clip_img, mask

    def divide_clip(self, dataset_name='HumanEva', amass_dir=None, stride=None):
        npz_fnames = sorted(glob.glob(os.path.join(amass_dir, dataset_name, '*/*_poses.npz')))
        cnt_sub_clip = 0

        for npz_fname in npz_fnames:
            cdata = np.load(npz_fname)
            fps = int(cdata['mocap_framerate'])

            if fps == 150:
                sample_rate = 5
            elif fps == 120:
                sample_rate = 4
            elif fps == 60:
                sample_rate = 2
            else:
                # Skip other FPS
                cdata.close()
                continue
            cdata.close()

            clip_len = self.clip_seconds * fps + sample_rate + 1
            stride_ = stride or clip_len
            # We only store metadata here, not loading now
            with np.load(npz_fname) as cdata:
                N = len(cdata['poses'])
            if N >= clip_len:
                for start_idx in range(0, N - clip_len + 1, stride_):
                    self.data_metadata_list.append({
                        'npz_fname': npz_fname,
                        'start_idx': start_idx,
                        'sample_rate': sample_rate,
                        'fps': fps
                    })
                    cnt_sub_clip += 1

        print(f'Generated {cnt_sub_clip} subclips from dataset {dataset_name}.')

    def read_data(self, amass_datasets, amass_dir, stride=None):
        for dataset_name in tqdm(amass_datasets):
            self.divide_clip(dataset_name, amass_dir, stride)
        self.n_samples = len(self.data_metadata_list)
        print(f'[INFO] Generated {self.n_samples} subclips in total.')

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        meta = self.data_metadata_list[index]
        npz_fname = meta['npz_fname']
        start_idx = meta['start_idx']
        sample_rate = meta['sample_rate']
        fps = meta['fps']

        cdata = np.load(npz_fname)
        betas = cdata['betas'][:10]
        gender = str(cdata['gender'])
        trans = cdata['trans'][start_idx:start_idx + self.clip_len * sample_rate:sample_rate]
        poses = cdata['poses'][start_idx:start_idx + self.clip_len * sample_rate:sample_rate]
        cdata.close()

        # Set SMPL-X parameters
        body_param_ = {
            'transl': torch.from_numpy(trans).float().to(device),
            'global_orient': torch.from_numpy(poses[:, :3]).float().to(device),
            'body_pose': torch.from_numpy(poses[:, 3:66]).float().to(device),
            'betas': torch.from_numpy(np.tile(betas, (len(trans), 1))).float().to(device),
        }

        # Global rotation normalization
        body_param_['transl'] = body_param_['transl'] - body_param_['transl'][0].clone()  # center pelvis
        body_param_['transl'][:, 1] += 0.4


        smplx_model = self.smplx_model_male if gender == 'male' else self.smplx_model_female
        with torch.no_grad():
            smplx_output = smplx_model(return_verts=False, return_joints=True, **body_param_)
            joints = smplx_output.joints[:, :25, :]  # [T, 25, 3]

        # Convert to numpy for processing
        joints_np = joints.detach().cpu().numpy()
        # Swap Y and Z axes
        joints_np[:, :, [1, 2]] = joints_np[:, :, [2, 1]]

        # Align to floor
        joints_np[:, :, 1] -= joints_np[:, :, 1].min()

        # Add reference joint: from the original code snippet logic
        reference = joints_np[:, 0] * np.array([1, 0, 1])
        cur_body = np.concatenate([reference[:, np.newaxis], joints_np], axis=1)  # [T, 26, 3]

        # Compute velocity and remove root motion
        velocity = cur_body[1:, 0:1] - cur_body[:-1, 0:1]
        cur_body[:, :, 0] -= cur_body[:, 0:1, 0]
        cur_body[:, :, 2] -= cur_body[:, 0:1, 2]

        # Compute forward direction
        across = joints_np[:, 2] - joints_np[:, 1]  # based on joints 1 and 2
        across /= np.linalg.norm(across, axis=-1, keepdims=True)
        forward = np.cross(across, np.array([[0, 1, 0]]))
        forward = filters.gaussian_filter1d(forward, 20, axis=0, mode='nearest')
        forward /= np.linalg.norm(forward, axis=-1, keepdims=True)

        # Rotate so forward aligns with Z axis
        target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
        rotation = Quaternions.between(forward, target)[:, np.newaxis]
        cur_body = rotation * cur_body

        # Swap Y and Z back
        cur_body[:, :, [1, 2]] = cur_body[:, :, [2, 1]]

        # Remove last frame and the reference joint column
        # Original code: cur_body = cur_body[:-1, 1:, :]
        cur_body = cur_body[:-1, 1:, :]

        # Normalize if stats are available
        if self.normalize and self.Xmean is not None and self.Xstd is not None:
            cur_body = (cur_body - self.Xmean) / self.Xstd

        original_clip = torch.from_numpy(cur_body).float()  # [T-1, 25, 3]
        masked_clip = original_clip.clone()

        if self.mask_ratio > 0.0:
            masked_clip, mask = self.apply_nan_masking(masked_clip)
        else:
            mask = torch.ones(original_clip.shape[0], dtype=torch.float32)

        return masked_clip, mask, original_clip