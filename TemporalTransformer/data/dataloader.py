
'''
using markers on GraspMotion dataset (including hand markers)
'''
import glob
import json
import os
import sys

import numpy as np
import scipy.ndimage.filters as filters
import smplx
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as Rwhy 
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
    def __init__(self, clip_seconds=8, clip_fps=30, normalize=False, split='train', markers_type=None, mode=None, is_debug=False, mask_ratio=0.0, log_dir=''):
        """
        markers_type = ['f0_p0, f15_p0, f0_p5, f15_p5, f0_p22, f15_p22']
        f{m}_p{n}: m, n are the number of markers on the single-hand finger and palm respetively.
        I would suggest you to try: 
            (1) 'f0_p0': no markers on fingers and palm, but we still have 3 on the hand
            (2) 'f0_p5': 5 markers on 5 fingertips
            (3) 'f15_p5'
        """


        self.clip_seconds = clip_seconds
        self.clip_len = clip_seconds * clip_fps + 2 # T+2 frames for each clip
        self.data_dict_list = []
        self.normalize = normalize
        self.clip_fps = clip_fps
        self.split = split  # train/test
        self.mode = mode
        self.is_debug = is_debug
        self.markers_type = markers_type
        self.log_dir = log_dir
        self.mask_ratio = mask_ratio

        assert self.markers_type is not None


        f, p = self.markers_type.split('_')
        finger_n, palm_n = int(f[1:]), int(p[1:])

        with open('./body_utils/smplx_markerset.json') as f:
            markerset = json.load(f)['markersets']
            self.markers_ids = []
            for marker in markerset:
                if marker['type'] == 'finger' and finger_n == 0:
                    continue
                elif 'palm' in marker['type']:
                    if palm_n == 5 and marker['type'] == 'palm_5':
                        self.markers_ids += list(marker['indices'].values())
                    elif palm_n == 22 and marker['type'] == 'palm':
                        self.markers_ids += list(marker['indices'].values())
                    else:
                        continue
                else:
                    self.markers_ids += list(marker['indices'].values())


        print(f"Number of markers selected for '{self.markers_type}': {len(self.markers_ids)}")

    def apply_masking(self, clip_img):
        """
        Apply masking to random frames with Gaussian noise based on the mask_ratio.
        Args:
            clip_img: Tensor of shape (num_frames, num_markers, marker_dim).
        Returns:
            clip_img: Masked input tensor.
            mask: Mask tensor of shape (num_frames,).
        """
        num_frames = clip_img.shape[0]  # Number of frames in the clip
        mask = torch.ones(num_frames, dtype=torch.float32)  # Initialize mask with ones
        num_masked_frames = int(self.mask_ratio * num_frames)

        if num_masked_frames > 0:
            # Randomly select frames to mask
            masked_indices = torch.randperm(num_frames)[:num_masked_frames]
            mask[masked_indices] = 0.0  # Set masked frames to 0

            # Replace masked frames with Gaussian noise
            noise = torch.randn_like(clip_img[masked_indices])
            clip_img[masked_indices] = noise  # Apply Gaussian noise to masked frames

        return clip_img, mask

    def divide_clip(self, dataset_name='HumanEva', amass_dir=None, stride=None):
        npz_fnames = sorted(glob.glob(os.path.join(amass_dir, dataset_name, '*/*_poses.npz')))  # List of all npz sequence files
        fps_list = []
        cnt_sub_clip = 0

        for npz_fname in npz_fnames:
            cdata = np.load(npz_fname)

            fps = int(cdata['mocap_framerate'])  # Check FPS of current sequence
            fps_list.append(fps)
            if fps == 150:
                sample_rate = 5
            elif fps == 120:
                sample_rate = 4
            elif fps == 60:
                sample_rate = 2
            else:
                continue

            # Calculate clip length and stride
            clip_len = self.clip_seconds * fps + sample_rate + 1
            stride = stride or clip_len  # Default stride is no overlap

            N = len(cdata['poses'])  # Total frame number of the current sequence
            if N >= clip_len:
                for start_idx in range(0, N - clip_len + 1, stride):
                    # Extract subclip data
                    data_dict = {
                        'trans': cdata['trans'][start_idx:start_idx + clip_len:sample_rate],  # [T, 3]
                        'poses': cdata['poses'][start_idx:start_idx + clip_len:sample_rate],  # [T, 156]
                        'betas': cdata['betas'],  # [10]
                        'gender': str(cdata['gender']),  # male/female
                        'mocap_framerate': fps,
                    }
                    self.data_dict_list.append(data_dict)
                    cnt_sub_clip += 1
            else:
                continue

        print(f'Generated {cnt_sub_clip} subclips from dataset {dataset_name}.')


        print('get {} sub clips from dataset {}'.format(cnt_sub_clip, dataset_name))
        # print('fps range:', min(fps_list), max(fps_list), '\n')


    def read_data(self, amass_datasets, amass_dir, stride=None):
        for dataset_name in tqdm(amass_datasets):
            self.divide_clip(dataset_name, amass_dir, stride)
        self.n_samples = len(self.data_dict_list)
        print(f'[INFO] Generated {self.n_samples} subclips in total.')



    def create_body_repr(self, global_rot_norm=True, smplx_model_path=None):
        """
        Creates body representations specifically for the 'local_joints_3dv' mode.

        Args:
            global_rot_norm (bool): Whether to normalize the global rotation to face the y-axis.
            smplx_model_path (str): Path to the SMPL-X model.
        """
        print('[INFO] Creating motion clip images for local_joints_3dv...')

        # Load SMPL-X models
        smplx_model_male = smplx.create(
            smplx_model_path, model_type='smplx', gender='male', ext='npz',
            use_pca=False, flat_hand_mean=True, create_global_orient=True, 
            create_body_pose=True, create_betas=True, create_left_hand_pose=True,
            create_right_hand_pose=True, create_expression=True, create_jaw_pose=True,
            create_leye_pose=True, create_reye_pose=True, create_transl=True,
            batch_size=self.clip_len
        ).to(device)
        
        smplx_model_female = smplx.create(
            smplx_model_path, model_type='smplx', gender='female', ext='npz',
            use_pca=False, flat_hand_mean=True, create_global_orient=True,
            create_body_pose=True, create_betas=True, create_left_hand_pose=True,
            create_right_hand_pose=True, create_expression=True, create_jaw_pose=True,
            create_leye_pose=True, create_reye_pose=True, create_transl=True,
            batch_size=self.clip_len
        ).to(device)

        # Initialize a list to store processed clip images
        self.clip_img_list = []

        for i in tqdm(range(self.n_samples)):
            # Set SMPL-X parameters
            body_param_ = {
                'transl': torch.from_numpy(self.data_dict_list[i]['trans']).float().to(device),
                'global_orient': torch.from_numpy(self.data_dict_list[i]['poses'][:, :3]).float().to(device),
                'body_pose': torch.from_numpy(self.data_dict_list[i]['poses'][:, 3:66]).float().to(device),
                'betas': torch.from_numpy(
                    np.tile(self.data_dict_list[i]['betas'][:10], (self.clip_len, 1))
                ).float().to(device),
            }

            # Normalize global translation/orientation (if enabled)
            if global_rot_norm:
                body_param_['transl'] -= body_param_['transl'][0].clone()  # Center pelvis at origin
                body_param_['transl'][:, 1] += 0.4  # Adjust height for visualization

            # Generate joints using SMPL-X
            smplx_model = smplx_model_male if self.data_dict_list[i]['gender'] == 'male' else smplx_model_female
            with torch.no_grad():
                smplx_output = smplx_model(return_verts=False, return_joints=True, **body_param_)
                joints = smplx_output.joints[:, :25, :]  # Extract the first 25 joints

            # Normalize joints for local coordinates
            joints_np = joints.detach().cpu().numpy()
            joints_np[:, :, [1, 2]] = joints_np[:, :, [2, 1]]  # Swap y and z axes (x, z, y)
            

            # Align to floor and reference joint
            joints_np[:, :, 1] -= joints_np[:, :, 1].min()  # Place on floor
            
            # Extract pelvis global translation before reducing to local motion
            pelvis_global = joints_np[:, 0:1, :].copy()  # [T-1, 1, 3]
            pelvis_global[:, :, 2] = 0  # Set the z-dimension (third dimension) to 0

            
            reference = joints_np[:, 0] * np.array([1, 0, 1])  # Add reference joint
            cur_body = np.concatenate([reference[:, np.newaxis], joints_np], axis=1)

            # Compute velocities and normalize forward direction
            velocity = cur_body[1:, 0:1] - cur_body[:-1, 0:1]
            cur_body[:, :, 0] -= cur_body[:, 0:1, 0]
            cur_body[:, :, 2] -= cur_body[:, 0:1, 2]

            # Filter forward direction
            across = joints_np[:, 2] - joints_np[:, 1]
            across /= np.linalg.norm(across, axis=-1, keepdims=True)
            forward = np.cross(across, np.array([[0, 1, 0]]))
            forward = filters.gaussian_filter1d(forward, 20, axis=0, mode='nearest')
            forward /= np.linalg.norm(forward, axis=-1, keepdims=True)

            # Remove Y rotation
            target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
            rotation = Quaternions.between(forward, target)[:, np.newaxis]
            cur_body = rotation * cur_body

            # Adjust axes back and finalize
            cur_body[:, :, [1, 2]] = cur_body[:, :, [2, 1]]
            cur_body = cur_body[:-1, 1:, :]  # Remove the reference frame and final frame
            
            # Append pelvis global translation to cur_body
            cur_body = np.concatenate([pelvis_global[:-1], cur_body], axis=1)  # [T-1, num_joints+1, 3]

            # Append processed clip to list
            self.clip_img_list.append(cur_body)

        # Convert list to numpy array
        self.clip_img_list = np.asarray(self.clip_img_list)

        if self.normalize:
            prefix = os.path.join(self.log_dir, 'statistics')
            epsilon = 1e-6  # To handle zero std

            if self.mode in ['local_joints_3dv', 'local_markers_3dv']:
                Xmean = self.clip_img_list.mean(axis=(0, 1))  # Compute mean across all clips and frames
                Xstd = self.clip_img_list.std(axis=(0, 1))    # Compute std across all clips and frames
                Xstd = np.maximum(Xstd, epsilon)             # Avoid division by zero

                if self.split == 'train':
                    np.savez_compressed(f"{prefix}_{self.mode}.npz", Xmean=Xmean, Xstd=Xstd)
                    self.clip_img_list = (self.clip_img_list - Xmean) / Xstd
                elif self.split == 'test':
                    stats = np.load(f"{prefix}_{self.mode}.npz")
                    self.clip_img_list = (self.clip_img_list - stats['Xmean']) / stats['Xstd']



        print('[INFO] motion clip imgs created.')


    def __len__(self):
        return self.n_samples


    def __getitem__(self, index):
        if self.mode in ['local_joints_3dv', 'local_markers_3dv']:
            original_clip = torch.from_numpy(self.clip_img_list[index]).float()  # [T, num_markers, 3]
            masked_clip = original_clip.clone()  # Make a copy for masking
 
            if self.mask_ratio > 0.0:
                masked_clip, mask = self.apply_masking(masked_clip)  # Apply masking
            else:
                mask = torch.ones(original_clip.shape[0], dtype=torch.float32)  # All frames visible

            return masked_clip, mask, original_clip

        elif self.mode in ['local_joints_3dv_4chan', 'local_markers_3dv_4chan']:
            clip_img = self.clip_img_list[index]  # [4, T, d]
            clip_img = torch.from_numpy(clip_img).float().permute(0, 2, 1)  # [4, d, T]
            
        return [clip_img]



if __name__ == "__main__":

    # amass_datasets = ['HumanEva', 'MPI_HDM05', 'MPI_mosh', 'SFU', 'SSM_synced', 'Transitions_mocap',
    #                         'ACCAD', 'BMLhandball', 'BMLmovi', 'BioMotionLab_NTroje', 'CMU',
    #                         'DFaust_67', 'Eyes_Japan_Dataset', 'EKUT', 'KIT', 'ACCAD', 'MPI_HDM05', 'MPI_mosh']
    amass_datasets = ['HumanEva']
    amass_dir = 'dataset/AMASS'
    smplx_model_path = 'body_utils/body_models'
    vposer_model_path = 'body_utils/body_models/VPoser/vposer_v1_0'
    vposer_model, _ = load_vposer(vposer_model_path, vp_model='snapshot')
    vposer_model = vposer_model.to(device)

    dataset = MotionLoader(clip_seconds=2, clip_fps=30, mode='local_markers_3dv')
    dataset.read_data(amass_datasets, amass_dir)
    dataset.create_body_repr(with_hand=True, smplx_model_path=smplx_model_path)