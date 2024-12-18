import glob
import json
import os
import numpy as np
import scipy.ndimage.filters as filters
import smplx
import torch
from torch.utils import data
from utils.Pivots import Pivots
from utils.Quaternions import Quaternions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_body_model(type, body_model_path, gender, batch_size, device='cpu', v_template=None):
    body_model = smplx.create(body_model_path, model_type=type,
                              gender=gender, ext='npz',
                              num_pca_comps=24,
                              create_global_orient=True,
                              create_body_pose=True,
                              create_betas=True,
                              create_left_hand_pose=True,
                              create_right_hand_pose=True,
                              create_expression=True,
                              create_jaw_pose=True,
                              create_leye_pose=True,
                              create_reye_pose=True,
                              create_transl=True,
                              batch_size=batch_size,
                              v_template=v_template)
    if device == 'cuda':
        return body_model.cuda()
    else:
        return body_model

class GRAB_DataLoader(data.Dataset):
    def __init__(self, clip_seconds=8, clip_fps=30, normalize=False, split='train', markers_type=None, mode=None, is_debug=False, log_dir='', smplx_model_path=None):
        self.clip_seconds = clip_seconds
        self.clip_len = clip_seconds * clip_fps + 2
        self.normalize = normalize
        self.clip_fps = clip_fps
        self.split = split
        self.mode = mode
        self.is_debug = is_debug
        self.markers_type = markers_type
        self.log_dir = log_dir
        self.smplx_model_path = smplx_model_path

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

        self.npz_fnames = []
        self.n_samples = 0

    def read_data(self, amass_datasets, amass_dir, stride=1):
        for dataset_name in amass_datasets:
            data_path = os.path.join(amass_dir, dataset_name)
            npz_files = sorted(glob.glob(os.path.join(data_path, '*.npz')))
            self.npz_fnames.extend(npz_files)
        self.n_samples = len(self.npz_fnames)
        print('[INFO] Found {} sequences in total for split: {}.'.format(self.n_samples, self.split))

    def generate_linear_frames(self, marker_start, marker_end, num_frames=61):
        t = np.linspace(0, 1, num_frames)[:, np.newaxis, np.newaxis]
        interpolated_frames = marker_start[np.newaxis, :, :] * (1 - t) + marker_end[np.newaxis, :, :] * t
        return interpolated_frames

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        npz_fname = self.npz_fnames[index]
        cdata = np.load(npz_fname, allow_pickle=True)

        fps = int(cdata['framerate'])
        if fps == 150:
            sample_rate = 5
        elif fps == 120:
            sample_rate = 4
        elif fps == 60:
            sample_rate = 2
        else:
            raise ValueError("Unsupported framerate: {}".format(fps))

        clip_len = self.clip_seconds * fps + sample_rate + 1
        N = cdata['n_frames']

        # Extract SMPLX parameters
        seq_transl = cdata['body'][()]['params']['transl']
        seq_global_orient = cdata['body'][()]['params']['global_orient']
        seq_body_pose = cdata['body'][()]['params']['body_pose']
        seq_left_hand_pose = cdata['body'][()]['params']['left_hand_pose']
        seq_right_hand_pose = cdata['body'][()]['params']['right_hand_pose']
        seq_leye_pose = cdata['body'][()]['params']['leye_pose']
        seq_reye_pose = cdata['body'][()]['params']['reye_pose']
        seq_betas = cdata['betas']
        seq_gender = str(cdata['gender'])

        # Pad if needed
        if N < clip_len:
            diff = clip_len - N
            seq_transl = np.concatenate([np.repeat(seq_transl[0:1], diff, axis=0), seq_transl], axis=0)
            seq_global_orient = np.concatenate([np.repeat(seq_global_orient[0:1], diff, axis=0), seq_global_orient], axis=0)
            seq_body_pose = np.concatenate([np.repeat(seq_body_pose[0:1], diff, axis=0), seq_body_pose], axis=0)
            seq_left_hand_pose = np.concatenate([np.repeat(seq_left_hand_pose[0:1], diff, axis=0), seq_left_hand_pose], axis=0)
            seq_right_hand_pose = np.concatenate([np.repeat(seq_right_hand_pose[0:1], diff, axis=0), seq_right_hand_pose], axis=0)
            seq_leye_pose = np.concatenate([np.repeat(seq_leye_pose[0:1], diff, axis=0), seq_leye_pose], axis=0)
            seq_reye_pose = np.concatenate([np.repeat(seq_reye_pose[0:1], diff, axis=0), seq_reye_pose], axis=0)

        body_param_ = {
            'transl': seq_transl[-clip_len:][::sample_rate],
            'global_orient': seq_global_orient[-clip_len:][::sample_rate],
            'body_pose': seq_body_pose[-clip_len:][::sample_rate],
            'left_hand_pose': seq_left_hand_pose[-clip_len:][::sample_rate],
            'right_hand_pose': seq_right_hand_pose[-clip_len:][::sample_rate],
            'leye_pose': seq_leye_pose[-clip_len:][::sample_rate],
            'reye_pose': seq_reye_pose[-clip_len:][::sample_rate],
            'betas': np.repeat(seq_betas, seq_transl[-clip_len:][::sample_rate].shape[0], axis=0)
        }

        for param_name in body_param_:
            body_param_[param_name] = torch.from_numpy(body_param_[param_name]).float().to(device)

        bs = body_param_['transl'].shape[0]

        # Load SMPLX model
        gender_to_use = 'female' if seq_gender == 'female' else 'male'
        body_model = get_body_model('smplx', self.smplx_model_path, gender_to_use, bs, 'cuda')

        smplx_output = body_model(return_verts=True, **body_param_)
        joints = smplx_output.joints  # [T, 127, 3]
        markers = smplx_output.vertices[:, self.markers_ids, :]

        # --- Global rotation normalization to face Y axis (as original code) ---
        joints_frame0 = joints[0].detach()
        x_axis = joints_frame0[2, :] - joints_frame0[1, :]
        x_axis[-1] = 0
        x_axis = x_axis / torch.norm(x_axis)
        z_axis = torch.tensor([0, 0, 1], device=device, dtype=torch.float32)
        y_axis = torch.cross(z_axis, x_axis)
        y_axis = y_axis / torch.norm(y_axis)
        transf_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=1)
        joints = torch.matmul(joints - joints_frame0[0], transf_rotmat)
        markers = torch.matmul(markers - joints_frame0[0], transf_rotmat)

        # Extract body joints (no hands) for local representation
        body_joints = joints[:, :25]

        # Convert to numpy
        joints_np = joints.detach().cpu().numpy()
        markers_np = markers.detach().cpu().numpy()

        # Swap y/z for joints and markers
        joints_np[:, :, [1, 2]] = joints_np[:, :, [2, 1]]
        markers_np[:, :, [1, 2]] = markers_np[:, :, [2, 1]]

        # Put on floor
        min_height = joints_np[:, :, 1].min()
        joints_np[:, :, 1] -= min_height
        markers_np[:, :, 1] -= min_height

        # Add reference joint (pelvis projection) for joints
        reference = joints_np[:, 0] * np.array([1, 0, 1])
        joints_np = np.concatenate([reference[:, np.newaxis], joints_np], axis=1)

        # Compute velocity
        velocity = (joints_np[1:, 0:1] - joints_np[:-1, 0:1]).copy()

        # Move to local coordinates
        joints_np[:, :, 0] -= joints_np[:, 0:1, 0]
        joints_np[:, :, 2] -= joints_np[:, 0:1, 2]
        markers_np[:, :, 0] -= markers_np[0:1, 0:1, 0]  # Align markers to pelvis origin as well
        markers_np[:, :, 2] -= markers_np[0:1, 0:1, 2]

        # Compute forward direction from across vector (original code uses joints)
        across = joints_np[:, 2] - joints_np[:, 1]
        across = across / np.sqrt((across ** 2).sum(-1))[..., None]

        direction_filterwidth = 20
        forward = np.cross(across, np.array([[0, 1, 0]]))
        forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')
        forward = forward / np.sqrt((forward ** 2).sum(-1))[..., None]

        target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
        rotation = Quaternions.between(forward, target)[:, None]

        # Apply rotation to joints and markers
        joints_np = rotation * joints_np
        markers_np = rotation * markers_np
        velocity = rotation[1:] * velocity

        # Swap y/z again after rotation to match original final coordinates
        joints_np[:, :, [1, 2]] = joints_np[:, :, [2, 1]]
        markers_np[:, :, [1, 2]] = markers_np[:, :, [2, 1]]

        # Remove last frame from representation as original code does
        # (Original code uses [0:-1, 1:, :], we mimic that)
        joints_np = joints_np[:-1, 1:, :]  # local body representation (T-1, body_joints_count, 3)
        markers_np = markers_np[:-1, :, :]  # similarly remove last frame

        # Extract start/end frames BEFORE slicing if needed
        # We need start/end from the fully rotated/normalized space but before removing the last frame
        # The original code uses joint_start = joints_np[0], joint_end = joints_np[-2]
        # We must get them before slicing off the last frame. Let's redo that properly:

        # Let's redo step to get start/end frames from the original, unsliced arrays:
        # We'll apply the same transformations but store start/end before slicing:
        
        # Rebuild full transformed joints_np for start/end extraction:
        full_joints_np = rotation * (joints.detach().cpu().numpy() - joints_frame0[0].detach().cpu().numpy()) @ transf_rotmat.detach().cpu().numpy()
        full_joints_np[:, :, [1, 2]] = full_joints_np[:, :, [2, 1]]
        full_joints_np[:, :, 1] -= full_joints_np[:, :, 1].min()
        full_joints_np[:, :, 0] -= full_joints_np[0:1, 0:1, 0]
        full_joints_np[:, :, 2] -= full_joints_np[0:1, 0:1, 2]
        full_joints_np[:, :, [1, 2]] = full_joints_np[:, :, [2, 1]]

        # Extract joint_start and joint_end (25 joints) after all transformations
        # Use the same indexing as original: start = [0], end = [-2]
        joint_start = full_joints_np[0, :25, :]
        joint_end = full_joints_np[-2, :25, :]

        # Generate linear frames (no rotation needed, they are already in final space)
        slerp_img = self.generate_linear_frames(joint_start, joint_end, 61)
        slerp_img = torch.from_numpy(slerp_img).float()

        # Convert final clip to tensor
        clip_img_joints = torch.from_numpy(joints_np).float().permute(2, 1, 0)  # [3, joints, T-1]
        clip_img_markers = torch.from_numpy(markers_np).float().permute(2, 1, 0) # [3, markers, T-1]

        return clip_img_joints.numpy(), clip_img_markers.numpy(), slerp_img.numpy()
