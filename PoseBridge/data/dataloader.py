import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.filters as filters
import smplx
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.Pivots import Pivots
from utils.Quaternions import Quaternions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def color_hex2rgb(hex):
    h = hex.lstrip('#')
    return np.array(  tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) )/255

def get_body_model(type, body_model_path, gender, batch_size,device='cpu',v_template=None):
    '''
    type: smpl, smplx smplh and others. Refer to smplx tutorial
    gender: male, female, neutral
    batch_size: an positive integar
    '''
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
                              v_template=v_template
                              )
    if device == 'cuda':
        return body_model.cuda()
    else:
        return body_model

class GRAB_DataLoader(data.Dataset):
    def __init__(self, clip_seconds=8, clip_fps=30, normalize=False, split='train', markers_type=None, mode=None, is_debug=False, log_dir=''):
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

    def divide_clip(self, dataset_name='GraspMotion', data_dir=None):
        npz_fnames = sorted(glob.glob(os.path.join(data_dir, dataset_name) + '/*.npz'))  # name list of all npz sequence files in current dataset
        fps_list = []
        # print('sequence #: ', len(npz_fnames))
        cnt_sub_clip = 0
        # print('reading sequences in %s...' % (dataset_name))
        for npz_fname in npz_fnames:
            # print(npz_fname)
            cdata = np.load(npz_fname, allow_pickle=True)

            fps = int(cdata['framerate'])  # check fps of current sequence
            fps_list.append(fps)
            if fps == 150:
                sample_rate = 5
            elif fps == 120:
                sample_rate = 4
            elif fps == 60:
                sample_rate = 2
            else:
                continue
            # clip_seconds*30 + 2 more frames (122 after sample, then become 121)
            clip_len = self.clip_seconds*fps + sample_rate + 1 

            N = cdata['n_frames']  # total frame number of the current sequence

            if N >= clip_len:
                seq_transl = cdata['body'][()]['params']['transl']
                seq_global_orient = cdata['body'][()]['params']['global_orient']
                seq_body_pose = cdata['body'][()]['params']['body_pose']
                seq_left_hand_pose = cdata['body'][()]['params']['left_hand_pose']
                seq_right_hand_pose = cdata['body'][()]['params']['right_hand_pose']
                seq_leye_pose = cdata['body'][()]['params']['leye_pose']
                seq_reye_pose = cdata['body'][()]['params']['reye_pose']
            else:
                diff = clip_len - N
                seq_transl = np.concatenate([np.repeat(cdata['body'][()]['params']['transl'][0].reshape(1, -1), diff, axis=0), cdata['body'][()]['params']['transl']], axis=0)
                seq_global_orient = np.concatenate([np.repeat(cdata['body'][()]['params']['global_orient'][0].reshape(1, -1), diff, axis=0), cdata['body'][()]['params']['global_orient']], axis=0)
                seq_body_pose = np.concatenate([np.repeat(cdata['body'][()]['params']['body_pose'][0].reshape(1, -1), diff, axis=0), cdata['body'][()]['params']['body_pose']], axis=0)
                seq_left_hand_pose = np.concatenate([np.repeat(cdata['body'][()]['params']['left_hand_pose'][0].reshape(1, -1), diff, axis=0), cdata['body'][()]['params']['left_hand_pose']], axis=0)
                seq_right_hand_pose = np.concatenate([np.repeat(cdata['body'][()]['params']['right_hand_pose'][0].reshape(1, -1), diff, axis=0), cdata['body'][()]['params']['right_hand_pose']], axis=0)
                seq_leye_pose = np.concatenate([np.repeat(cdata['body'][()]['params']['leye_pose'][0].reshape(1, -1), diff, axis=0), cdata['body'][()]['params']['leye_pose']], axis=0)
                seq_reye_pose = np.concatenate([np.repeat(cdata['body'][()]['params']['reye_pose'][0].reshape(1, -1), diff, axis=0), cdata['body'][()]['params']['reye_pose']], axis=0)

            seq_gender = str(cdata['gender'])
            seq_fps = int(cdata['framerate'])
            seq_vtemp = cdata['body'][()]['vtemp']
            seq_betas = cdata['betas']


            data_dict = {}
            data_dict['body'] = {}
            data_dict['body']['transl'] = seq_transl[-clip_len:][::sample_rate, ]
            data_dict['body']['global_orient'] = seq_global_orient[-clip_len:][::sample_rate, ]
            data_dict['body']['body_pose'] = seq_body_pose[-clip_len:][::sample_rate, ]
            data_dict['body']['left_hand_pose'] = seq_left_hand_pose[-clip_len:][::sample_rate, ]
            data_dict['body']['right_hand_pose'] = seq_right_hand_pose[-clip_len:][::sample_rate, ]
            data_dict['body']['leye_pose'] = seq_leye_pose[-clip_len:][::sample_rate, ]
            data_dict['body']['reye_pose'] = seq_reye_pose[-clip_len:][::sample_rate, ]

            data_dict['betas'] = seq_betas
            data_dict['gender'] = seq_gender
            data_dict['vtemp'] = seq_vtemp
            data_dict['framerate'] = seq_fps

            assert data_dict['body']['transl'].shape[0] == 62

            self.data_dict_list.append(data_dict)


    def read_data(self, amass_datasets, amass_dir):
        for dataset_name in tqdm(amass_datasets):
            print(dataset_name)
            self.divide_clip(dataset_name, amass_dir)
        self.n_samples = len(self.data_dict_list)
        print('[INFO] get {} sub clips in total.'.format(self.n_samples))
    
    def generate_slerp_frames(self, marker_start, marker_end, num_frames=60):
        """
        Generate SLERP (Spherical Linear Interpolation) frames between two markers.
        """
        def normalize(v):
            return v / np.linalg.norm(v, axis=-1, keepdims=True)
        
        # Normalize the start and end markers
        start_normalized = normalize(marker_start)
        end_normalized = normalize(marker_end)
        
        # Compute the angle between the vectors (theta)
        dot_product = np.sum(start_normalized * end_normalized, axis=-1)
        dot_product = np.clip(dot_product, -1.0, 1.0)  # Avoid numerical issues
        theta = np.arccos(dot_product)  # Angle in radians
        
        # Generate SLERP frames
        t_values = np.linspace(0, 1, num_frames)
        frames = []
        
        for t in t_values:
            # Compute SLERP for each marker
            slerp_t = (
                (np.sin((1 - t) * theta) / np.sin(theta))[:, np.newaxis] * start_normalized +
                (np.sin(t * theta) / np.sin(theta))[:, np.newaxis] * end_normalized
            )
            frames.append(slerp_t)
        
        frames = np.array(frames)  # Shape: (num_frames, num_markers, 3)
        
        # Restore magnitudes if needed
        start_magnitude = np.linalg.norm(marker_start, axis=-1, keepdims=True)
        end_magnitude = np.linalg.norm(marker_end, axis=-1, keepdims=True)
        magnitudes = np.linspace(start_magnitude, end_magnitude, num_frames)
        frames = frames * magnitudes
        
        return frames
    
    def generate_linear_frames(self, marker_start, marker_end, num_frames=61):
        """
        Generate interpolated frames using linear interpolation.
        
        Args:
            marker_start (numpy.ndarray): Starting marker positions (N, 3).
            marker_end (numpy.ndarray): Ending marker positions (N, 3).
            num_frames (int): Number of frames to generate.

        Returns:
            numpy.ndarray: Interpolated frames (T, N, 3).
        """
        t = np.linspace(0, 1, num_frames)[:, np.newaxis, np.newaxis]  # Shape: (T, 1, 1)
        interpolated_frames = marker_start[np.newaxis, :, :] * (1 - t) + marker_end[np.newaxis, :, :] * t
        return interpolated_frames


    def create_body_repr(self, with_hand=False, global_rot_norm=True, 
                         smplx_model_path=None):
        print('[INFO] create motion clip imgs by {}...'.format(self.mode))

        self.clip_img_list = []
        self.beta_list = []
        self.rot_0_pivot_list = []
        self.transf_matrix_smplx_list = []
        self.smplx_params_gt_list = []
        self.marker_start_list = []
        self.marker_end_list = []
        self.joint_start_list = []
        self.joint_end_list = []
        self.traj_gt_list = []

        self.male_body_model = get_body_model('smplx', smplx_model_path, 'male', self.clip_len, 'cpu')
        self.female_body_model = get_body_model('smplx', smplx_model_path, 'female', self.clip_len, 'cpu')

        for i in tqdm(range(self.n_samples)):
            ####################### set smplx params (gpu tensor) for each motion clip ##################
            body_param_ = self.data_dict_list[i]['body']
            bs = body_param_['transl'].shape[0]
            body_param_['betas'] = np.repeat(self.data_dict_list[i]['betas'], bs, axis=0)


            for param_name in body_param_:
                body_param_[param_name] = torch.from_numpy(body_param_[param_name]).float().to(device)


            ############################### normalize first frame transl/gloabl_orient #############################
            if not global_rot_norm:
                # move bodies s.t. pelvis of the first frame is at the origin
                body_param_['transl'] = body_param_['transl'] - body_param_['transl'][0]  # [T, 3]
                body_param_['transl'][:,1] = body_param_['transl'][:,1] + 0.4

            bs = body_param_['transl'].shape[0]

            ############### set  body representations (global joints for body/hand) #####################
            body_model = get_body_model('smplx', smplx_model_path, str(self.data_dict_list[i]['gender']), self.clip_len, 'cuda')
            smplx_output = body_model(return_verts=True, **body_param_)
            joints = smplx_output.joints  # [T, 127, 3]
            markers = smplx_output.vertices[:, self.markers_ids, :]

            if global_rot_norm:
                ##### transfrom to pelvis at origin, face y axis
                joints_frame0 = joints[0].detach()  # [N, 3] joints of first frame
                x_axis = joints_frame0[2, :] - joints_frame0[1, :]  # [3]
                x_axis[-1] = 0
                x_axis = x_axis / torch.norm(x_axis)
                z_axis = torch.tensor([0, 0, 1]).float().to(device)
                y_axis = torch.cross(z_axis, x_axis)
                y_axis = y_axis / torch.norm(y_axis)
                transf_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=1)  # [3, 3]
                joints = torch.matmul(joints - joints_frame0[0], transf_rotmat)  # [T(/bs), 25, 3]
                transl_1 = - joints_frame0[0]
                markers = torch.matmul(markers - joints_frame0[0], transf_rotmat)   # [T(/bs), n_marker, 3]

            # save trajectory
            x_axes = joints[:, 2, :] - joints[:, 1, :]  # [T, 3]
            x_axes[:, -1] = 0
            x_axes = x_axes / torch.norm(x_axes, dim=-1).unsqueeze(1)
            z_axes = torch.zeros_like(x_axes).to(device)
            z_axes[:, -1] = 1
            # get "forward" direction of the body
            y_axes = torch.cross(z_axes, x_axes, dim=-1)
            y_axes = y_axes / torch.norm(y_axes, dim=-1).unsqueeze(1)
            
            global_x = joints[:, 0, 0]
            global_y = joints[:, 0, 1]
            # the first and second elements of y_axes (sin and cos of theta)
            rot_forward_x = y_axes[:, 0]
            rot_forward_y = y_axes[:, 1]

            # use only first 61 timestamps for global traj (to stay same dimention as using velocity)
            global_x = global_x.unsqueeze(0).detach().cpu().numpy()
            global_y = global_y.unsqueeze(0).detach().cpu().numpy()
            rot_forward_x = rot_forward_x.unsqueeze(0).detach().cpu().numpy()
            rot_forward_y = rot_forward_y.unsqueeze(0).detach().cpu().numpy()

            global_traj = np.concatenate([global_x, global_y, rot_forward_x, rot_forward_y], axis=0) # [4, 61]
            # self.traj_gt_list.append(global_traj)

            ######## get body representation
            body_joints = joints[:, 0:25]    # [T, 25, 3]  root(1) + body(21) + jaw/leye/reye(3)
            hand_joints = joints[:, 25:55]   # [T, 30, 3]

            if self.mode in ['local_joints_3dv', 'local_joints_3dv_4chan']:
                if with_hand:
                    cur_body = torch.cat([body_joints, hand_joints], axis=1)  # [T, 25, 3]
                else:
                    cur_body = body_joints
            if self.mode in ['local_markers_3dv', 'local_markers_3dv_4chan']:
                cur_body = torch.cat([body_joints[:, 0:1], markers], dim=1)  # first row: pelvis joint [T, 67+1, 3]

            ############################# local joints from Holten ###############################
            if self.mode in ['local_joints_3dv', 'local_joints_3dv_4chan',
                                'local_markers_3dv', 'local_markers_3dv_4chan']:
                cur_body = cur_body.detach().cpu().numpy()  # numpy, [T, 25 or 68, 3], in (x,y,z)
                cur_body[:, :, [1, 2]] = cur_body[:, :, [2, 1]] # swap y/z axis  --> in (x,z,y)

                """ Put on Floor """
                cur_body[:, :, 1] = cur_body[:, :, 1] - cur_body[:, :, 1].min()
                z_transl = cur_body[:, :, 1].min()

                """ Add Reference Joint """
                reference = cur_body[:, 0] * np.array([1, 0, 1])  # [T, 3], (x,y,0)
                cur_body = np.concatenate([reference[:, np.newaxis], cur_body], axis=1)  # [T, 1+25 or 1+68, 3]

                """ Get Root Velocity in floor plane """
                velocity = (cur_body[1:, 0:1] - cur_body[0:-1, 0:1]).copy()  # [T-1, 3] ([:, 1]==0)

                """ To local coordinates """
                cur_body[:, :, 0] = cur_body[:, :, 0] - cur_body[:, 0:1, 0]  # [T, 1+25 or 1+68, 3]
                cur_body[:, :, 2] = cur_body[:, :, 2] - cur_body[:, 0:1, 2]

                """ Get Forward Direction """
                # using joints
                joints_np = joints.detach().cpu().numpy()
                joints_np[:, :, [1, 2]] = joints_np[:, :, [2, 1]] # swap y/z axis
                across = joints_np[:, 2] - joints_np[:, 1]
                across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
                # ipdb.set_trace()

                direction_filterwidth = 20
                forward = np.cross(across, np.array([[0, 1, 0]]))
                forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')
                forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

                """ Remove Y Rotation """
                target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
                rotation = Quaternions.between(forward, target)[:, np.newaxis]
                cur_body = rotation * cur_body  # [T, 1+25 or 1+68, 3]

                """ Get Root Rotation """
                velocity = rotation[1:] * velocity  # [T-1, 1, 3]
                rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps   # [T-1, 1]

                rot_0_pivot = Pivots.from_quaternions(rotation[0]).ps   # [T-1, 1]


                cur_body[:, :, [1, 2]] = cur_body[:, :, [2, 1]]
                cur_body = cur_body[0:-1, 1:, :]  # [T-1, 25 or 68, 3]

                markers_np = markers.detach().cpu().numpy()
                joints_np = joints.detach().cpu().numpy()

                markers_np[:, :, [1, 2]] = markers_np[:, :, [2, 1]]
                joints_np[:, :, [1, 2]] = joints_np[:, :, [2, 1]]

                # 3. Put on floor (subtract the min y from both markers and joints)
                markers_np[:, :, 1] -= (markers_np[:, :, 1].min())
                joints_np[:, :, 1] -= (joints_np[:, :, 1].min())

                # 4. Shift to local coordinates (like we did with cur_body)
                # Align to the root of the first frame (pelvis joint is at index 0)
                markers_np[:, :, 0] -= joints_np[:, 0:1, 0]
                markers_np[:, :, 2] -= joints_np[:, 0:1, 2]

                joints_np[:, :, 0] -= joints_np[:, 0:1, 0]
                joints_np[:, :, 2] -= joints_np[:, 0:1, 2]

                # 5. Apply the same quaternion rotation for forward direction
                # (We already computed 'rotation' above using the forward direction from joints)
                # Apply rotation to markers_np and joints_np
                markers_np = rotation * markers_np
                joints_np = rotation * joints_np

                # 6. Swap back y/z after rotation (because cur_body does it again at the end)
                markers_np[:, :, [1, 2]] = markers_np[:, :, [2, 1]]
                joints_np[:, :, [1, 2]] = joints_np[:, :, [2, 1]]

                # Extract start/end AFTER all transformations
                self.marker_start_list.append(markers_np[0])
                self.marker_end_list.append(markers_np[-2])
                self.joint_start_list.append(joints_np[0])
                self.joint_end_list.append(joints_np[-2])


                if self.mode in ['local_joints_3dv', 'local_markers_3dv']:
                    # global vel (3) + local pose (25*3) + contact label (4)
                    global_vel = np.concatenate([velocity[:, :, 0], velocity[:, :, 2], rvelocity], axis=-1)
                    cur_body = cur_body

                elif self.mode in ['local_joints_3dv_4chan', 'local_markers_3dv_4chan']:
                    channel_local = np.concatenate([cur_body, contact_lbls[0:-1]], axis=-1)[np.newaxis, :, :]  # [1, T-1, d=75(204)+4]
                    T, d = channel_local.shape[1], channel_local.shape[-1]
                    global_x, global_y = velocity[:, :, 0], velocity[:, :, 2]  # [T-1, 1]
                    channel_global_x = np.repeat(global_x, d).reshape(1, T, d)  # [1, T-1, d]
                    channel_global_y = np.repeat(global_y, d).reshape(1, T, d)  # [1, T-1, d]
                    channel_global_r = np.repeat(rvelocity, d).reshape(1, T, d)  # [1, T-1, d]

                    cur_body = np.concatenate([channel_local, channel_global_x, channel_global_y, channel_global_r], axis=0)  # [4, T-1, d]

            self.clip_img_list.append(cur_body)
            self.traj_gt_list.append(global_traj)
            self.rot_0_pivot_list.append(rot_0_pivot)

        self.clip_img_list = np.asarray(self.clip_img_list)  # [N, T-1, d] / [N, 4, T-1, d]
        self.traj_gt_list = np.asarray(self.traj_gt_list)  # [N, 4, T]


        if self.normalize:
            prefix_traj = os.path.join(self.log_dir, 'prestats_GRAB_traj')
            prefix = os.path.join(self.log_dir, 'prestats_GRAB_contact_given_global')
            if with_hand:
                prefix += '_withHand'
            # Xmean = self.clip_img_list.mean(axis=1).mean(axis=0)[np.newaxis, np.newaxis, :]  # [d]
            # Xstd = np.ones(self.clip_img_list.shape[-1]) * self.clip_img_list.std()  # [d]

            if self.mode in ['local_joints_3dv', 'local_markers_3dv']:
                Xmean = self.clip_img_list.mean(axis=1).mean(axis=0)  # [1, 1, d]
                Xmean[-4:] = 0.0

                Xstd = np.ones(self.clip_img_list.shape[-1])
                Xstd[0:2] = self.clip_img_list[:, :, 0:2].std()  # global traj vel x/y
                Xstd[2] = self.clip_img_list[:, :, 2].std()  # rotation vel
                Xstd[3:-4] = self.clip_img_list[:, :, 3:-4].std()  # local joints
                Xstd[-4:] = 1.0

                if self.split == 'train':
                    # if not self.is_debug:
                    np.savez_compressed('{}_{}.npz'.format(prefix, self.mode), Xmean=Xmean, Xstd=Xstd)
                    # else:
                    #     print('local debug. do not save stats.')
                    self.clip_img_list = (self.clip_img_list - Xmean) / Xstd
                elif self.split == 'test':
                    stats = np.load('{}_{}.npz'.format(prefix, self.mode))
                    self.clip_img_list = (self.clip_img_list - stats['Xmean']) / stats['Xstd']

            elif self.mode in ['local_joints_3dv_4chan', 'local_markers_3dv_4chan']:
                d = self.clip_img_list.shape[-1]
                Xmean_local = self.clip_img_list[:, 0].mean(axis=1).mean(axis=0)  # [d]
                Xmean_local[-4:] = 0.0
                Xstd_local = np.ones(d)
                Xstd_local[0:] = self.clip_img_list[:, 0].std()  # [d]
                Xstd_local[-4:] = 1.0

                Xmean_global_xy = self.clip_img_list[:, 1:3].mean()  # scalar
                Xstd_global_xy = self.clip_img_list[:, 1:3].std()  # scalar

                Xmean_global_r = self.clip_img_list[:, 3].mean()  # scalar
                Xstd_global_r = self.clip_img_list[:, 3].std()  # scalar

                if self.split == 'train':
                    # if not self.is_debug:
                    np.savez_compressed('{}_{}.npz'.format(prefix, self.mode),
                                        Xmean_local=Xmean_local, Xstd_local=Xstd_local,
                                        Xmean_global_xy=Xmean_global_xy, Xstd_global_xy=Xstd_global_xy,
                                        Xmean_global_r=Xmean_global_r, Xstd_global_r=Xstd_global_r)
                    # else:
                    #     print('local debug. do not save stats.')
                    self.clip_img_list[:, 0] = (self.clip_img_list[:, 0] - Xmean_local) / Xstd_local
                    self.clip_img_list[:, 1:3] = (self.clip_img_list[:, 1:3] - Xmean_global_xy) / Xstd_global_xy
                    self.clip_img_list[:, 3] = (self.clip_img_list[:, 3] - Xmean_global_r) / Xstd_global_r
                elif self.split == 'test':
                    stats = np.load('{}_{}.npz'.format(prefix, self.mode))
                    self.clip_img_list[:, 0] = (self.clip_img_list[:, 0] - stats['Xmean_local']) / stats['Xstd_local']
                    self.clip_img_list[:, 1:3] = (self.clip_img_list[:, 1:3] - stats['Xmean_global_xy']) / stats['Xstd_global_xy']
                    self.clip_img_list[:, 3] = (self.clip_img_list[:, 3] - stats['Xmean_global_r']) / stats['Xstd_global_r']

            # for trajectory
            # traj_prefix = '/local/home/wuyan/data/GRAB/GraspMotion/Processed_Traj/prestats_GRAB_global_traj'
            traj_Xmean = self.traj_gt_list.mean(axis=-1).mean(axis=0)#.detach().cpu().numpy()  # [4]
            traj_Xstd = np.ones(self.traj_gt_list.shape[1])
            for ith in range(self.traj_gt_list.shape[1]):
                traj_Xstd[ith] = self.traj_gt_list[:, ith, :].std()

            if self.split == 'train':
                np.savez_compressed('{}.npz'.format(prefix_traj),
                                    traj_Xmean=traj_Xmean, traj_Xstd=traj_Xstd)
                
                for ith in range(self.traj_gt_list.shape[1]):
                    self.traj_gt_list[:, ith, :] = (self.traj_gt_list[:, ith, :] - traj_Xmean[ith]) / traj_Xstd[ith]
                # self.traj_Xmean = traj_Xmean
                # self.traj_Xstd = traj_Xstd

            elif self.split == 'test':
                traj_stats = np.load('{}.npz'.format(prefix_traj))

                for ith in range(self.traj_gt_list.shape[1]):
                    self.traj_gt_list[:, ith, :] = (self.traj_gt_list[:, ith, :] - traj_stats['traj_Xmean'][ith]) / traj_stats['traj_Xstd'][ith]

                # self.traj_Xmean = traj_stats['traj_Xmean']
                # self.traj_Xstd = traj_stats['traj_Xstd']


        # self.clip_img_list: [N, T-1, d] / [N, 4, T-1, d]
        if self.mode in ['local_joints_3dv_4chan', 'local_markers_3dv_4chan']:
            print('max/min value in  motion clip: local joints',
                  np.max(self.clip_img_list[:, 0]), np.min(self.clip_img_list[:, 0]))
            print('max/min value in  motion clip: global traj',
                  np.max(self.clip_img_list[:, 1:3]), np.min(self.clip_img_list[:, 1:3]))
            print('max/min value in  motion clip: global rot',
                  np.max(self.clip_img_list[:, 3]), np.min(self.clip_img_list[:, 3]))

        print('[INFO] motion clip imgs created.')


    def __len__(self):
        return self.n_samples


    def __getitem__(self, index):
        if self.mode in ['local_joints_3dv', 'local_markers_3dv']:
            clip_img = self.clip_img_list[index]  # [T, d] d dims of body representation
            clip_img = torch.from_numpy(clip_img).float().permute(2, 1, 0)
        elif self.mode in ['local_joints_3dv_4chan', 'local_markers_3dv_4chan']:
            clip_img = self.clip_img_list[index]  # [4, T, d]
            clip_img = torch.from_numpy(clip_img).float().permute(0, 2, 1)  # [4, d, T]

        smplx_beta = torch.from_numpy(self.data_dict_list[index]['betas'][0:10]).float()  # [10]
        gender = self.data_dict_list[index]['gender']
        rot_0_pivot = self.rot_0_pivot_list[index]
        transf_matrix_smplx = self.transf_matrix_smplx_list[index]
        smplx_params_gt = self.smplx_params_gt_list[index]
        traj = self.traj_gt_list[index]

        if gender == 'female':
            gender = 0
        elif gender == 'male':
            gender = 1

        marker_start = self.marker_start_list[index]
        marker_end = self.marker_end_list[index]
        joint_start = self.joint_start_list[index]
        joint_end = self.joint_end_list[index]

        first_frame = joint_start[0:25, :]  # First 25 joints from the first frame
        last_frame = joint_end[0:25, :] 
        slerp_img = self.generate_linear_frames(first_frame, last_frame, 61)

        slerp_img = torch.from_numpy(slerp_img).float()
       
        return [clip_img.numpy(),slerp_img.numpy(), traj, smplx_beta.numpy(), gender, rot_0_pivot, transf_matrix_smplx.numpy(), smplx_params_gt.numpy(),
                marker_start, marker_end, joint_start, joint_end]

        # return [clip_img]



if __name__ == "__main__":

    # amass_datasets = ['HumanEva', 'MPI_HDM05', 'MPI_mosh', 'SFU', 'SSM_synced', 'Transitions_mocap',
    #                         'ACCAD', 'BMLhandball', 'BMLmovi', 'BioMotionLab_NTroje', 'CMU',
    #                         'DFaust_67', 'Eyes_Japan_Dataset', 'EKUT', 'KIT', 'ACCAD', 'MPI_HDM05', 'MPI_mosh']
    # grab_datasets = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
    grab_datasets = ['s1']
    grab_dir = 'data/GraspMotion'
    smplx_model_path = 'data/AMASS/body_models'  # '/local/home/wuyan/code/smplx/models_smplx_v1_1/models'

    dataset = DataLoader(clip_seconds=2, clip_fps=30, mode='local_markers_3dv_4chan', markers_type='f0_p5')
    dataset.read_data(grab_datasets, grab_dir)

    """143 markers / 55 joints if with_hand else 72 markers / 25 joints"""
    dataset.create_body_repr(with_hand=False, smplx_model_path=smplx_model_path)

    print('length of dataset:', len(dataset))
    print(dataset[0][0].shape)