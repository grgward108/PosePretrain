
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
        Apply masking to random frames based on the mask_ratio.
        """
        num_frames = clip_img.shape[0]  # Number of frames in the clip
        mask = torch.ones(num_frames, dtype=torch.float32)  # Initialize mask with ones
        num_masked_frames = int(self.mask_ratio * num_frames)

        if num_masked_frames > 0:
            # Randomly select frames to mask
            masked_indices = torch.randperm(num_frames)[:num_masked_frames]
            mask[masked_indices] = 0.0  # Set masked frames to 0
            clip_img[masked_indices] = 0.0  # Zero out the masked frames

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



    def create_body_repr(self, with_hand=False, global_rot_norm=True, 
                         smplx_model_path=None):
        print('[INFO] create motion clip imgs by {}...'.format(self.mode))

        smplx_model_male = smplx.create(smplx_model_path, model_type='smplx', gender='male', ext='npz',
                                        use_pca=False,  flat_hand_mean=True, # different here, true: mean hand pose is a flat hand
                                        create_global_orient=True, create_body_pose=True, create_betas=True,
                                        create_left_hand_pose=True, create_right_hand_pose=True, create_expression=True,
                                        create_jaw_pose=True, create_leye_pose=True, create_reye_pose=True, create_transl=True,
                                        batch_size=self.clip_len).to(device)
        smplx_model_female = smplx.create(smplx_model_path, model_type='smplx', gender='female', ext='npz',
                                          use_pca=False, flat_hand_mean=True, # different here, true: mean hand pose is a flat hand
                                          create_global_orient=True, create_body_pose=True, create_betas=True,
                                          create_left_hand_pose=True, create_right_hand_pose=True, create_expression=True,
                                          create_jaw_pose=True, create_leye_pose=True, create_reye_pose=True, create_transl=True,
                                          batch_size=self.clip_len).to(device)
        self.clip_img_list = []
        for i in tqdm(range(self.n_samples)):
            ####################### set smplx params (gpu tensor) for each motion clip ##################
            body_param_ = {}
            body_param_['transl'] = self.data_dict_list[i]['trans']  # [T, 3]
            body_param_['global_orient'] = self.data_dict_list[i]['poses'][:, 0:3]
            body_param_['body_pose'] = self.data_dict_list[i]['poses'][:, 3:66]  # [T, 63]
            body_param_['left_hand_pose'] = self.data_dict_list[i]['poses'][:, 66:111]  # [T, 45]
            body_param_['right_hand_pose'] = self.data_dict_list[i]['poses'][:, 111:]  # [T, 45]
            body_param_['betas'] = np.tile(self.data_dict_list[i]['betas'][0:10], (len(body_param_['transl']), 1))  # [T, 10]

            for param_name in body_param_:
                body_param_[param_name] = torch.from_numpy(body_param_[param_name]).float().to(device)


            ############################### normalize first frame transl/gloabl_orient #############################
            if not global_rot_norm:
                # move bodies s.t. pelvis of the first frame is at the origin
                body_param_['transl'] = body_param_['transl'] - body_param_['transl'][0]  # [T, 3]
                body_param_['transl'][:,1] = body_param_['transl'][:,1] + 0.4

            bs = body_param_['transl'].shape[0]

            ######################## set  body representations (global joints for body/hand) #####################
            if self.mode in ['global_joints', 'local_joints', 'local_joints_3dv', 'local_joints_3dv_v1',
                             'global_markers', 'local_markers', 'local_joints_3dv_4chan',
                             'local_markers_3dv', 'local_markers_3dv_4chan']:
                if self.data_dict_list[i]['gender'] == 'male':
                    smplx_output = smplx_model_male(return_verts=True, **body_param_)  # generated human body mesh
                elif self.data_dict_list[i]['gender'] == 'female':
                    smplx_output = smplx_model_female(return_verts=True, **body_param_)  # generated human body mesh
                joints = smplx_output.joints  # [T, 127, 3]

                if self.mode in ['global_markers', 'local_markers', 'local_markers_3dv', 'local_markers_3dv_4chan']:
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
                    if self.mode in ['global_markers', 'local_markers', 'local_markers_3dv', 'local_markers_3dv_4chan']:
                        # markers_frame0 = markers[0].detach()
                        # markers = torch.matmul(markers - markers_frame0[0], transf_rotmat)   # [T(/bs), n_marker, 3]
                        markers = torch.matmul(markers - joints_frame0[0], transf_rotmat)   # [T(/bs), n_marker, 3]

                ######## obtain binary contact labels
                if self.mode in ['local_joints_3dv', 'local_joints_3dv_4chan']:
                    # left_heel_id, right_heel_id = 7, 8, left_toe, right_toe = 10, 11
                    # velocity criteria
                    left_heel_vel = torch.norm((joints[1:, 7:8] - joints[0:-1, 7:8]) * self.clip_fps, dim=-1)  # [T-1, 1]
                    right_heel_vel = torch.norm((joints[1:, 8:9] - joints[0:-1, 8:9]) * self.clip_fps, dim=-1)
                    left_toe_vel = torch.norm((joints[1:, 10:11] - joints[0:-1, 10:11]) * self.clip_fps, dim=-1)
                    right_toe_vel = torch.norm((joints[1:, 11:12] - joints[0:-1, 11:12]) * self.clip_fps, dim=-1)

                    foot_joints_vel = torch.cat([left_heel_vel, right_heel_vel, left_toe_vel, right_toe_vel],
                                                dim=-1)  # [T-1, 4]

                    is_contact = torch.abs(foot_joints_vel) < 0.22  # todo: set contact threshold speed
                    contact_lbls = torch.zeros([joints.shape[0], 4]).to(device)  # all -1, [T, 4]
                    contact_lbls[0:-1, :][is_contact == True] = 1.0  # 0/1, 1 if contact for first T-1 frames

                    # z height criteria
                    z_thres = torch.min(joints[:, :, -1]) + 0.10   # todo: set contact threshold speed
                    foot_joints = torch.cat([joints[:, 7:8], joints[:, 8:9], joints[:, 10:11], joints[:, 11:12]],
                                            dim=-2)  # [T, 4, 3]
                    thres_lbls = (foot_joints[:, :, 2] < z_thres).float()  # 0/1, [T, 4]

                    # combine 2 criterias
                    contact_lbls = contact_lbls * thres_lbls
                    contact_lbls[-1, :] = thres_lbls[-1, :]  # last frame contact lbl: only look at z height
                    # contact_lbls[contact_lbls == 0] = -1.0  # tensor, 1 for contact, -1 for no contact
                    contact_lbls = contact_lbls.detach().cpu().numpy()

                if self.mode in ['local_markers_3dv', 'local_markers_3dv_4chan']:
                    # left foot markers: 25, 30, 16, right_foot_markers: 55, 60, 47
                    # velocity criteria
                    left_heel_vel = torch.norm((markers[1:, 9:10] - markers[0:-1, 9:10]) * self.clip_fps, dim=-1)  # [T-1, 1]
                    right_heel_vel = torch.norm((markers[1:, 25:26] - markers[0:-1, 25:26]) * self.clip_fps, dim=-1)
                    left_toe_vel = torch.norm((markers[1:, 40:41] - markers[0:-1, 40:41]) * self.clip_fps, dim=-1)
                    right_toe_vel = torch.norm((markers[1:, 43:44] - markers[0:-1, 43:44]) * self.clip_fps, dim=-1)


                    foot_markers_vel = torch.cat([left_heel_vel, right_heel_vel, left_toe_vel, right_toe_vel],
                                                dim=-1)  # [T-1, 4]

                    is_contact = torch.abs(foot_markers_vel) < 0.22  # todo: set contact threshold speed
                    contact_lbls = torch.zeros([markers.shape[0], 4]).to(device)  # all -1, [T, 4]
                    contact_lbls[0:-1, :][is_contact == True] = 1.0  # 0/1, 1 if contact for first T-1 frames

                    # z height criteria
                    z_thres = torch.min(markers[:, :, -1]) + 0.10  # todo: set contact threshold speed
                    foot_markers = torch.cat([markers[:, 9:10], markers[:, 25:26], markers[:, 40:41], markers[:, 43:44]],
                                             dim=-2)  # [T, 4, 3]
                    thres_lbls = (foot_markers[:, :, 2] < z_thres).float()  # 0/1, [T, 4]

                    # combine 2 criterias
                    contact_lbls = contact_lbls * thres_lbls
                    contact_lbls[-1, :] = thres_lbls[-1, :]  # last frame contact lbl: only look at z height
                    contact_lbls = contact_lbls.detach().cpu().numpy()



                ######## get body representation
                body_joints = joints[:, 0:25]    # [T, 25, 3]  root(1) + body(21) + jaw/leye/reye(3)
                hand_joints = joints[:, 25:55]   # [T, 30, 3]

                # if not with_hand:
                #     cur_body = body_joints  # [T, 25, 3]
                # else:
                #     cur_body = torch.cat([body_joints, hand_joints], axis=1)  # [T, 55, 3]


                if self.mode in ['local_joints_3dv', 'local_joints_3dv_4chan']:
                    cur_body = body_joints  # [T, 25, 3]
                if self.mode in ['local_markers_3dv', 'local_markers_3dv_4chan']:
                    cur_body = markers  # [T, n_marker, 3]

                ############################# local joints from Holten ###############################
                if self.mode in ['local_joints_3dv', 'local_joints_3dv_4chan',
                                 'local_markers_3dv', 'local_markers_3dv_4chan']:
                    cur_body = cur_body.detach().cpu().numpy()  # numpy, [T, 25 or 68, 3], in (x,y,z)

                    cur_body[:, :, [1, 2]] = cur_body[:, :, [2, 1]] # swap y/z axis  --> in (x,z,y)

                    """ Put on Floor """
                    cur_body[:, :, 1] = cur_body[:, :, 1] - cur_body[:, :, 1].min()

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


                    cur_body[:, :, [1, 2]] = cur_body[:, :, [2, 1]]
                    cur_body = cur_body[0:-1, 1:, :]  # [T-1, 25 or 68, 3]

                    if self.mode in ['local_joints_3dv', 'local_markers_3dv']:
                        # global vel (3) + local pose (25*3) + contact label (4)
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




        self.clip_img_list = np.asarray(self.clip_img_list)  # [N, T-1, d] / [N, 4, T-1, d]

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

            return masked_clip, mask, original_clip, trans

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