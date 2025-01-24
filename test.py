import os
import torch
import numpy as np
from PoseBridge.data.dataloader import GRAB_DataLoader  # Replace 'your_module' with the actual module where MotionLoader is defined
from torch.utils.data import DataLoader

# Constants
grab_datasets = ['s1']
grab_dir = '../../../data/edwarde/dataset/grab/GraspMotion'
smplx_model_path = 'body_utils/body_models'
clip_seconds = 2
clip_fps = 30
markers_type = 'f15_p22'  # Example markers type
mode = 'local_joints_3dv'  # Choose mode to test
output_folder = "./grab_datadump"  # Directory to save output
normalize = False
mask_ratio = 0.15
log_dir = ''

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
MASK_RATIO = 0.15
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLIP_SECONDS = 2
CLIP_FPS = 30
VAL_BATCH_SIZE = 32
STRIDE = 30



val_dataset = GRAB_DataLoader(clip_seconds=2, clip_fps=30, mode=mode, markers_type=markers_type)

val_dataset.read_data(grab_datasets, grab_dir)

"""143 markers / 55 joints if with_hand else 72 markers / 25 joints"""
val_dataset.create_body_repr(with_hand=False, smplx_model_path=smplx_model_path)

print('length of dataset:', len(val_dataset))

val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=4)

def generate_static_mask(length=61):
    mask = torch.ones(length, dtype=torch.float32)
    mask[0] = 0  # First frame masked
    mask[-1] = 0  # Last frame masked
    return mask