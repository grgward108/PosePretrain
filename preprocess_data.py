import os
import torch
import numpy as np
from LiftUpTransformer.data.dataloader import FrameLoader
from tqdm import tqdm

DATASET_DIR = '../../../../gs/bs/tga-openv/edwarde/AMASS'
SMPLX_MODEL_PATH = '../../../../gs/bs/tga-openv/edwarde/body_utils/body_models'
REGRESSOR_PATH = '../../../../gs/bs/tga-openv/edwarde/body_utils/J_regressor_h36m_correct.npy'
MARKERS_TYPE = 'f15_p22'
NORMALIZE = True

TRAIN_DATASET = ['HumanEva', 'ACCAD', 'CMU', 'DanceDB', 'Eyes_Japan_Dataset', 'GRAB']
VAL_DATASET = ['HUMAN4D', 'KIT']

# Initialize dataset and models as before
dataset = FrameLoader(
    dataset_dir=DATASET_DIR,
    smplx_model_path=SMPLX_MODEL_PATH,
    markers_type=MARKERS_TYPE,
    normalize=NORMALIZE,
    dataset_list=TRAIN_DATASET,
    regressor_path=REGRESSOR_PATH
)

os.makedirs('preprocessed_data', exist_ok=True)

for i in tqdm(range(len(dataset))):
    sample = dataset[i]
    # Run your SMPL-X and SMPL-H computations once
    markers, joints = dataset.batch_process_frames([sample['frame']], [sample['gender']])
    # Save results
    np.savez_compressed(os.path.join('../../../../gs/bs/tga-openv/edwarde/AMASS/preprocessed', f'{i}.npz'),
                        markers=markers.numpy(),
                        joints=joints.numpy())
