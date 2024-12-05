#visualize using smpl and not smplx
from LiftUpTransformer.data.dataloader_temp import FrameLoader
from torch.utils.data import DataLoader
import torch
import numpy as np

BATCH_SIZE = 128
LEARNING_RATE = 2e-4

NUM_EPOCHS = 50
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 6
N_PARTS = 9  # Number of body parts
N_MARKERS = 143  # Number of markers
MASKING_RATIO = 0.35  # Ratio of markers to mask
REGRESSOR_PATH = 'body_utils/J_regressor_h36m_correct.npy'
SMPLX_MODEL_PATH = 'body_utils/body_models'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths and Dataset Parameters
DATASET_DIR = '../../../data/edwarde/dataset/AMASS'
MARKERS_TYPE = 'f15_p22'
NORMALIZE = True

TRAIN_DATASET = ['HumanEva']
VAL_DATASET = ['HUMAN4D', 'KIT']


dataset = FrameLoader(
        dataset_dir=DATASET_DIR,
        markers_type=MARKERS_TYPE,
        smplx_model_path=SMPLX_MODEL_PATH,
        normalize=NORMALIZE,
        dataset_list=TRAIN_DATASET,
        apply_masking=True,        
        masking_ratio=MASKING_RATIO,
        regressor_path=REGRESSOR_PATH
)
# Create DataLoader
data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)
# Get first batch
batch = next(iter(data_loader))
# Extract joints
# Extract the original markers and joints from the batch
original_markers = batch['markers'].cpu().numpy()
joints = batch['joints']  # Already converted to NumPy in the collate_fn

original_markers = original_markers.cpu().numpy() if isinstance(original_markers, torch.Tensor) else original_markers
joints = joints.cpu().numpy() if isinstance(joints, torch.Tensor) else joints
# Save the data to an npz file for visualization
output_file = "no_normalize_human36m_regressor_first_batch_data.npz"
np.savez(output_file, original_markers=original_markers, joints=joints)
print(f"Data saved to {output_file}")
# Visualize joints for the first batch
print("First batch joints:")
print(joints[0])  # Joints for visualization