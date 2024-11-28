from LiftUpTransformer.data.dataloader import FrameLoader
from torch.utils.data import DataLoader
import torch


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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths and Dataset Parameters
DATASET_DIR = '../../../data/edwarde/dataset/AMASS'
SMPLX_MODEL_PATH = 'body_utils/body_models'
MARKERS_TYPE = 'f15_p22'
NORMALIZE = True

TRAIN_DATASET = ['HumanEva', 'ACCAD', 'CMU','DanceDB', 'Eyes_Japan_Dataset', 'GRAB']
VAL_DATASET = ['HUMAN4D', 'KIT']


dataset = FrameLoader(
        dataset_dir=DATASET_DIR,
        smplx_model_path=SMPLX_MODEL_PATH,
        markers_type=MARKERS_TYPE,
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
joints = batch['original_markers'], batch['part_labels'], batch['joints']

# Visualize joints for the first batch
print("First batch joints:")
print(joints[0])  # Joints for visualization
