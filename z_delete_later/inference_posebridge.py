import argparse
import os
import sys
import numpy as np
import torch
from smplx.lbs import batch_rodrigues
from utils.utils_body import get_body_mesh, get_markers_ids
from PoseBridge.models.models import EndToEndModel
from TemporalTransformer.models.models import TemporalTransformer
from LiftUpTransformer.models.models import LiftUpTransformer
from utils.Pivots import Pivots
from utils.Quaternions import Quaternions
import scipy.ndimage.filters as filters
import logging


# Hyperparameters and Settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_JOINTS = 22
NUM_MARKERS = 143
CLIP_SECONDS = 2
CLIP_FPS = 30

BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLIP_SECONDS = 2
CLIP_FPS = 30
MARKERS_TYPE = 'f15_p22'
MODE = 'local_markers_3dv'
SMPLX_MODEL_PATH = 'body_utils/body_models'
NUM_JOINTS = 22
NUM_MARKERS = 143
VALIDATE_EVERY = 5
PELVIS_LOSS_WEIGHT = 5.0
LEG_RECONSTRUCTION_WEIGHT = 2.0
RIGHT_HAND_WEIGHTS = 2.0

FINAL_RECONSTRUCTION_LOSS_WEIGHT = 0.6
FINAL_VELOCITY_LOSS_WEIGHT = 0.1
FINAL_ACCELERATION_LOSS_WEIGHT = 0.05
FINAL_PELVIS_LOSS_WEIGHT = 0.10
FINAL_FOOT_SKATING_LOSS_WEIGHT = 0.15

def save_reconstruction_npz(
    slerp_img, temp_filled_joints, temp_original_joints, lift_predicted_markers,
    lift_original_markers, traj_gt, rot_0_pivot, transf_matrix_smplx, joint_start,
    save_dir, epoch, exp_name
):
    """
    Save reconstruction data for the entire batch.

    Args:
        slerp_img: Tensor of shape (B, T, J, C) for the entire batch.
        temp_filled_joints: Tensor of shape (B, T, J, C) for the reconstructed joints.
        temp_original_joints: Tensor of shape (B, T, J, C) for the ground truth joints.
        lift_predicted_markers: Tensor of shape (B, T, M, C) for the predicted markers.
        lift_original_markers: Tensor of shape (B, T, M, C) for the ground truth markers.
        save_dir: Directory to save the file.
        epoch: Current epoch number.
        exp_name: Experiment name.
    """
    save_path = os.path.join(save_dir, exp_name)
    os.makedirs(save_path, exist_ok=True)

    # Construct a single file name for the entire batch
    file_name = f"evaluation_posebridge__1234.npz"

    # Save all data in a single file
    np.savez_compressed(
        os.path.join(save_path, file_name),
        slerp_img=slerp_img.cpu().numpy(),
        temp_filled_joints=temp_filled_joints.cpu().numpy(),
        temp_original_joints=temp_original_joints.cpu().numpy(),
        lift_predicted_markers=lift_predicted_markers.cpu().numpy(),
        lift_original_markers=lift_original_markers.cpu().numpy(),
        traj_gt=traj_gt.cpu().numpy() if traj_gt is not None else None,
        rot_0_pivot=rot_0_pivot.cpu().numpy() if rot_0_pivot is not None else None,
        transf_matrix_smplx=transf_matrix_smplx.cpu().numpy() if transf_matrix_smplx is not None else None,
        joint_start=joint_start.cpu().numpy() if joint_start is not None else None,
    )
    print(f"Reconstruction saved at {os.path.join(save_path, file_name)}")
def load_ending_pose(args, grasppose_result_path):
    end_data = np.load(grasppose_result_path, allow_pickle=True)

    sample_index = np.arange(0, len(end_data['markers']))   # ignore

    end_body_full = end_data['body'][()]
    end_body = {}
    for k in end_body_full:
        end_body[k] = end_body_full[k][sample_index]

    # object data
    object_transl_0 = torch.tensor(end_data['object'][()]['transl'])[sample_index].to(device)
    object_global_orient_0 = torch.tensor(end_data['object'][()]['global_orient'])[sample_index].to(device)
    object_global_orient_0 = batch_rodrigues(object_global_orient_0.view(-1, 3)).view([len(object_global_orient_0), 3, 3])#.detach().cpu().numpy()

    # get ending body (optional) mesh and markers/joints
    smplx_beta = end_body['betas']
    start_idx = 0
    bs = len(smplx_beta)
    end_body_mesh, end_smplx_results = get_body_mesh(end_body, args.gender, start_idx, bs)
    marker_end = end_smplx_results.vertices.detach().cpu().numpy()[:, markers_ids, :]
    joint_end = end_smplx_results.joints[:, :25, :].detach().cpu().numpy()

    return end_data, end_body, marker_end, joint_end, object_transl_0, object_global_orient_0
def set_initial_pose(args, end_smplx, markers_ids):
    betas = end_smplx['betas']
    n = betas.shape[0]

    ### can be customized
    initial_orient = None   # Set initial body pose orientation / None (same orientation as the ending pose)
    initial_pose = np.array([-0.10901122,  0.0461413,   0.02993835,  0.11612727, -0.06200547,  0.08139142,
                                -0.02208922,  0.06683847, -0.02794579,  0.45293584, -0.16446967, -0.06646398,
                                0.07430738,  0.16469607,  0.05346995,  0.23588121, -0.09054547,  0.06633219,
                                -0.08885075,  0.25389493, -0.04105648, -0.1263972,  -0.2095012,  -0.01349497,
                                -0.1308483,   0.00866051, -0.00762679, -0.20351738, -0.0055567,   0.09453899,
                                0.09627768,  0.10411494,  0.03997851,  0.07713828, -0.01521101, -0.04545524,
                                0.10470242, -0.09646956, -0.40639114,  0.11441539,  0.09596836,  0.3891292,
                                0.1657324,   0.12639643,  0.01392403,  0.0669774,  -0.25228527, -0.69750136,
                                -0.01904383,  0.1466294,   0.6928179,   0.00282627,  0.00742727, -0.11434615,
                                -0.08387394, -0.05599072,  0.0974379,   0.00966642, -0.03484239,  0.10031673,
                                0.04399946,  0.04642308, -0.10101389]).reshape(-1, 63)
    rand_x = np.random.rand(n).reshape(-1, 1) * 0.04 - 0.02
    rand_y = np.random.rand(n).reshape(-1, 1) + 0.05
    rand_z = np.zeros(n).reshape(-1, 1)

    end_transl = end_smplx['transl']
    end_global_orient = end_smplx['global_orient']
    rand_displacement = np.concatenate([rand_x, rand_y, rand_z], axis=-1).reshape(n, -1)

    start_smplx = {}
    start_smplx['betas'] = betas
    start_smplx['transl'] = end_transl + rand_displacement
    start_smplx['global_orient'] = end_global_orient if initial_orient is None else initial_orient.repeat(n, axis=0).astype(np.float32)

    if initial_pose is not None:
        start_smplx['body_pose'] = initial_pose.repeat(n, axis=0).astype(np.float32)

    start_body_mesh, start_smplx_results = get_body_mesh(start_smplx, args.gender, 0, betas.shape[0])
    marker_start = start_smplx_results.vertices.detach().cpu().numpy()[:, markers_ids, :]
    joint_start = start_smplx_results.joints[:, :25, :].detach().cpu().numpy()

	
    return marker_start.astype(np.float32), joint_start.astype(np.float32)
def set_initial_pose_2(args, end_smplx, joint_end, markers_ids):
    betas = end_smplx['betas']
    n = betas.shape[0]

    ### can be customized
    initial_orient = None
    initial_pose = np.zeros((1, 63), dtype=np.float32)  # Set initial pose to zero

    end_transl = end_smplx['transl']
    end_global_orient = end_smplx['global_orient']

    # Reference joint is the first joint in smplx_output.joints
    reference_joint = joint_end[:, 0, :]  # Shape: [n, 3]

    # Calculate direction vector from reference joint to the end translation
    direction_vector = end_transl - reference_joint  # Vector from reference_joint to end_transl

    # Project onto the horizontal plane (x and y only, ignoring z)
    direction_vector[:, 2] = 0  # Ignore vertical (z) component

    # Normalize the direction vector
    norms = np.linalg.norm(direction_vector, axis=-1, keepdims=True)  # Calculate norms
    direction_vector = direction_vector / (norms + 1e-8)  # Avoid division by zero

    # Set displacement to 0.5 meters forward along the direction vector
    displacement = 0.5 * direction_vector  # Shape: [n, 3]
    
    reference_joint = np.array(reference_joint)
    displacement = np.array(displacement)
    
    print("reference_joint shape:", reference_joint.shape)
    print("displacement shape:", displacement.shape)


    # Compute the starting translation
    start_transl = reference_joint + displacement

    # Initialize SMPL-X parameters
    start_smplx = {}
    start_smplx['betas'] = betas
    start_smplx['transl'] = start_transl
    start_smplx['global_orient'] = end_global_orient if initial_orient is None else initial_orient.repeat(n, axis=0).astype(np.float32)

    if initial_pose is not None:
        start_smplx['body_pose'] = initial_pose.repeat(n, axis=0).astype(np.float32)

    # Generate body mesh and extract markers/joints
    start_body_mesh, start_smplx_results = get_body_mesh(start_smplx, args.gender, 0, betas.shape[0])
    marker_start = start_smplx_results.vertices.detach().cpu().numpy()[:, markers_ids, :]
    joint_start = start_smplx_results.joints[:, :25, :].detach().cpu().numpy()

    return marker_start.astype(np.float32), joint_start.astype(np.float32)
def get_forward_joint(joint_start):
	""" Joint_start: [B, N, 3] in xyz """
	x_axis = joint_start[:, 2, :] - joint_start[:, 1, :]
	x_axis[:, -1] = 0
	x_axis = x_axis / torch.norm(x_axis, dim=-1).unsqueeze(1)
	z_axis = torch.tensor([0, 0, 1]).float().unsqueeze(0).repeat(len(x_axis), 1).to(device)
	y_axis = torch.cross(z_axis, x_axis)
	y_axis = y_axis / torch.norm(y_axis, dim=-1).unsqueeze(1)
	transf_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=1)
	return y_axis, transf_rotmat
def generate_linear_frames_batch(joint_start, joint_end, num_frames=61, device="cpu"):
    """
    Generate interpolated frames for a batch of joint positions.

    Args:
        joint_start (numpy.ndarray or torch.Tensor): Starting joint positions (batch_size, 25, 3).
        joint_end (numpy.ndarray or torch.Tensor): Ending joint positions (batch_size, 25, 3).
        num_frames (int): Number of frames to generate.
        device (str or torch.device): Device to perform computations on.

    Returns:
        torch.Tensor: Interpolated frames (batch_size, num_frames, 25, 3).
    """
    # Convert inputs to PyTorch tensors if they are NumPy arrays
    if isinstance(joint_start, np.ndarray):
        joint_start = torch.tensor(joint_start, device=device, dtype=torch.float32)
    if isinstance(joint_end, np.ndarray):
        joint_end = torch.tensor(joint_end, device=device, dtype=torch.float32)
        
    device = joint_start.device
    joint_end = joint_end.to(device)

    # Ensure inputs have the correct shape
    assert joint_start.shape == joint_end.shape, "joint_start and joint_end must have the same shape"
    assert joint_start.ndim == 3, "joint_start and joint_end must be 3D (batch_size, 25, 3)"

    # Create interpolation factor
    t = torch.linspace(0, 1, num_frames, device=device).view(1, num_frames, 1, 1)  # Shape: (1, num_frames, 1, 1)

    # Expand inputs for broadcasting
    joint_start = joint_start.unsqueeze(1)  # Shape: (batch_size, 1, 25, 3)
    joint_end = joint_end.unsqueeze(1)  # Shape: (batch_size, 1, 25, 3)

    # Perform linear interpolation
    interpolated_frames = joint_start * (1 - t) + joint_end * t  # Shape: (batch_size, num_frames, 25, 3)

    return interpolated_frames
def save_only_inference(model, slerp_img, DEVICE, save_dir=None, exp_name="default"):
    """
    Perform inference with the model and save the outputs, skipping any loss calculations.

    Args:
        model (torch.nn.Module): The model to use for inference.
        slerp_img (torch.Tensor): Input tensor for the model (B, T, J, C).
        DEVICE (torch.device): Device for computation (e.g., CUDA or CPU).
        save_dir (str, optional): Directory to save the outputs. Defaults to None.
        exp_name (str, optional): Experiment name for saving outputs. Defaults to "default".
    """
    model.eval()

    # Move input to device
    slerp_img = slerp_img.to(DEVICE, dtype=torch.float32)

    with torch.no_grad():
        # Perform forward pass
        temp_filled_joints, lift_predicted_markers = model(slerp_img)

        # Save outputs if save_dir is specified
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{exp_name}_inference_results.npz")
            np.savez_compressed(
                save_path,
                slerp_img=slerp_img.cpu().numpy(),
                temp_filled_joints=temp_filled_joints.cpu().numpy(),
                lift_predicted_markers=lift_predicted_markers.cpu().numpy(),
            )
            print(f"Saved inference results to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name for GraspPose.")
    # parser.add_argument("--object", type=str, required=True, help="Object name used in GraspPose.")
    parser.add_argument("--gender", type=str, default="male", help="Gender for SMPL-X model.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    SAVE_DIR = os.path.join('posebridge_inference_log', args.exp_name)
    os.makedirs(SAVE_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[logging.FileHandler(os.path.join(SAVE_DIR, f"{args.exp_name}_inference.log")), logging.StreamHandler()],
    )
    logger = logging.getLogger()

    # main(args)
    
    load_path = '/home/edwarde/PosePretrain/evaluation_01/GraspPose/camera/fitting_results.npz'
    markers_ids= get_markers_ids('f15_p22')

    end_data, end_smplx, marker_end, joint_end, object_transl_0, object_global_orient_0 = load_ending_pose(args, load_path)
    marker_start, joint_start = set_initial_pose_2(args, end_smplx, joint_end, markers_ids)

    # Ensure inputs are PyTorch tensors
    joint_start = torch.tensor(joint_start, dtype=torch.float32).to(device)
    joint_end = torch.tensor(joint_end, dtype=torch.float32).to(device)
    
    start_y_axis, start_transform = get_forward_joint(joint_start)
    end_y_axis, end_transform = get_forward_joint(joint_end)

    print("joint_start shape:", joint_start.shape)
    print("joint_end shape: ", joint_end.shape)

    new_joint_start = torch.bmm(joint_start, start_transform.transpose(1, 2))
    new_joint_end = torch.bmm(joint_end, end_transform.transpose(1, 2))
    
    translation_offset_x = new_joint_start[:, 0:1, 0]  # Only x-coordinate
    translation_offset_2_x = new_joint_end[:, 0:1, 0]  # Only x-coordinate


    # Shift joint_start to align x to 0
    new_joint_start_adjusted = new_joint_start.clone()  # Create a copy to avoid modifying the original tensor
    new_joint_start[:, :, 0] -= translation_offset_x  # Subtract only the x-offset
    
    new_joint_end_adjusted = new_joint_end.clone()  # Create a copy to avoid modifying the original tensor
    new_joint_end[:, :, 0] -= translation_offset_2_x  # Subtract only the x-offset

    
    translation_offset = new_joint_start[:, 0:1, :2]  # Extract the x, y coordinates of the reference joint (first joint)

    # Shift joint_start to align x and y to 0
    new_joint_start_adjusted = new_joint_start.clone()  # Create a copy to avoid modifying the original tensor
    new_joint_start_adjusted[:, :, :2] -= translation_offset  # Subtract the offset for x, y coordinates

    # Shift joint_end by the same offset to preserve relative distances
    new_joint_end_adjusted = new_joint_end.clone()  # Create a copy to avoid modifying the original tensor
    new_joint_end_adjusted[:, :, :2] -= translation_offset
    
    print("Translation offset:", translation_offset)
    print("Adjusted joint_start:", new_joint_start_adjusted[0, 0, :])
    print("Adjusted joint_end:", new_joint_end_adjusted[0, 0, :])
    
    global_pelvis_start = torch.cat(
        [new_joint_start_adjusted[:, 0:1, :2], torch.zeros_like(new_joint_start_adjusted[:, 0:1, 2:3])], dim=2
    )  # Shape: [T, 1, 3]

    global_pelvis_end = torch.cat(
        [new_joint_end_adjusted[:, 0:1, :2], torch.zeros_like(new_joint_end_adjusted[:, 0:1, 2:3])], dim=2
    ) 
    #here i want to add global_pelvis_start third element as 0
    
    
    def get_local_alignment(cur_body_joints, device):
        cur_body_joints = cur_body_joints.to(device)  # Ensure it's on the correct device

        cur_body_joints[:, :, [1, 2]] = cur_body_joints[:, :, [2, 1]]  # Swap y/z axis  --> now (x,z,y)

        """ Put on Floor for joints """
        cur_body_joints[:, :, 1] = cur_body_joints[:, :, 1] - cur_body_joints[:, :, 1].min()
        
        """ To local coordinates (for joints) """
        cur_body_joints[:, :, 0] = cur_body_joints[:, :, 0] - cur_body_joints[:, 0:1, 0]  # [T, 1+(25 or 55), 3]
        cur_body_joints[:, :, 2] = cur_body_joints[:, :, 2] - cur_body_joints[:, 0:1, 2]
        
        cur_body_joints[:, :, [1, 2]] = cur_body_joints[:, :, [2, 1]]
        
        return cur_body_joints

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transform joints
    new_joint_start_locally_aligned = get_local_alignment(new_joint_start_adjusted, device)
    new_joint_end_locally_aligned = get_local_alignment(new_joint_end_adjusted, device)
    
    new_joint_start_locally_aligned = torch.cat([global_pelvis_start, new_joint_start_adjusted], dim=1)  # [T, 26 or 56, 3]
    new_joint_end_locally_aligned = torch.cat([global_pelvis_end, new_joint_end_adjusted], dim=1)  # [T, 26 or 56, 3]

    # Generate interpolated frames
    slerp_img = generate_linear_frames_batch(new_joint_start_locally_aligned, new_joint_end_locally_aligned)
    print("Slerp_img shape:", slerp_img.shape)
    
        # Initialize Models
    temporal_transformer = TemporalTransformer(
        dim_in=3,
        dim_out=3,
        dim_feat=128,
        depth=5,
        num_heads=8,
        num_joints=26,
        maxlen=CLIP_SECONDS * CLIP_FPS + 1,
    ).to(DEVICE)

    liftup_transformer = LiftUpTransformer(
        input_dim=3,
        embed_dim=64,
        num_joints=NUM_JOINTS,
        num_markers=NUM_MARKERS,
        num_layers=6,
        num_heads=4,
    ).to(DEVICE)
    
    temporal_checkpoint_path = '../../../data/edwarde/dataset/finetune_temporal_log/testdifferentweights/epoch_100.pth'
    liftup_checkpoint_path = 'finetune_liftup_log/test3/epoch_100.pth'

    if os.path.exists(temporal_checkpoint_path):
        logger.info(f"Loading TemporalTransformer from checkpoint: {temporal_checkpoint_path}")
        temporal_checkpoint = torch.load(temporal_checkpoint_path, map_location=DEVICE)
        temporal_transformer.load_state_dict(temporal_checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError(f"TemporalTransformer checkpoint not found at {temporal_checkpoint_path}.")

    if os.path.exists(liftup_checkpoint_path):
        logger.info(f"Loading LiftUpTransformer from checkpoint: {liftup_checkpoint_path}")
        liftup_checkpoint = torch.load(liftup_checkpoint_path, map_location=DEVICE)
        liftup_transformer.load_state_dict(liftup_checkpoint['model_state_dict'])
    else:
        raise FileNotFoundError(f"LiftUpTransformer checkpoint not found at {liftup_checkpoint_path}.")

    # Combine Models
    model = EndToEndModel(temporal_transformer, liftup_transformer).to(DEVICE)
    model.eval()  # Set the model to evaluation mode
    logger.info("Starting Evaluation...")
    # Prepare `slerp_img` and model
    save_only_inference(model, slerp_img, DEVICE, save_dir=SAVE_DIR, exp_name=args.exp_name)
