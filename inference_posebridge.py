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

# Hyperparameters and Settings
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_JOINTS = 22
NUM_MARKERS = 143
CLIP_SECONDS = 2
CLIP_FPS = 30

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
    joint_end = end_smplx_results.joints.detach().cpu().numpy()

    return end_data, end_body, marker_end, joint_end, object_transl_0, object_global_orient_0


def set_initial_pose(args, end_smplx, markers_ids):
    betas = end_smplx['betas']
    n = betas.shape[0]

    ### can be customized
    initial_orient = np.array([[1.5421, -0.00219, -0.0171]])   # Set initial body pose orientation / None (same orientation as the ending pose)
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

    end_transl = end_smplx['transl']
    end_global_orient = end_smplx['global_orient']

    rand_z = np.zeros(n).reshape(-1, 1)
    rand_displacement = np.concatenate([rand_x, rand_y, rand_z], axis=-1).reshape(n, -1)

    start_smplx = {}
    start_smplx['betas'] = betas
    start_smplx['transl'] = end_transl + rand_displacement
    start_smplx['global_orient'] = end_global_orient if initial_orient is None else initial_orient.repeat(n, axis=0).astype(np.float32)

    if initial_pose is not None:
        start_smplx['body_pose'] = initial_pose.repeat(n, axis=0).astype(np.float32)

    start_body_mesh, start_smplx_results = get_body_mesh(start_smplx, args.gender, 0, betas.shape[0])
    marker_start = start_smplx_results.vertices.detach().cpu().numpy()[:, markers_ids, :]
    joint_start = start_smplx_results.joints.detach().cpu().numpy()

	
    return marker_start.astype(np.float32), joint_start.astype(np.float32)

def generate_linear_frames(marker_start, marker_end, num_frames=61):
    t = np.linspace(0, 1, num_frames)[:, np.newaxis, np.newaxis]
    return marker_start[np.newaxis, :, :] * (1 - t) + marker_end[np.newaxis, :, :] * t

def main(args):
    # Ensure paths exist
    inference_save_dir = 'path_to_save_inference_results'
    os.makedirs(inference_save_dir, exist_ok=True)
    
    # Initialize Models
    temporal_transformer = TemporalTransformer(
        dim_in=3, dim_out=3, dim_feat=128, depth=5, num_heads=8,
        num_joints=NUM_JOINTS, maxlen=CLIP_SECONDS * CLIP_FPS + 1
    ).to(DEVICE)
    
    liftup_transformer = LiftUpTransformer(
        input_dim=3, embed_dim=64, num_joints=NUM_JOINTS,
        num_markers=NUM_MARKERS, num_layers=6, num_heads=4
    ).to(DEVICE)
    
    model = EndToEndModel(temporal_transformer, liftup_transformer).to(DEVICE)
    if os.path.exists(args.best_model_checkpoint_path):
        model.load_state_dict(torch.load(args.best_model_checkpoint_path, map_location=DEVICE))
        print(f"Loaded model from {args.best_model_checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {args.best_model_checkpoint_path}")

    model.eval()

    # Load Data
    grasppose_result_path = os.path.join(
        os.getcwd(), f'results/{args.GraspPose_exp_name}/GraspPose/{args.object}/fitting_results.npz'
    )
    marker_end, joint_end, _, _ = load_ending_pose(args, grasppose_result_path)

    # Set Initial Pose
    markers_ids = get_markers_ids('f15_p22')
    marker_start, joint_start = set_initial_pose(args, {}, markers_ids)
    marker_start = torch.tensor(marker_start).to(DEVICE)
    joint_start = torch.tensor(joint_start).to(DEVICE)
    marker_end = torch.tensor(marker_end).to(DEVICE)

    # Generate Trajectories
    slerp_img = generate_linear_frames(joint_start, joint_end, num_frames=61)

    # Inference
    with torch.no_grad():
        temp_filled_joints, lift_predicted_markers = model(slerp_img)

    # Save Results
    result_path = os.path.join(inference_save_dir, "inference_results.npz")
    np.savez_compressed(
        result_path,
        temp_filled_joints=temp_filled_joints.cpu().numpy(),
        lift_predicted_markers=lift_predicted_markers.cpu().numpy()
    )
    print(f"Inference results saved to {result_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--GraspPose_exp_name", type=str, required=True, help="Experiment name for GraspPose.")
    # parser.add_argument("--object", type=str, required=True, help="Object name used in GraspPose.")
    # parser.add_argument("--best_model_checkpoint_path", type=str, required=True, help="Path to the best model checkpoint.")
    parser.add_argument("--gender", type=str, default="male", help="Gender for SMPL-X model.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # main(args)
    
    load_path = '/home/edwarde/PosePretrain/test_02/GraspPose/camera/fitting_results.npz'
    markers_ids= get_markers_ids('f15_p22')

    # Call the function
    end_data, end_smplx, marker_end, joint_end, object_transl_0, object_global_orient_0 = load_ending_pose(args, load_path)

    # Print the shapes of the outputs from load_ending_pose
    print(f"Shape of marker_end: {marker_end.shape}")
    print(f"Shape of joint_end: {joint_end.shape}")
    print(f"Shape of object_transl_0: {object_transl_0.shape}")
    print(f"Shape of object_global_orient_0: {object_global_orient_0.shape}")

    marker_start, joint_start = set_initial_pose(args, end_smplx, markers_ids)

    # Print the shapes of the outputs from set_initial_pose
    print(f"Shape of marker_start: {marker_start.shape}")
    print(f"Shape of joint_start: {joint_start.shape}")

    # Generate interpolated frames
    slerp_img = generate_linear_frames(joint_start, joint_end, num_frames=61)

    # Print the shape of the interpolated frames
    print(f"Shape of slerp_img: {slerp_img.shape}")
