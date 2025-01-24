from PoseBridge.data.end2end_dataloader import GRAB_DataLoader, save_preprocessed_data

grab_datasets = ['s9', 's10']
grab_dir = '../../../data/edwarde/dataset/grab/GraspMotion'
smplx_model_path = 'body_utils/body_models'
save_dir = "../../../data/edwarde/dataset/testingforSAGA"

# Preload and save processed data
dataset = GRAB_DataLoader(clip_seconds=2, clip_fps=30, mode='local_markers_3dv', markers_type='f0_p5')
dataset.read_data(grab_datasets, grab_dir)
dataset.create_body_repr(with_hand=False, smplx_model_path=smplx_model_path)
save_preprocessed_data(dataset, save_dir)

# import numpy as np
# """TODO: Implement the function below to compute the global mean and standard deviation of the dataset."""

# def compute_incremental_stats(grab_datasets, grab_dir, smplx_model_path, save_path):
#     """
#     Compute global mean and standard deviation across multiple folders incrementally.
#     Args:
#         grab_datasets (list of str): List of dataset folder names (e.g., ['s1', 's2']).
#         grab_dir (str): Base directory containing the dataset folders.
#         smplx_model_path (str): Path to the SMPL-X body model.
#         save_path (str): Path to save the computed statistics.
#     """
#     n_total = 0  # Total number of samples processed
#     global_mean = 0.0  # or torch.zeros_like(batch_mean) if it's a tensor
#     global_var = 0.0  # or torch.zeros_like(batch_var) if it's a tensor


#     for dataset_name in grab_datasets:
#         print(f"Processing dataset: {dataset_name}")

#         # Initialize the GRAB_DataLoader for the current dataset
#         dataset = GRAB_DataLoader(
#             clip_seconds=2, clip_fps=30, mode='local_markers_3dv', markers_type='f15_p22'
#         )
#         dataset.read_data([dataset_name], grab_dir)
#         dataset.create_body_repr(with_hand=False, smplx_model_path=smplx_model_path)

#         # Iterate over all samples in the current dataset
#         # Assuming traj_gt_list contains the trajectory data
#         for global_traj in dataset.traj_gt_list:
#             traj = global_traj.reshape(-1, global_traj.shape[-1])  # Flatten to [N, features]

#             # Perform statistics computation as before
#             batch_mean = np.mean(traj, axis=0)
#             batch_var = np.var(traj, axis=0)
#             batch_size = traj.shape[0]

#             # Incremental mean and variance computation
#             delta = batch_mean - global_mean
#             n_total += batch_size

#             global_mean += delta * batch_size / n_total
#             global_var += batch_var * batch_size / n_total + delta**2 * (n_total - batch_size) * batch_size / n_total**2


#     # Compute the final standard deviation
#     global_std = np.sqrt(global_var)

#     # Save the statistics
#     np.savez_compressed(save_path, mean=global_mean, std=global_std)
#     print(f"Global statistics saved to {save_path}")

# grab_datasets = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']
# grab_dir = '../../../data/edwarde/dataset/grab/GraspMotion'
# smplx_model_path = 'body_utils/body_models'
# stats_save_path = 'normalization_stats.npz'

# compute_incremental_stats(grab_datasets, grab_dir, smplx_model_path, stats_save_path)