import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from TemporalTransformer.models.models import TemporalTransformer
from TemporalTransformer.data.lazyloading import MotionLoader  # This should be the updated lazy-loading version
import argparse
import logging
import wandb
import numpy as np
import torch.distributed as dist

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 3e-4
MASK_RATIO = 0.70
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLIP_SECONDS = 2
CLIP_FPS = 30
MARKERS_TYPE = 'f15_p5'  # Not really used for joint extraction now, but keep consistent
MODE = 'local_joints_3dv'
SMPLX_MODEL_PATH = '../../../../gs/bs/tga-openv/edwarde/body_utils/body_models'
STRIDE = 30






def validate_combined(model, dataloader, DEVICE, save_dir=None, epoch=None, exp_name="default"):
    model.eval() 
    total_error = 0
    total_pskl_pred_to_gt = 0
    total_pskl_gt_to_pred = 0
    total_sequences = 0
    total_skating_ratio = 0

    with torch.no_grad():
        for i, (clip_img_joints, clip_img_markers, slerp_img, traj, joint_start_global, joint_end_global, *_) in enumerate(tqdm(dataloader, desc="Validating")):
            exp_name = "abb_no_acc_no_footskat"     
            data_path = f"../dataset/saga_graspmotion_on_grab/{exp_name}/batch_{batch_index}/{exp_name}/batch_{batch_index + 1}_evaluation_posebridge_{exp_name}.npz"
            npz_data = np.load(data_path)
            batch_size = npz_data['slerp_img'].shape[0]

            for sequence_index in range(batch_size):
                # Process slerp_img
                slerp_img = npz_data['slerp_img'][sequence_index]
                slerp_img = restore_clip(slerp_img)

                # Process temp_filled_joints
                temp_filled_joints = npz_data['temp_filled_joints'][sequence_index]
                temp_filled_joints = restore_clip(temp_filled_joints)
                temp_filled_joints = apply_dct_smoothing(temp_filled_joints)

                # Process temp_original_joints
                temp_original_joints = npz_data['temp_original_joints'][sequence_index]
                temp_original_joints = restore_clip(temp_original_joints)

                # Process lift_predicted_markers
                lift_predicted_markers = npz_data['lift_predicted_markers'][sequence_index]
                lift_predicted_markers = apply_dct_smoothing(lift_predicted_markers)
                lift_predicted_markers = restore_clip(lift_predicted_markers)
            
                
                foot_indices = [9, 25]
                skating_ratio, avg_skating_speed = compute_foot_skating_loss(lift_predicted_markers, foot_indices)

                # Process lift_original_markers
                lift_original_markers = npz_data['lift_original_markers'][sequence_index]
                lift_original_markers = restore_clip(lift_original_markers)
                
                counter_subset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
            22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 
            41, 42, 43, 44, 45, 46, 47, 48, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 
            90, 91, 92, 93, 94, 95, 96, 97, 98, 108, 111, 114, 117, 119, 130, 133, 136, 
            139, 141]


                # Compute error
                mask = np.array([i for i in range(lift_original_markers.shape[1]) if i in counter_subset])

                # Apply the mask to both lift_predicted_markers and lift_original_markers
                filtered_predicted_markers = lift_predicted_markers[:, mask, :]
                filtered_original_markers = lift_original_markers[:, mask, :]

                # Compute error using the filtered markers
                my_error = compute_average_l2_distance_error(filtered_predicted_markers, filtered_original_markers)
                total_error += my_error

                # Compute PSKL-J for filtered predicted and original markers
                pskl_pred_to_gt, pskl_gt_to_pred = compute_pskl_j_aggregated(filtered_predicted_markers, filtered_original_markers)
                total_pskl_pred_to_gt += pskl_pred_to_gt
                total_pskl_gt_to_pred += pskl_gt_to_pred
                total_skating_ratio += skating_ratio

                # Increment sequence count
                total_sequences += 1

    average_error = total_error / total_sequences
    average_pskl_pred_to_gt = total_pskl_pred_to_gt / total_sequences
    average_pskl_gt_to_pred = total_pskl_gt_to_pred / total_sequences
    average_skating_ratio = total_skating_ratio / total_sequences

    print(f"Overall Average Error: {average_error}")
    print(f"Overall Average PSKL (Predicted || GT): {average_pskl_pred_to_gt}")
    print(f"Overall Average PSKL (GT || Predicted): {average_pskl_gt_to_pred}")
    print(f"Overall Average Skating Ratio: {average_skating_ratio}")
    

def main(exp_name):
    # Logging Setup
    SAVE_DIR = os.path.join('posebridge_eval_log', exp_name)
    os.makedirs(SAVE_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[logging.FileHandler(os.path.join(SAVE_DIR, f"{exp_name}_inference.log")), logging.StreamHandler()],
    )
    logger = logging.getLogger()

    # Dataset paths
    grab_dir = '../../../data/edwarde/dataset/include_global_traj'
    test_datasets = ['s9', 's10']
    val_dataset = PreprocessedMotionLoader(grab_dir, test_datasets)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Load Models
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

    temporal_checkpoint_path = 'finetune_temporal_log/abb_no_acc_no_footskat/epoch_100.pth'
    liftup_checkpoint_path = 'finetune_liftup_log/test4_fromscratch/epoch_55.pth'

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

    # Evaluate Model
    logger.info("Starting Evaluation...")
    validate_combined(model, val_loader, DEVICE, save_dir=SAVE_DIR, exp_name=exp_name)
    logger.info("Evaluation Complete.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoseBridge Inference and Evaluation Script")
    parser.add_argument("--exp_name", required=True, help="Experiment name")
    args = parser.parse_args()
    main(args.exp_name)
