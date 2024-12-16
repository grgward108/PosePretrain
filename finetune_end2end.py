# Training Script with Lazy-Loading Dataset
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PoseBridge.data.end2end_dataloader_lazyloading import GRAB_DataLoader as MotionLoader
from tqdm import tqdm
import wandb
import argparse
import logging

from PoseBridge.models.models import EndToEndModel
from TemporalTransformer.models.models import TemporalTransformer
from LiftUpTransformer.models.models import LiftUpTransformer

# Hyperparameters and Settings
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLIP_SECONDS = 2
CLIP_FPS = 30
MARKERS_TYPE = 'f15_p5'
MODE = 'local_markers_3dv'
SMPLX_MODEL_PATH = 'body_utils/body_models'
NUM_JOINTS = 25
NUM_MARKERS = 143

TEMPORAL_CHECKPOINT_PATH = "temporal_pretrained/epoch_15.pth"
LIFTUP_CHECKPOINT_PATH = "liftup_log/test3/ckpt/best_model_epoch_222.pth"

grab_dir = '../../../data/edwarde/dataset/grab/GraspMotion'
train_datasets = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8']
test_datasets = ['s9', 's10']


def train_combined(model, optimizer, dataloader, epoch, logger, DEVICE):
    model.train()
    epoch_loss = 0.0

    for clip_img_joints, clip_img_markers, slerp_img in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
        slerp_img = slerp_img.to(DEVICE, dtype=torch.float32)
        temp_original_joints = clip_img_joints.to(DEVICE, dtype=torch.float32)
        lift_original_markers = clip_img_markers.to(DEVICE, dtype=torch.float32)

        # Forward Pass
        temp_filled_joints, lift_predicted_markers = model(slerp_img)

        # Temporal Transformer Loss
        loss_temporal = nn.MSELoss()(temp_filled_joints, temp_original_joints)

        # LiftUp Transformer Loss
        loss_liftup = nn.MSELoss()(lift_predicted_markers, lift_original_markers)

        # Combine Losses
        total_loss = loss_temporal + loss_liftup

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        epoch_loss += total_loss.item()

    avg_loss = epoch_loss / len(dataloader)
    logger.info(f"Epoch {epoch + 1} Training Loss: {avg_loss:.8f}")
    return avg_loss


def validate_combined(model, dataloader, DEVICE, save_dir=None, epoch=None):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for clip_img_joints, clip_img_markers, slerp_img in tqdm(dataloader, desc="Validating"):
            slerp_img = slerp_img.to(DEVICE, dtype=torch.float32)
            temp_original_joints = clip_img_joints.to(DEVICE, dtype=torch.float32)
            lift_original_markers = clip_img_markers.to(DEVICE, dtype=torch.float32)

            # Forward Pass
            temp_filled_joints, lift_predicted_markers = model(slerp_img)

            # Temporal Transformer Loss
            loss_temporal = nn.MSELoss()(temp_filled_joints, temp_original_joints)

            # LiftUp Transformer Loss
            loss_liftup = nn.MSELoss()(lift_predicted_markers, lift_original_markers)

            # Combine Losses
            total_loss = loss_temporal + loss_liftup
            val_loss += total_loss.item()

    avg_loss = val_loss / len(dataloader)
    return avg_loss


def main(exp_name):
    # Logging and Checkpoint Directories
    SAVE_DIR = os.path.join('posebridge_log', exp_name, 'ckpt')
    os.makedirs(SAVE_DIR, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(SAVE_DIR, f"{exp_name}.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    wandb.init(entity='edward-effendy-tokyo-tech696', project='PoseBridge', name=exp_name, mode='dryrun')

    # Initialize Dataset and Loaders
    train_dataset = MotionLoader(
        clip_seconds=CLIP_SECONDS,
        clip_fps=CLIP_FPS,
        normalize=True,
        split='train',
        markers_type=MARKERS_TYPE,
        mode=MODE,
        log_dir=SAVE_DIR,
        smplx_model_path=SMPLX_MODEL_PATH,
    )
    train_dataset.read_data(train_datasets, grab_dir)

    val_dataset = MotionLoader(
        clip_seconds=CLIP_SECONDS,
        clip_fps=CLIP_FPS,
        normalize=True,
        split='val',
        markers_type=MARKERS_TYPE,
        mode=MODE,
        log_dir=SAVE_DIR,
        smplx_model_path=SMPLX_MODEL_PATH,
    )
    val_dataset.read_data(test_datasets, grab_dir)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Initialize Models
    temporal_transformer = TemporalTransformer(
        dim_in=3,
        dim_out=3,
        dim_feat=128,
        depth=5,
        num_heads=8,
        num_joints=NUM_JOINTS,
        maxlen=CLIP_SECONDS * CLIP_FPS + 1
    ).to(DEVICE)

    liftup_transformer = LiftUpTransformer(
        input_dim=3,
        embed_dim=64,
        num_joints=NUM_JOINTS,
        num_markers=143,
        num_layers=6,
        num_heads=4,
    ).to(DEVICE)

    model = EndToEndModel(temporal_transformer, liftup_transformer).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_loss = float('inf')
    for epoch in range(NUM_EPOCHS):
        train_loss = train_combined(model, optimizer, train_loader, epoch, logger, DEVICE)
        val_loss = validate_combined(model, val_loader, DEVICE)

        wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"best_model_epoch_{epoch + 1}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoseBridge Training Script")
    parser.add_argument("--exp_name", required=True, help="Experiment name")
    args = parser.parse_args()
    main(args.exp_name)
