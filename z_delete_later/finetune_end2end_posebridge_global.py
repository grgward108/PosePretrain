import torch

def restore_global_context_translation_only(clip_img_markers, traj):
    """
    Restore the markers from local to global coordinates using only translation.

    Args:
        clip_img_markers (torch.Tensor): Shape (B, 3, N, T), local marker positions.
        traj (torch.Tensor): Shape (B, 4, T+1), global trajectory.

    Returns:
        global_markers (torch.Tensor): Shape (B, 3, N, T), global marker positions.
    """
    B, C, N, T = clip_img_markers.shape  # Batch size, (x, y, z), number of markers, frames
    assert traj.shape[0] == B and traj.shape[2] == T + 1, "Trajectory dimensions do not match markers"

    # Extract global positions
    global_x = traj[:, 0, :-1]  # Shape (B, T)
    global_y = traj[:, 1, :-1]  # Shape (B, T)

    # Stack global positions into a translation vector
    global_translation = torch.stack([global_x, global_y], dim=1).unsqueeze(2)  # Shape (B, 2, 1, T)

    # Apply only translation to local marker positions
    global_markers = clip_img_markers.clone()  # Preserve the Z coordinate
    global_markers[:, :2, :, :] += global_translation  # Add global translation to (x, y)

    return global_markers


def train_combined(model, traj_model, optimizer, dataloader, epoch, logger, DEVICE):
    model.train()
    epoch_loss = 0.0
    num_batches = len(dataloader)

    # Define leg and hand indices
    leg_joint_indices = [1, 2, 4, 5, 7, 8, 10, 11]
    right_hand_indices = torch.cat([torch.arange(64, 79), torch.arange(121, 143)]).to(DEVICE)

    for clip_img_joints, clip_img_markers, slerp_img, traj, *_ in tqdm(dataloader, desc=f"Training Epoch {epoch + 1}"):
        slerp_img = slerp_img.to(DEVICE, dtype=torch.float32)
        temp_original_joints = clip_img_joints.to(DEVICE, dtype=torch.float32)
        lift_original_markers = clip_img_markers.to(DEVICE, dtype=torch.float32)

        # Forward Pass
        temp_filled_joints, lift_predicted_markers = model(slerp_img)
        predicted_traj = traj_model(traj_input)

        # Align dimensions
        temp_original_joints = temp_original_joints.permute(0, 3, 2, 1)  # [B, C, J, T]
        
        ############################ Temporal Transformer Loss ############################
        weights = torch.ones_like(temp_filled_joints).to(DEVICE)
        weights[:, :, leg_joint_indices, :] *= 3.0
        weighted_rec_loss = ((temp_filled_joints - temp_original_joints) ** 2 * weights).sum() / weights.sum()

        # Velocity Loss
        original_velocity = temp_original_joints[:, :, :, 1:] - temp_original_joints[:, :, :, :-1]
        reconstructed_velocity = temp_filled_joints[:, :, :, 1:] - temp_filled_joints[:, :, :, :-1]
        weighted_velocity_loss = ((original_velocity - reconstructed_velocity) ** 2 * weights[:, :, :, 1:]).sum() / weights[:, :, :, 1:].sum()

        # Acceleration Loss
        original_acceleration = original_velocity[:, :, :, 1:] - original_velocity[:, :, :, :-1]
        reconstructed_acceleration = reconstructed_velocity[:, :, :, 1:] - reconstructed_velocity[:, :, :, :-1]
        weighted_acceleration_loss = ((original_acceleration - reconstructed_acceleration) ** 2 * weights[:, :, :, 2:]).sum() / weights[:, :, :, 2:].sum()

        # Temporal Loss
        temporal_loss = (
            0.6 * weighted_rec_loss +
            0.3 * weighted_velocity_loss +
            0.1 * weighted_acceleration_loss
        )

        ############################ Lift-Up Transformer Loss ############################
        lift_original_markers = lift_original_markers.permute(0, 3, 2, 1)  # [B, C, M, T]
        weights = torch.ones_like(lift_predicted_markers).to(DEVICE)
        weights[:, :, right_hand_indices, :] *= 2.0
        lift_loss = ((lift_predicted_markers - lift_original_markers) ** 2 * weights).sum() / weights.sum()
        
        ############################ Global Pelvis Trajectory MLP Loss ############################
        
        global_lifted_markers = restore_global_context_translation_only(lift_predicted_markers, predicted_traj)

        # Total Loss
        total_loss = 0.6 * temporal_loss + 0.4 * lift_loss

        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Track Epoch Losses
        epoch_loss += total_loss.item()
        
    avg_epoch_loss = epoch_loss / num_batches
    logger.info(
        f"Epoch {epoch + 1}: Temporal Loss: {temporal_loss:.4f}, "
        f"Lift-Up Loss: {lift_loss:.4f}, Total Loss: {avg_epoch_loss:.4f}"
    )

    return avg_epoch_loss  # Return average loss for the epoch
