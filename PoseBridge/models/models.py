# Combined Model
import os
import torch
import torch.nn as nn

class EndToEndModel(nn.Module):
    def __init__(self, temporal_transformer, liftup_transformer):
        super(EndToEndModel, self).__init__()
        self.temporal_transformer = temporal_transformer
        self.liftup_transformer = liftup_transformer

    def forward(self, masked_joints):
        # Forward pass through temporal transformer
        filled_joints = self.temporal_transformer(masked_joints)

        # Temporarily strip the first element (global pelvis joint)
        pelvis_joint = filled_joints[:, :, 0:1, :]  # Shape: [batch_size, num_frames, 1, coords]
        remaining_joints = filled_joints[:, :, 1:23, :]  # Shape: [batch_size, num_frames, 22, coords]

        # Reshape for liftup_transformer
        batch_size, num_frames, num_joints_subset, coords = remaining_joints.shape
        reshaped_joints = remaining_joints.view(batch_size * num_frames, num_joints_subset, coords)  # [batch_size * num_frames, 22, 3]

        # Forward pass through liftup_transformer
        markers = self.liftup_transformer(reshaped_joints)  # Output shape: [batch_size * num_frames, num_markers, 3]

        # Reshape back to [batch_size, num_frames, num_markers, 3]
        markers = markers.view(batch_size, num_frames, -1, coords)

        # Prepend the pelvis joint back to the markers
        pelvis_joint_expanded = pelvis_joint.expand(batch_size, num_frames, 1, coords)  # Match dimensions
        markers_with_pelvis = torch.cat([pelvis_joint_expanded, markers], dim=2)  # Concatenate along joint dimension

        return filled_joints, markers_with_pelvis
