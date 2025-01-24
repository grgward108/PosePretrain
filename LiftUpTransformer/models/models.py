#Lift up 17 joints to 143 markers using a transformer-based model
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class LiftUpTransformer(nn.Module):
    def __init__(self, input_dim=3, embed_dim=128, num_joints=22, num_markers=143, num_layers=4, num_heads=8):
        super(LiftUpTransformer, self).__init__()

        # Joint embedding layer
        self.joint_embedding = nn.Linear(input_dim, embed_dim)

        # Relative positional encoding: (num_joints, num_joints, embed_dim)
        self.relative_position_encoding = nn.Parameter(
            torch.randn(num_joints, num_joints, embed_dim)
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Marker queries for the decoder
        self.marker_queries = nn.Parameter(torch.randn(num_markers, embed_dim))

        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)

        # Final projection to 3D space
        self.output_projection = nn.Linear(embed_dim, input_dim)

    def forward(self, joints):
        """
        Forward pass for the LiftUpTransformer.
        Args:
            joints (Tensor): Shape (batch_size * num_frames, num_joints, input_dim=3)

        Returns:
            Tensor: Predicted markers, shape (batch_size, num_markers, input_dim)
        """
        assert joints.ndim == 3, f"Expected input with 3 dimensions, but got shape {joints.shape}"
        batch_size_times_frames, num_joints, input_dim = joints.shape
        assert input_dim == 3, f"Expected last dimension to be 3 (x, y, z coordinates), but got {input_dim}"

        # Embed input joints
        joints_embedded = self.joint_embedding(joints)  # Shape: (batch_size * num_frames, num_joints, embed_dim)

        # Positional encoding and transformer encoder
        joints_embedded += self.relative_position_encoding.mean(dim=1)  # Simplified positional encoding
        encoded_joints = self.transformer_encoder(joints_embedded)  # Shape: (batch_size * num_frames, num_joints, embed_dim)

        # Cross-attention and marker prediction
        marker_queries = self.marker_queries.unsqueeze(0).expand(batch_size_times_frames, -1, -1)
        markers, _ = self.cross_attention(
            marker_queries.transpose(0, 1),
            encoded_joints.transpose(0, 1),
            encoded_joints.transpose(0, 1),
        )
        markers = markers.transpose(0, 1)  # Shape: (batch_size * num_frames, num_markers, embed_dim)

        # Project to 3D space
        markers_3d = self.output_projection(markers)  # Shape: (batch_size * num_frames, num_markers, input_dim)
        return markers_3d

