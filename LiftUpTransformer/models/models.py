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
            joints (Tensor): Shape (batch_size, num_joints, input_dim)

        Returns:
            Tensor: Predicted markers, shape (batch_size, num_markers, input_dim)
        """
        batch_size, num_joints, _ = joints.shape

        # Embed input joints
        joints_embedded = self.joint_embedding(joints)  # Shape: (batch_size, num_joints, embed_dim)

        # Add learnable relative positional encodings
        relative_pos_enc = self.relative_position_encoding.unsqueeze(0)  # Shape: (1, num_joints, num_joints, embed_dim)
        joints_embedded = joints_embedded.unsqueeze(2)  # Shape: (batch_size, num_joints, 1, embed_dim)
        joints_embedded = joints_embedded + relative_pos_enc  # Shape: (batch_size, num_joints, num_joints, embed_dim)

        # Aggregate over relative positions
        joints_embedded = joints_embedded.mean(dim=2)  # Shape: (batch_size, num_joints, embed_dim)

        # Ensure joints_embedded is 3D
        joints_embedded = joints_embedded.view(batch_size, num_joints, -1)  # Safeguard for 3D shape

        # Transformer encoder
        encoded_joints = self.transformer_encoder(joints_embedded)  # Shape: (batch_size, num_joints, embed_dim)

        # Marker queries
        marker_queries = self.marker_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attention
        markers, _ = self.cross_attention(
            marker_queries.transpose(0, 1),
            encoded_joints.transpose(0, 1),
            encoded_joints.transpose(0, 1),
        )
        markers = markers.transpose(0, 1)  # Shape: (batch_size, num_markers, embed_dim)

        # Project to 3D space
        markers_3d = self.output_projection(markers)  # Shape: (batch_size, num_markers, input_dim)
        return markers_3d
