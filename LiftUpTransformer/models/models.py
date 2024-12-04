#Lift up 17 joints to 143 markers using a transformer-based model
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class LiftUpTransformer(nn.Module):
    def __init__(self, input_dim=3, embed_dim=128, num_joints=17, num_markers=143, num_layers=4, num_heads=8):
        super(LiftUpTransformer, self).__init__()
        # Linear layer to embed joints into a higher dimension
        self.joint_embedding = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(num_joints, embed_dim))
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        
        # Marker queries for the decoder
        self.marker_queries = nn.Parameter(torch.randn(num_markers, embed_dim))
        
        # Cross-attention mechanism for lifting
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
        # Input embedding + positional encoding
        batch_size = joints.size(0)
        joints_embedded = self.joint_embedding(joints) + self.positional_encoding.unsqueeze(0)  # Add batch dim

        # Transformer encoder
        encoded_joints = self.transformer_encoder(joints_embedded)  # Shape: (batch_size, num_joints, embed_dim)

        # Prepare marker queries
        marker_queries = self.marker_queries.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: (batch_size, num_markers, embed_dim)

        # Apply cross-attention
        markers, _ = self.cross_attention(
            marker_queries.transpose(0, 1),  # Query: (num_markers, batch_size, embed_dim)
            encoded_joints.transpose(0, 1),  # Key: (num_joints, batch_size, embed_dim)
            encoded_joints.transpose(0, 1)   # Value: (num_joints, batch_size, embed_dim)
        )
        markers = markers.transpose(0, 1)  # Shape: (batch_size, num_markers, embed_dim)

        # Project to 3D space
        markers_3d = self.output_projection(markers)  # Shape: (batch_size, num_markers, input_dim)
        return markers_3d
